"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_space,
                 init_std,
                 hidden_layers,
                 activation_function,
                 action_low,
                 action_high,
                 normalise_actions,
                 min_std=1e-6,
                 use_task_encoder=False,
                 state_embed_dim=None,
                 task_dim=0,
                 latent_dim=0,
                 ):
        super(Policy, self).__init__()

        hidden_layers = [int(h) for h in hidden_layers]
        curr_input_dim = state_dim

        # output distributions of the policy
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(hidden_layers[-1], num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(hidden_layers[-1], num_outputs, init_std, min_std,
                                     action_low=action_low, action_high=action_high,
                                     normalise_actions=normalise_actions)
        else:
            raise NotImplementedError

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        # initialise task encoder (for the oracle)

        self.use_task_encoder = use_task_encoder
        self.task_dim = task_dim
        self.latent_dim = latent_dim
        if self.use_task_encoder:
            self.task_encoder = utl.FeatureExtractor(self.task_dim, self.latent_dim, self.activation_function)
            self.state_encoder = utl.FeatureExtractor(state_dim - self.task_dim, state_embed_dim,
                                                      self.activation_function)
            curr_input_dim = state_embed_dim + latent_dim

        # initialise actor and critic

        init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.actor_layers.append(fc)

            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.critic_layers.append(fc)
            curr_input_dim = hidden_layers[i]

        self.critic_linear = nn.Linear(hidden_layers[-1], 1)

    def get_actor_params(self):
        return [*self.actor.parameters(), *self.dist.parameters()]

    def get_critic_params(self):
        return [*self.critic.parameters(), *self.critic_linear.parameters()]

    def forward_actor(self, inputs):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward_critic(self, inputs):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward(self, inputs):

        if self.use_task_encoder:
            state_embedding = self.state_encoder(inputs[:, :-self.task_dim])
            latent_state = self.task_encoder(inputs[:, -self.task_dim:])
            inputs = torch.cat((state_embedding, latent_state), dim=1)

        # forward through critic/actor part
        hidden_critic = self.forward_critic(inputs)
        hidden_actor = self.forward_actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, inputs, deterministic=False):
        value, actor_features = self.forward(inputs)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.forward(inputs)
        return value

    def evaluate_actions(self, inputs, action, return_action_mean=False):
        value, actor_features = self.forward(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if not return_action_mean:
            return value, action_log_probs, dist_entropy
        else:
            return value, action_log_probs, dist_entropy, dist.mode(), dist.stddev


FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, min_std,
                 action_low, action_high, normalise_actions):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std))
        self.min_std = torch.tensor([min_std]).to(device)

        # whether or not to conform to the action space given by the env
        # (scale / squash actions that the network outpus)
        self.normalise_actions = normalise_actions
        if len(np.unique(action_low)) == 1 and len(np.unique(action_high)) == 1:
            self.unique_action_limits = True
        else:
            self.unique_action_limits = False

        self.action_low = torch.from_numpy(action_low).to(device)
        self.action_high = torch.from_numpy(action_high).to(device)

    def forward(self, x):
        action_mean = self.fc_mean(x)
        if self.normalise_actions:
            if self.unique_action_limits and torch.unique(self.action_low) == -1 and torch.unique(
                    self.action_high) == 1:
                action_mean = torch.tanh(action_mean)
            else:
                # TODO: this isn't tested
                action_mean = torch.sigmoid(action_mean) * (self.action_high - self.action_low) + self.action_low
        std = torch.max(self.min_std, self.logstd.exp())
        return FixedNormal(action_mean, std)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().reshape(1, -1)
        else:
            bias = self._bias.t().reshape(1, -1, 1, 1)

        return x + bias
