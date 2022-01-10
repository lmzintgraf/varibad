# VariBAD

Code for the paper "[VariBAD: A very good method for Bayes-Adaptive Deep RL via Meta-Learning](https://arxiv.org/abs/1910.08348)" - 
Luisa Zintgraf, Kyriacos Shiarlis, Maximilian Igl, Sebastian Schulze, 
Yarin Gal, Katja Hofmann, Shimon Whiteson, published at ICLR 2020.

```
@inproceedings{zintgraf2020varibad,
  title={VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning},
  author={Zintgraf, Luisa and Shiarlis, Kyriacos and Igl, Maximilian and Schulze, Sebastian and Gal, Yarin and Hofmann, Katja and Whiteson, Shimon},
  booktitle={International Conference on Learning Representation (ICLR)},
  year={2020}}
```

> ! Important !
> 
> If you use this code with your own environments, 
> make sure to not use `np.random` in them 
> (e.g. to generate the tasks) because it is not thread safe 
> (and it may cause duplicates across threads).
> Instead, use the python native random function. 
> For an example see
> [here](https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/ant_goal.py#L38).

### Requirements

We use PyTorch for this code, and log results using TensorboardX.

The main requirements can be found in `requirements.txt`. 

For the MuJoCo experiments, you need to install MuJoCo.
Make sure you have the right MuJoCo version:
- For the Cheetah and Ant environments, use `mujoco150`. 
(You can also use `mujoco200` except for AntGoal, 
because there's a bug which leads to 80% of the env state being zero).
- For Walker/Hopper, use `mujoco131`.

For `mujoco131`, use: `gym==0.9.1 gym[mujoco]==0.9.1 mujoco-py==0.5.7`

### Overview

The main training loop for VariBAD can be found in `metalearner.py`,
the models are in `models/`, the VAE set-up and losses are in `vae.py` and the RL algorithms in `algorithms/`.

There's quite a bit of documentation in the respective scripts so have a look there for details.

### Running an experiment

To evaluate variBAD on the gridworld from the paper, run

`python main.py --env-type gridworld_varibad`

which will use hyperparameters from `config/gridworld/args_grid_varibad.py`. 

To run variBAD on the MuJoCo experiments use:
```
python main.py --env-type cheetah_dir_varibad
python main.py --env-type cheetah_vel_varibad
python main.py --env-type ant_dir_varibad
python main.py --env-type walker_varibad
```

You can also run RL2 and the Oracle, just replace `varibad` above with the respective string. 
See `main.py` for all options.

The results will by default be saved at `./logs`, 
but you can also pass a flag with an alternative directory using `--results_log_dir /path/to/dir`.

The default configs are in the `config/` folder. 
You can overwrite any default hyperparameters using command line arguments.

Results will be written to tensorboard event files, 
and some visualisations will be printed every now and then.

### Configs

Some comments on the flags in the config files:
- You can choose what type of decoder you by setting the respective flags to true: 
`--decode_reward True` and/or `--decode_state True`.
- You can also choose a task decoder (`--decode_task True`), which was proposed by 
[Humplik et al. (2019)](https://arxiv.org/abs/1905.06424). 
This method uses privileged information during meta-training (e.g., the task description or ID)
to learn the posterior distribution in a supervised way. 
(Note that our implementation is based on the variBAD architecture, 
so differs slightly from theirs.)
- The size of the latent dimension can be changed using `--latent_dim`.
- In our experience, the performance of PPO depends a lot on 
the number of minibatches (`--ppo_num_minibatch`),
the number of epochs (`ppo_num_epochs`),
and the batchsize (change with `--policy_num_steps` and/or `--num_processes`).
Another important parameter is the weight of the kl term (`--kl_weight`) in the ELBO.

### Results 

The MuJoCo results (smoothened learning curves) and a script to plot them 
can be found [here](https://www.dropbox.com/sh/1bi7er3j67ylrkb/AADmgWwi4kbTwVNev3NQP_11a).
### Comments

- When the flag `disable_metalearner` is activated, the file `learner.py` will be used instead of `metalearner.py`. 
This is a stripped down version without encoder, decoder, stochastic latent variables, etc. 
It can be used to train (belief) oracles or policies that are good on average.
- Currently, the VAE never looks at the starting state, but the prior is independent
of where the agent starts. It was easier to implement like this. 
Since actions/rewards aren't available at the first time step, 
another option would be to just fill them with zeros.
- I added an example environment with empty methods in `environments/example_env.py`,
if you want to know what an environment should do.
