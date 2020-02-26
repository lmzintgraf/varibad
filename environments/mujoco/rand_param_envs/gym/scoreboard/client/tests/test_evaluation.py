from environments.mujoco.rand_param_envs.gym import scoreboard
from environments.mujoco.rand_param_envs.gym.scoreboard.client.tests import helper


class EvaluationTest(helper.APITestCase):
    def test_create_evaluation(self):
        self.mock_response(helper.TestData.evaluation_response())

        evaluation = scoreboard.Evaluation.create()
        assert isinstance(evaluation, scoreboard.Evaluation)

        self.requestor_mock.request.assert_called_with(
            'post',
            '/v1/evaluations',
            {},
            None
        )
