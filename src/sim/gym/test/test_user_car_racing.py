import unittest

import numpy as np

from src.sim.gym.gym_environment import environment_setup, setup_user_interaction


class CarracingTestCase(unittest.TestCase):

    def test_user(self):
        env = environment_setup()
        env, action = setup_user_interaction(env)
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        actions = []
        for step in range(100):
            s, r, done, info = env.step(action)
            actions.append(action)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            env.render()
            if done or restart:
                break
        env.close()
        self.assertTrue(np.sum(np.asarray(actions)) != 0)


if __name__ == '__main__':
    unittest.main()
