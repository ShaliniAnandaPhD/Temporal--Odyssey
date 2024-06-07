import unittest
import numpy as np
from temporal_odyssey.envs.time_travel_env import TimeTravelEnv

class TestTimeTravelEnv(unittest.TestCase):
    
    def setUp(self):
        """
        Set up the environment for testing.
        This method is called before each test case.
        """
        self.env = TimeTravelEnv()
        self.env.reset()

    def test_initial_state(self):
        """
        Test if the initial state of the environment is set correctly.
        """
        initial_observation = self.env._get_observation()
        expected_observation = np.zeros((10, 10, 3), dtype=np.uint8)
        expected_observation[0, 0] = [255, 0, 0]  # Agent's initial position in red
        np.testing.assert_array_equal(initial_observation, expected_observation,
                                      "Initial state observation does not match expected state.")
    
    def test_reset(self):
        """
        Test if the environment resets correctly.
        """
        self.env.step(1)  # Take a step to change the state
        observation_after_reset = self.env.reset()
        expected_observation = np.zeros((10, 10, 3), dtype=np.uint8)
        expected_observation[0, 0] = [255, 0, 0]  # Agent's initial position in red
        np.testing.assert_array_equal(observation_after_reset, expected_observation,
                                      "Environment did not reset correctly.")
    
    def test_step(self):
        """
        Test if the step function updates the state correctly.
        """
        observation, reward, done, info = self.env.step(1)  # Move down
        expected_observation = np.zeros((10, 10, 3), dtype=np.uint8)
        expected_observation[1, 0] = [255, 0, 0]  # Agent's new position after moving down
        np.testing.assert_array_equal(observation, expected_observation,
                                      "Step function did not update the state correctly.")
        self.assertFalse(done, "Episode ended prematurely.")
        self.assertEqual(reward, -0.1, "Reward calculation is incorrect.")
    
    def test_time_travel(self):
        """
        Test if the time travel action updates the time period correctly.
        """
        self.env.step(4)  # Perform time travel action
        self.assertEqual(self.env.current_time_period, 1, "Time travel did not update the time period correctly.")
        self.env.step(4)  # Perform time travel action again
        self.assertEqual(self.env.current_time_period, 2, "Time travel did not update the time period correctly.")
        self.env.step(4)  # Perform time travel action again
        self.assertEqual(self.env.current_time_period, 0, "Time travel did not cycle back to the initial time period.")
    
    def test_interact(self):
        """
        Test if the interaction with objects works correctly.
        """
        self.env.current_position = (2, 3)  # Move to position of the tree
        observation, reward, done, info = self.env.step(5)  # Interact with the tree
        # Since interaction outputs to console, we will manually check logs for the correct interaction
        # Potential solution: Modify interact method to return a message or state change
        self.assertIn((2, 3), self.env.interactive_objects, "Tree object is not present in the environment.")
    
    def test_done_condition(self):
        """
        Test if the done condition is met correctly.
        """
        self.env.current_position = (9, 9)  # Move to the bottom-right corner
        observation, reward, done, info = self.env.step(0)  # Any action to trigger step
        self.assertTrue(done, "Episode did not end when the agent reached the bottom-right corner.")
        self.assertEqual(reward, 10, "Reward for reaching the goal position is incorrect.")
    
    def test_invalid_action(self):
        """
        Test if the environment handles invalid actions correctly.
        """
        with self.assertRaises(ValueError):
            self.env.step(6)  # Invalid action

if __name__ == '__main__':
    unittest.main()

