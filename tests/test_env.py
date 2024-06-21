import unittest
import numpy as np
from temporal_odyssey.envs.time_travel_env import TimeTravelEnv

class TestTimeTravelEnv(unittest.TestCase):
    
    def setUp(self):
        # Set up the environment for testing.
        # This method is called before each test case to ensure a fresh environment instance.
        self.env = TimeTravelEnv()
        self.env.reset()

    def test_initial_state(self):
        # Test if the initial state of the environment is set correctly.
        # This ensures that the environment starts in the expected configuration.
        initial_observation = self.env._get_observation()
        expected_observation = np.zeros((10, 10, 3), dtype=np.uint8)
        expected_observation[0, 0] = [255, 0, 0]  # Agent's initial position in red
        np.testing.assert_array_equal(initial_observation, expected_observation,
                                      "Initial state observation does not match expected state.")
    
    def test_reset(self):
        # Test if the environment resets correctly.
        # This ensures that after taking an action, resetting the environment
        # returns it to its initial state.
        self.env.step(1)  # Take a step to change the state
        observation_after_reset = self.env.reset()
        expected_observation = np.zeros((10, 10, 3), dtype=np.uint8)
        expected_observation[0, 0] = [255, 0, 0]  # Agent's initial position in red
        np.testing.assert_array_equal(observation_after_reset, expected_observation,
                                      "Environment did not reset correctly.")
    
    def test_step(self):
        # Test if the step function updates the state correctly.
        # This checks if taking an action (moving down) results in the expected
        # state change, reward, and done flag.
        observation, reward, done, info = self.env.step(1)  # Move down
        expected_observation = np.zeros((10, 10, 3), dtype=np.uint8)
        expected_observation[1, 0] = [255, 0, 0]  # Agent's new position after moving down
        np.testing.assert_array_equal(observation, expected_observation,
                                      "Step function did not update the state correctly.")
        self.assertFalse(done, "Episode ended prematurely.")
        self.assertEqual(reward, -0.1, "Reward calculation is incorrect.")
    
    def test_time_travel(self):
        # Test if the time travel action updates the time period correctly.
        # This ensures that the time travel mechanic cycles through time periods as expected.
        self.env.step(4)  # Perform time travel action
        self.assertEqual(self.env.current_time_period, 1, "Time travel did not update the time period correctly.")
        self.env.step(4)  # Perform time travel action again
        self.assertEqual(self.env.current_time_period, 2, "Time travel did not update the time period correctly.")
        self.env.step(4)  # Perform time travel action again
        self.assertEqual(self.env.current_time_period, 0, "Time travel did not cycle back to the initial time period.")
    
    def test_interact(self):
        # Test if the interaction with objects works correctly.
        # This checks if the agent can interact with objects in the environment
        # and if the interaction is registered correctly.
        self.env.current_position = (2, 3)  # Move to position of the tree
        observation, reward, done, info = self.env.step(5)  # Interact with the tree
        # Since interaction outputs to console, we will manually check logs for the correct interaction
        # Potential improvement: Modify interact method to return a message or state change
        self.assertIn((2, 3), self.env.interactive_objects, "Tree object is not present in the environment.")
    
    def test_done_condition(self):
        # Test if the done condition is met correctly.
        # This ensures that the episode ends when the agent reaches the goal state
        # and that the correct reward is given.
        self.env.current_position = (9, 9)  # Move to the bottom-right corner
        observation, reward, done, info = self.env.step(0)  # Any action to trigger step
        self.assertTrue(done, "Episode did not end when the agent reached the bottom-right corner.")
        self.assertEqual(reward, 10, "Reward for reaching the goal position is incorrect.")
    
    def test_invalid_action(self):
        # Test if the environment handles invalid actions correctly.
        # This ensures that the environment raises an appropriate error when
        # an invalid action is attempted.
        with self.assertRaises(ValueError):
            self.env.step(6)  # Invalid action

    def test_out_of_bounds_movement(self):
        # Test the environment's behavior when the agent tries to move out of bounds.
        # This ensures that the agent cannot leave the defined grid space.
        self.env.current_position = (0, 0)
        observation, reward, done, info = self.env.step(3)  # Try to move left
        self.assertEqual(self.env.current_position, (0, 0), "Agent should not move out of bounds")

    def test_reward_consistency(self):
        # Test if rewards are consistent across different non-goal scenarios.
        # This ensures that the reward function behaves consistently for similar actions.
        
        # Move to a non-goal position
        self.env.current_position = (5, 5)
        _, reward1, _, _ = self.env.step(0)
        
        # Move to another non-goal position
        self.env.current_position = (3, 3)
        _, reward2, _, _ = self.env.step(0)
        
        self.assertEqual(reward1, reward2, "Rewards should be consistent for non-goal movements")

    def test_state_representation(self):
        # Test if the state representation accurately reflects the environment's current state.
        # This ensures that the observation returned by the environment correctly
        # represents the agent's position and other relevant features.
        self.env.current_position = (2, 2)
        self.env.current_time_period = 1
        observation = self.env._get_observation()
        
        expected_observation = np.zeros((10, 10, 3), dtype=np.uint8)
        expected_observation[2, 2] = [255, 0, 0]  # Agent's position
        # Add any other expected features for the current time period
        
        np.testing.assert_array_equal(observation, expected_observation,
                                      "State representation does not match expected state")

    def test_time_period_effects(self):
        # Test if different time periods have distinct effects on the environment.
        # This ensures that time travel meaningfully changes the environment state.
        initial_objects = set(self.env.interactive_objects.keys())
        self.env.step(4)  # Time travel
        new_objects = set(self.env.interactive_objects.keys())
        
        self.assertNotEqual(initial_objects, new_objects, 
                            "Time travel should change the interactive objects in the environment")

    def test_long_episode(self):
        # Test if the environment behaves correctly over a long sequence of actions.
        # This ensures the stability of the environment over extended use.
        for _ in range(100):
            _, _, done, _ = self.env.step(self.env.action_space.sample())
            if done:
                self.env.reset()
        # If this completes without errors, the test passes
        self.assertTrue(True, "Environment should handle long episodes without errors")

    def test_determinism(self):
        # Test if the environment behaves deterministically given the same seed.
        # This ensures that the environment produces consistent results for reproducibility.
        env1 = TimeTravelEnv(seed=42)
        env2 = TimeTravelEnv(seed=42)
        
        for _ in range(10):
            action = env1.action_space.sample()
            obs1, reward1, done1, _ = env1.step(action)
            obs2, reward2, done2, _ = env2.step(action)
            
            np.testing.assert_array_equal(obs1, obs2, "Observations should be identical for the same seed and actions")
            self.assertEqual(reward1, reward2, "Rewards should be identical for the same seed and actions")
            self.assertEqual(done1, done2, "Done flags should be identical for the same seed and actions")

if __name__ == '__main__':
    unittest.main()
