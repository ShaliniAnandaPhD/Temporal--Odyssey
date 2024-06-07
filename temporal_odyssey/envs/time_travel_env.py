
import gym
from gym import spaces
import numpy as np

class TimeTravelEnv(gym.Env):
    def __init__(self):
        super(TimeTravelEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # 4 directions + time travel
        self.observation_space = spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)  # 10x10 RGB image

        # Initialize environment state
        self.current_time_period = 0
        self.current_position = (0, 0)

    def reset(self):
        # Reset environment to initial state
        self.current_time_period = 0
        self.current_position = (0, 0)
        return self._get_observation()

    def step(self, action):
        # Perform action and update environment state
        self._take_action(action)
        
        # Calculate reward based on action and current state
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Get observation of current state
        observation = self._get_observation()
        
        return observation, reward, done, {}

    def _take_action(self, action):
        if action < 4:  # Movement actions
            self._move(action)
        elif action == 4:  # Time travel action
            self._time_travel()

    def _move(self, action):
        # Example movement logic (0: up, 1: down, 2: left, 3: right)
        x, y = self.current_position
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < 9:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < 9:
            y += 1
        self.current_position = (x, y)

    def _time_travel(self):
        # Example time travel logic
        self.current_time_period = (self.current_time_period + 1) % 3  # Cycle through 3 time periods

    def _get_observation(self):
        # Example observation logic
        observation = np.zeros((10, 10, 3), dtype=np.uint8)
        x, y = self.current_position
        observation[x, y] = [255, 0, 0]  # Mark the agent's position in red
        return observation

    def _calculate_reward(self):
        # Example reward calculation
        if self.current_position == (9, 9):  # Reward for reaching bottom-right corner
            return 10
        return -0.1  # Small penalty for each step to encourage efficiency

    def _is_done(self):
        # Example done condition
        if self.current_position == (9, 9):  # Episode ends when agent reaches bottom-right corner
            return True
        return False
