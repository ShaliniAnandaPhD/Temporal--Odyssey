# temporal_odyssey/envs/time_travel_env.py

import gym
from gym import spaces
import numpy as np
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeTravelEnv(gym.Env):
    def __init__(self):
        super(TimeTravelEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(6)  # 4 directions + time travel + interact
        self.observation_space = spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)  # 10x10 RGB image

        # Initialize environment state
        self.current_time_period = 0
        self.current_position = (0, 0)
        self.interactive_objects = self._initialize_interactive_objects()

        # Visualization setup
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self._get_observation())

        logger.info("TimeTravelEnv initialized.")

    def _initialize_interactive_objects(self):
        """
        Initialize the interactive objects in the environment.
        """
        # Example interactive objects
        return {
            (2, 3): "tree",
            (5, 5): "rock",
            (7, 8): "river"
        }

    def reset(self):
        """
        Reset environment to initial state.
        """
        self.current_time_period = 0
        self.current_position = (0, 0)
        logger.info("Environment reset.")
        return self._get_observation()

    def step(self, action):
        """
        Perform action and update environment state.
        """
        self._take_action(action)
        
        # Calculate reward based on action and current state
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self._is_done()
        
        # Get observation of current state
        observation = self._get_observation()

        # Update visualization
        self.img.set_data(observation)
        self.ax.set_title(f"Action: {action}, Reward: {reward}, Done: {done}")
        plt.pause(0.1)

        return observation, reward, done, {}

    def _take_action(self, action):
        if action < 4:  # Movement actions
            self._move(action)
        elif action == 4:  # Time travel action
            self._time_travel()
        elif action == 5:  # Interact action
            self._interact()

    def _move(self, action):
        """
        Example movement logic (0: up, 1: down, 2: left, 3: right).
        """
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
        logger.info(f"Moved to position: {self.current_position}")

    def _time_travel(self):
        """
        Example time travel logic.
        """
        self.current_time_period = (self.current_time_period + 1) % 3  # Cycle through 3 time periods
        logger.info(f"Traveled to time period: {self.current_time_period}")

    def _interact(self):
        """
        Interact with objects at the current position.
        """
        obj = self.interactive_objects.get(self.current_position)
        if obj:
            logger.info(f"Interacting with {obj} at position {self.current_position}")
            # Implement interaction logic based on object type
            if obj == "tree":
                self._chop_tree()
            elif obj == "rock":
                self._mine_rock()
            elif obj == "river":
                self._fetch_water()
        else:
            logger.info("Nothing to interact with here.")

    def _chop_tree(self):
        """
        Example interaction: Chop down a tree.
        """
        logger.info("Chopped down a tree.")
        # Implement logic for chopping down a tree
        # e.g., add wood to inventory, remove tree from environment

    def _mine_rock(self):
        """
        Example interaction: Mine a rock.
        """
        logger.info("Mined a rock.")
        # Implement logic for mining a rock
        # e.g., add stone to inventory, remove rock from environment

    def _fetch_water(self):
        """
        Example interaction: Fetch water from a river.
        """
        logger.info("Fetched water from the river.")
        # Implement logic for fetching water
        # e.g., add water to inventory

    def _get_observation(self):
        """
        Example observation logic.
        """
        observation = np.zeros((10, 10, 3), dtype=np.uint8)
        x, y = self.current_position
        observation[x, y] = [255, 0, 0]  # Mark the agent's position in red
        for (obj_x, obj_y), obj_type in self.interactive_objects.items():
            if obj_type == "tree":
                observation[obj_x, obj_y] = [0, 255, 0]  # Tree in green
            elif obj_type == "rock":
                observation[obj_x, obj_y] = [128, 128, 128]  # Rock in gray
            elif obj_type == "river":
                observation[obj_x, obj_y] = [0, 0, 255]  # River in blue
        return observation

    def _calculate_reward(self, action):
        """
        Example reward calculation.
        """
        if self.current_position == (9, 9):  # Reward for reaching bottom-right corner
            return 10
        return -0.1  # Small penalty for each step to encourage efficiency

    def _is_done(self):
        """
        Example done condition.
        """
        if self.current_position == (9, 9):  # Episode ends when agent reaches bottom-right corner
            return True
        return False

# Example usage
if __name__ == "__main__":
    env = TimeTravelEnv()
    observation = env.reset()
    done = False

    plt.ion()  # Enable interactive mode for live plot updates

    while not done:
        action = env.action_space.sample()  # Random action for demonstration
        observation, reward, done, info = env.step(action)
        logger.info(f"Action: {action}, Reward: {reward}, Done: {done}")

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep the plot open after the loop ends
