import numpy as np
import logging
from sklearn.neighbors import KDTree

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransferLearning:
    def __init__(self):
        """
        Initialize the TransferLearning class with knowledge storage and a KD-tree for efficient retrieval.
        """
        self.knowledge = []
        self.kd_tree = None
        logger.info("TransferLearning initialized.")

    def store_knowledge(self, state, action, reward, next_state):
        """
        Store knowledge consisting of state, action, reward, and next state.

        Parameters:
        state (np.array): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (np.array): The next state after taking the action.
        """
        self.knowledge.append((state, action, reward, next_state))
        self._update_kd_tree()
        logger.info("Knowledge stored and KD-tree updated.")

    def _update_kd_tree(self):
        """
        Update the KD-tree with the current knowledge.
        """
        if self.knowledge:
            states = np.array([k[0] for k in self.knowledge])
            self.kd_tree = KDTree(states)
            logger.info("KD-tree updated with current knowledge.")

    def apply(self, state, action, reward, next_state):
        """
        Modify the reward and next state based on transferred knowledge.

        Parameters:
        state (np.array): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (np.array): The next state after taking the action.

        Returns:
        tuple: Modified reward and next state.
        """
        if self.kd_tree is None:
            logger.warning("No knowledge available for transfer.")
            return reward, next_state

        # Retrieve the nearest knowledge entry
        dist, idx = self.kd_tree.query([state], k=1)
        nearest_knowledge = self.knowledge[idx[0][0]]
        logger.info(f"Nearest knowledge retrieved with distance {dist[0][0]}.")

        # Modify the reward and next state based on the retrieved knowledge
        transferred_reward = self._modify_reward(nearest_knowledge, reward)
        transferred_next_state = self._modify_next_state(nearest_knowledge, next_state)

        return transferred_reward, transferred_next_state

    def _modify_reward(self, knowledge, current_reward):
        """
        Modify the reward based on the transferred knowledge.

        Parameters:
        knowledge (tuple): The retrieved knowledge entry.
        current_reward (float): The current reward.

        Returns:
        float: The modified reward.
        """
        _, _, stored_reward, _ = knowledge
        modified_reward = (stored_reward + current_reward) / 2  # Example: Simple averaging
        logger.info(f"Reward modified from {current_reward} to {modified_reward}.")
        return modified_reward

    def _modify_next_state(self, knowledge, current_next_state):
        """
        Modify the next state based on the transferred knowledge.

        Parameters:
        knowledge (tuple): The retrieved knowledge entry.
        current_next_state (np.array): The current next state.

        Returns:
        np.array: The modified next state.
        """
        _, _, _, stored_next_state = knowledge
        modified_next_state = (stored_next_state + current_next_state) / 2  # Example: Simple averaging
        logger.info("Next state modified based on transferred knowledge.")
        return modified_next_state

# Example usage
if __name__ == "__main__":
    transfer_learning = TransferLearning()

    # Store some example knowledge
    state = np.array([0.1, 0.2, 0.3])
    action = 1
    reward = 10
    next_state = np.array([0.2, 0.3, 0.4])
    transfer_learning.store_knowledge(state, action, reward, next_state)

    # Apply transfer learning to modify reward and next state
    new_state = np.array([0.15, 0.25, 0.35])
    new_action = 1
    new_reward = 15
    new_next_state = np.array([0.25, 0.35, 0.45])
    modified_reward, modified_next_state = transfer_learning.apply(new_state, new_action, new_reward, new_next_state)

    print(f"Modified reward: {modified_reward}")
    print(f"Modified next state: {modified_next_state}")

