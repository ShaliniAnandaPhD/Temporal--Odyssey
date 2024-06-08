import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        """
        Initialize the Prioritized Replay Buffer.

        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
            alpha (float): Priority exponent to adjust the level of prioritization.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer with maximum priority.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode is done.
        """
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta):
        """
        Sample a batch of experiences from the buffer based on their priority.

        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): Importance-sampling exponent to adjust the bias.

        Returns:
            tuple: Sampled experiences, importance weights, and sampled indices.
        """
        if len(self.buffer) == 0:
            raise ValueError("The buffer is empty.")

        priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probabilities = priorities / np.sum(priorities)

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        importance_weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        importance_weights /= np.max(importance_weights)

        states, actions, rewards, next_states, dones = zip(*samples)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), importance_weights, indices

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled experiences.

        Args:
            indices (list): List of sampled indices.
            priorities (list): List of updated priorities.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# Example usage of PrioritizedReplayBuffer
if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)

    # Simulate adding some experiences
    for _ in range(10):
        state = np.random.rand(4)
        action = np.random.randint(4)
        reward = np.random.rand()
        next_state = np.random.rand(4)
        done = np.random.randint(2)
        buffer.add(state, action, reward, next_state, done)

    # Simulate sampling from the buffer
    states, actions, rewards, next_states, dones, weights, indices = buffer.sample(batch_size=4, beta=0.4)
    print("Sampled states:", states)
    print("Importance weights:", weights)

    # Simulate updating priorities
    new_priorities = np.random.rand(4)
    buffer.update_priorities(indices, new_priorities)
    print("Updated priorities:", [buffer.priorities[idx] for idx in indices])

