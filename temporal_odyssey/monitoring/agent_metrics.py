import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from statistics import mean

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentMetrics:
    def __init__(self, window_size=100):
        self.rewards = []
        self.steps = []
        self.epsilons = []
        self.window_size = window_size
        self.rewards_window = deque(maxlen=window_size)
        self.steps_window = deque(maxlen=window_size)
        logger.info("AgentMetrics initialized.")

    def record(self, reward, steps, epsilon):
        """
        Record metrics for each episode.
        """
        self.rewards.append(reward)
        self.steps.append(steps)
        self.epsilons.append(epsilon)
        self.rewards_window.append(reward)
        self.steps_window.append(steps)
        logger.info(f"Recorded metrics - Reward: {reward}, Steps: {steps}, Epsilon: {epsilon}")

    def plot_metrics(self):
        """
        Plot the recorded metrics.
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(self.rewards, label='Rewards')
        axs[0].set_title('Rewards per Episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].legend()

        axs[1].plot(self.steps, label='Steps')
        axs[1].set_title('Steps per Episode')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        axs[1].legend()

        axs[2].plot(self.epsilons, label='Epsilon')
        axs[2].set_title('Epsilon per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Epsilon')
        axs[2].legend()

        plt.tight_layout()
        plt.show()
        logger.info("Plotted metrics.")

    def get_average_metrics(self):
        """
        Get average metrics over the specified window size.
        """
        avg_reward = mean(self.rewards_window) if self.rewards_window else 0
        avg_steps = mean(self.steps_window) if self.steps_window else 0
        logger.info(f"Average metrics - Reward: {avg_reward}, Steps: {avg_steps}")
        return avg_reward, avg_steps

    def save_metrics(self, filepath):
        """
        Save the recorded metrics to a file.
        """
        try:
            np.savez(filepath, rewards=self.rewards, steps=self.steps, epsilons=self.epsilons)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def load_metrics(self, filepath):
        """
        Load metrics from a file.
        """
        try:
            data = np.load(filepath)
            self.rewards = data['rewards'].tolist()
            self.steps = data['steps'].tolist()
            self.epsilons = data['epsilons'].tolist()
            logger.info(f"Metrics loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")

# Example usage
if __name__ == "__main__":
    agent_metrics = AgentMetrics()

    # Simulate recording metrics
    for episode in range(200):
        reward = np.random.randint(0, 100)
        steps = np.random.randint(1, 200)
        epsilon = max(0.01, np.random.rand())
        agent_metrics.record(reward, steps, epsilon)

    # Plot metrics
    agent_metrics.plot_metrics()

    # Get average metrics
    avg_reward, avg_steps = agent_metrics.get_average_metrics()
    print(f"Average Reward: {avg_reward}, Average Steps: {avg_steps}")

    # Save metrics
    agent_metrics.save_metrics('metrics.npz')

    # Load metrics
    agent_metrics.load_metrics('metrics.npz')
