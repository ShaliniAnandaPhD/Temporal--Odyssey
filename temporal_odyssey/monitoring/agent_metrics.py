import tensorflow as tf
import numpy as np
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentMetrics:
    def __init__(self, log_dir="logs/"):
        """
        Initialize the AgentMetrics class.
        
        Args:
            log_dir (str): Directory to save TensorBoard logs.
        """
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.rewards = deque(maxlen=100)  # Store rewards for the last 100 episodes
        self.losses = deque(maxlen=100)   # Store losses for the last 100 episodes
        self.steps = 0  # Track the number of training steps
        logger.info("AgentMetrics initialized with log directory: %s", log_dir)

    def log_metrics(self, episode, reward, loss):
        """
        Log custom metrics to TensorBoard.
        
        Args:
            episode (int): The current episode number.
            reward (float): The reward obtained in the current episode.
            loss (float): The loss obtained in the current episode.
        """
        self.rewards.append(reward)
        self.losses.append(loss)
        avg_reward = np.mean(self.rewards)
        avg_loss = np.mean(self.losses)
        self.steps += 1
        
        with self.writer.as_default():
            tf.summary.scalar('Episode Reward', reward, step=episode)
            tf.summary.scalar('Average Reward', avg_reward, step=episode)
            tf.summary.scalar('Loss', loss, step=self.steps)
            tf.summary.scalar('Average Loss', avg_loss, step=self.steps)
        
        logger.info("Logged metrics for episode %d: reward=%.2f, avg_reward=%.2f, loss=%.2f, avg_loss=%.2f",
                    episode, reward, avg_reward, loss, avg_loss)

    def calculate_custom_metrics(self, rewards, losses):
        """
        Calculate and return custom metrics.
        
        Args:
            rewards (list): List of rewards.
            losses (list): List of losses.
        
        Returns:
            dict: A dictionary containing custom metrics.
        """
        avg_reward = np.mean(rewards)
        avg_loss = np.mean(losses)
        max_reward = np.max(rewards)
        min_loss = np.min(losses)
        
        metrics = {
            'average_reward': avg_reward,
            'average_loss': avg_loss,
            'max_reward': max_reward,
            'min_loss': min_loss,
        }
        
        logger.info("Calculated custom metrics: %s", metrics)
        return metrics

# Example usage within the agent's training loop
if __name__ == "__main__":
    import gym
    from hybrid_learning import HybridLearningAgent  # Ensure you import the agent correctly

    env = gym.make('CartPole-v1')
    agent = HybridLearningAgent(env)
    metrics = AgentMetrics()
    
    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            loss = agent.train_step(state, action, reward, next_state, done)
            total_loss += loss
            state = next_state
            steps += 1
        
        avg_loss = total_loss / steps if steps > 0 else total_loss
        metrics.log_metrics(episode, total_reward, avg_loss)

