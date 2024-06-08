import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaLearning:
    def __init__(self):
        self.meta_knowledge = {}
        self.learning_rate_history = {}
        self.exploration_rate_history = {}
        self.architecture_history = {}

    def store_meta_knowledge(self, agent_class, state, action, reward, next_state, done):
        """Stores the experience tuple in the meta-knowledge repository for the given agent class."""
        if agent_class not in self.meta_knowledge:
            self.meta_knowledge[agent_class] = []
        self.meta_knowledge[agent_class].append((state, action, reward, next_state, done))

    def retrieve_meta_knowledge(self, agent_class):
        """Retrieves stored meta-knowledge for the given agent class."""
        return self.meta_knowledge.get(agent_class, [])

    def update(self, agent, state, action, reward, next_state, done):
        """Applies meta-learning by leveraging meta-knowledge to update the agent's learning process."""
        agent_class = type(agent).__name__
        self.store_meta_knowledge(agent_class, state, action, reward, next_state, done)
        
        meta_knowledge = self.retrieve_meta_knowledge(agent_class)
        if meta_knowledge:
            try:
                # Apply meta-learning logic
                self._update_learning_rate(agent, meta_knowledge)
                self._update_exploration(agent, meta_knowledge)
                self._update_model_architecture(agent, meta_knowledge)
            except Exception as e:
                logger.error(f"Error updating agent: {e}")

    def _update_learning_rate(self, agent, meta_knowledge):
        """Dynamically updates the agent's learning rate based on meta-knowledge."""
        avg_reward = np.mean([k[2] for k in meta_knowledge])
        if avg_reward < 1.0:
            new_learning_rate = agent.learning_rate * 0.9
        else:
            new_learning_rate = agent.learning_rate * 1.1
        agent.learning_rate = max(min(new_learning_rate, 0.01), 0.00001)

        agent_class = type(agent).__name__
        self.learning_rate_history.setdefault(agent_class, []).append(agent.learning_rate)
        logger.info(f"Updated learning rate to: {agent.learning_rate}")

    def _update_exploration(self, agent, meta_knowledge):
        """Dynamically updates the agent's exploration strategy based on meta-knowledge."""
        avg_done = np.mean([k[4] for k in meta_knowledge])
        if avg_done > 0.5:
            new_epsilon = agent.epsilon * 0.9
        else:
            new_epsilon = agent.epsilon * 1.1
        agent.epsilon = max(min(new_epsilon, 1.0), 0.01)

        agent_class = type(agent).__name__
        self.exploration_rate_history.setdefault(agent_class, []).append(agent.epsilon)
        logger.info(f"Updated exploration rate to: {agent.epsilon}")

    def _update_model_architecture(self, agent, meta_knowledge):
        """Dynamically updates the agent's model architecture based on meta-knowledge."""
        avg_reward = np.mean([k[2] for k in meta_knowledge])
        if avg_reward < 1.0:
            agent.model.add(tf.keras.layers.Dense(64, activation='relu'))
        else:
            agent.model.pop()

        agent_class = type(agent).__name__
        self.architecture_history.setdefault(agent_class, []).append(str(agent.model.layers))
        logger.info(f"Updated model architecture to: {agent.model.summary()}")

    def plot_meta_learning_metrics(self, agent_class):
        """Plots meta-learning metrics such as learning rate, exploration rate, and architecture changes."""
        if agent_class in self.learning_rate_history:
            plt.figure(figsize=(12, 6))
            plt.subplot(3, 1, 1)
            plt.plot(self.learning_rate_history[agent_class])
            plt.title(f'Learning Rate History for {agent_class}')
            plt.xlabel('Update Step')
            plt.ylabel('Learning Rate')

            plt.subplot(3, 1, 2)
            plt.plot(self.exploration_rate_history[agent_class])
            plt.title(f'Exploration Rate History for {agent_class}')
            plt.xlabel('Update Step')
            plt.ylabel('Exploration Rate')

            plt.subplot(3, 1, 3)
            plt.plot([len(layers.split(',')) for layers in self.architecture_history[agent_class]])
            plt.title(f'Model Architecture Changes for {agent_class}')
            plt.xlabel('Update Step')
            plt.ylabel('Number of Layers')

            plt.tight_layout()
            plt.show()

    def save_meta_knowledge(self, filepath):
        """Saves the meta-knowledge repository to a file."""
        try:
            np.save(filepath, self.meta_knowledge)
            logger.info(f"Meta-knowledge saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save meta-knowledge: {e}")

    def load_meta_knowledge(self, filepath):
        """Loads the meta-knowledge repository from a file."""
        try:
            self.meta_knowledge = np.load(filepath, allow_pickle=True).item()
            logger.info(f"Meta-knowledge loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load meta-knowledge: {e}")

# Example agent class for testing
class Agent:
    def __init__(self):
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

# Example usage
if __name__ == "__main__":
    meta_learning = MetaLearning()

    # Simulated example data
    state = np.random.rand(10)
    action = 1
    reward = 1.0
    next_state = np.random.rand(10)
    done = False
    
    agent = Agent()
    meta_learning.update(agent, state, action, reward, next_state, done)

    # Plot meta-learning metrics
    meta_learning.plot_meta_learning_metrics(agent_class='Agent')

    # Save and load meta-knowledge
    meta_learning.save_meta_knowledge('meta_knowledge.npy')
    meta_learning.load_meta_knowledge('meta_knowledge.npy')
