import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DoubleDQN:
    def __init__(self, state_shape, action_size, learning_rate=0.001):
        """
        Initialize the Double DQN model.
        
        Args:
            state_shape (tuple): Shape of the input state.
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()
        logger.info("Double DQN model initialized.")
    
    def build_network(self):
        """
        Build the Q-network architecture.
        
        Returns:
            Model: Compiled Q-network model.
        """
        inputs = Input(shape=self.state_shape)
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        q_values = Dense(self.action_size)(x)
        model = Model(inputs, q_values)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        logger.info("Q-network built.")
        return model
    
    def update_target_network(self):
        """
        Update the target network weights with the Q-network weights.
        """
        self.target_network.set_weights(self.q_network.get_weights())
        logger.info("Target network updated.")
    
    def train_double_dqn(self, experience, gamma):
        """
        Train the Double DQN model.
        
        Args:
            experience (tuple): A batch of experience tuples (state, action, reward, next_state, done).
            gamma (float): Discount factor for future rewards.
        """
        states, actions, rewards, next_states, dones = experience
        
        # Predict Q-values for next states using the Q-network
        next_q_values = self.q_network.predict(next_states)
        
        # Predict Q-values for next states using the target network
        next_target_q_values = self.target_network.predict(next_states)
        
        # Select the best action using the Q-network and evaluate its value using the target network
        max_next_actions = np.argmax(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * gamma * next_target_q_values[np.arange(len(next_states)), max_next_actions]
        
        # Create mask to select the Q-value for the taken action
        mask = tf.one_hot(actions, self.action_size)
        
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values = tf.reduce_sum(q_values * mask, axis=1)
            loss = tf.keras.losses.MSE(target_q_values, q_values)
        
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        logger.info("Double DQN trained on batch with loss: %.4f", loss.numpy())

# Example usage
if __name__ == "__main__":
    state_shape = (4,)  # Example state shape for CartPole
    action_size = 2     # Example action size for CartPole
    
    double_dqn = DoubleDQN(state_shape, action_size)
    
    # Example experience batch
    batch_size = 32
    experience = (
        np.random.rand(batch_size, *state_shape),   # states
        np.random.randint(action_size, size=batch_size),  # actions
        np.random.rand(batch_size),  # rewards
        np.random.rand(batch_size, *state_shape),  # next_states
        np.random.randint(2, size=batch_size)  # dones
    )
    
    gamma = 0.99
    double_dqn.train_double_dqn(experience, gamma)
