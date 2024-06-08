import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract
from tensorflow.keras.optimizers import Adam
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuelingDQN:
    def __init__(self, state_shape, action_size, learning_rate=0.001):
        """
        Initialize the Dueling DQN model.
        
        Args:
            state_shape (tuple): Shape of the input state.
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.build_dueling_network()
        logger.info("Dueling DQN model initialized.")
    
    def build_dueling_network(self):
        """
        Construct the dueling network architecture.
        
        Returns:
            Model: Compiled Dueling DQN model.
        """
        inputs = Input(shape=self.state_shape)
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        
        # State Value Tower
        state_value = Dense(1)(x)
        
        # Action Advantage Tower
        action_advantage = Dense(self.action_size)(x)
        mean_advantage = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(action_advantage)
        action_advantage = Subtract()([action_advantage, mean_advantage])
        
        # Combine towers
        q_values = tf.keras.layers.Add()([state_value, action_advantage])
        
        model = Model(inputs, q_values)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        logger.info("Dueling DQN network built.")
        return model
    
    def train_dueling_dqn(self, experience, gamma):
        """
        Train the Dueling DQN model.
        
        Args:
            experience (tuple): A batch of experience tuples (state, action, reward, next_state, done).
            gamma (float): Discount factor for future rewards.
        """
        states, actions, rewards, next_states, dones = experience
        
        # Compute target Q-values
        next_q_values = self.model.predict(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * gamma * max_next_q_values
        
        # Create mask to select the Q-value for the taken action
        mask = tf.one_hot(actions, self.action_size)
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values = tf.reduce_sum(q_values * mask, axis=1)
            loss = tf.keras.losses.MSE(target_q_values, q_values)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        logger.info("Dueling DQN trained on batch with loss: %.4f", loss.numpy())

# Example usage
if __name__ == "__main__":
    state_shape = (4,)  # Example state shape for CartPole
    action_size = 2     # Example action size for CartPole
    
    dueling_dqn = DuelingDQN(state_shape, action_size)
    
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
    dueling_dqn.train_dueling_dqn(experience, gamma)
