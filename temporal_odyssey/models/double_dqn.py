import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Subtract
from tensorflow.keras.optimizers import Adam
import logging
from collections import deque
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DoubleDQN:
    def __init__(self, state_shape, action_size, learning_rate=0.001, batch_size=32, memory_size=10000, gamma=0.99):
        """
        Initialize the Double DQN model.

        Args:
            state_shape (tuple): Shape of the input state.
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Size of training batches.
            memory_size (int): Size of the replay memory.
            gamma (float): Discount factor for future rewards.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma

        self.model = self.build_network()
        self.target_model = self.build_network()
        self.update_target_model()
        logger.info("Double DQN model initialized.")
    
    def build_network(self):
        """
        Construct the neural network architecture.

        Returns:
            Model: Compiled Double DQN model.
        """
        inputs = Input(shape=self.state_shape)

        # Example convolutional layers for image-based inputs
        if len(self.state_shape) == 3:  # For image inputs
            x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
            x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = Flatten()(x)
        else:
            x = Flatten()(inputs)
        
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        q_values = Dense(self.action_size)(x)

        model = Model(inputs, q_values)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        logger.info("Neural network built.")
        return model
    
    def update_target_model(self):
        """
        Update the target model to match the current model.
        """
        self.target_model.set_weights(self.model.get_weights())
        logger.info("Target model updated to match current model.")
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the replay memory.

        Args:
            state (array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The next state.
            done (bool): Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))
        logger.debug("Experience stored in memory.")
    
    def act(self, state, epsilon):
        """
        Choose an action based on the current state and exploration rate.

        Args:
            state (array): The current state.
            epsilon (float): The exploration rate.

        Returns:
            int: The action chosen.
        """
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train the model using a batch of experiences from the replay memory.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        next_q_values = self.model.predict(next_states)
        next_target_q_values = self.target_model.predict(next_states)
        
        targets = rewards + self.gamma * (1 - dones) * next_target_q_values[range(self.batch_size), np.argmax(next_q_values, axis=1)]

        q_values = self.model.predict(states)
        q_values[range(self.batch_size), actions] = targets

        self.model.fit(states, q_values, epochs=1, verbose=0)
        logger.info("Model trained on replay batch.")
    
    def save_model(self, path):
        """
        Save the model weights to the specified path.
        """
        self.model.save_weights(path)
        logger.info(f"Model weights saved to {path}")
    
    def load_model(self, path):
        """
        Load the model weights from the specified path.
        """
        self.model.load_weights(path)
        logger.info(f"Model weights loaded from {path}")

# Example usage
if __name__ == "__main__":
    state_shape = (84, 84, 4)  # Example state shape for an image input
    action_size = 4            # Example action size
    
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
    
    epsilon = 0.1
    state = np.random.rand(*state_shape)
    action = double_dqn.act(state, epsilon)
    print(f"Chosen action: {action}")

    double_dqn.remember(*experience)
    double_dqn.replay()
    double_dqn.save_model('double_dqn_weights.h5')
    double_dqn.load_model('double_dqn_weights.h5')

