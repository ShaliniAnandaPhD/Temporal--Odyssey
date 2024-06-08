import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import logging
import random
from collections import deque
import configparser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load hyperparameters from configuration file
config = configparser.ConfigParser()
config.read('config.ini')
gamma = config.getfloat('hyperparameters', 'gamma')
learning_rate = config.getfloat('hyperparameters', 'learning_rate')
epsilon_start = config.getfloat('hyperparameters', 'epsilon_start')
epsilon_min = config.getfloat('hyperparameters', 'epsilon_min')
epsilon_decay = config.getfloat('hyperparameters', 'epsilon_decay')
memory_size = config.getint('hyperparameters', 'memory_size')
batch_size = config.getint('hyperparameters', 'batch_size')
prioritized_replay_alpha = config.getfloat('hyperparameters', 'prioritized_replay_alpha')
prioritized_replay_beta = config.getfloat('hyperparameters', 'prioritized_replay_beta')
noise_sigma = config.getfloat('hyperparameters', 'noise_sigma')
gradient_clip_value = config.getfloat('hyperparameters', 'gradient_clip_value')

class HybridLearningAgent:
    def __init__(self, env):
        self.env = env
        self.epsilon = epsilon_start
        self.memory = PrioritizedReplayBuffer(memory_size, prioritized_replay_alpha)
        self.scaler = StandardScaler()

        # Shared layers
        self.shared_layers = self._build_shared_layers()

        # Initialize models with shared layers
        self.q_network = self._build_q_network(self.shared_layers)
        self.target_network = self._build_q_network(self.shared_layers)
        self.supervised_model = self._build_supervised_model(self.shared_layers)
        self.update_target_network()

        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=gradient_clip_value)
        logger.info("HybridLearningAgent initialized.")

    def _build_shared_layers(self):
        """Build shared convolutional and dense layers for both Q-network and supervised model."""
        inputs = Input(shape=self.env.observation_space.shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        return Model(inputs, x)

    def _build_q_network(self, shared_layers):
        """Build the Q-network using shared layers."""
        x = shared_layers.output
        x = NoisyDense(512, activation='relu')(x)
        state_value = Dense(1)(x)
        action_advantage = Dense(self.env.action_space.n)(x)
        outputs = DuelingLayer()([state_value, action_advantage])
        model = Model(shared_layers.input, outputs)
        model.compile(optimizer=self.optimizer, loss='mse')
        logger.info("Q-network built.")
        return model

    def _build_supervised_model(self, shared_layers):
        """Build the supervised learning model using shared layers."""
        x = shared_layers.output
        x = Dense(512, activation='relu')(x)
        outputs = Dense(self.env.action_space.n, activation='softmax')(x)
        model = Model(shared_layers.input, outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy')
        logger.info("Supervised model built.")
        return model

    def update_target_network(self):
        """Update the target network weights with the Q-network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
        logger.info("Target network updated.")

    def train(self, num_episodes):
        """Unified training loop for Q-network and supervised learning."""
        for e in range(num_episodes):
            state = self.env.reset()
            state = self.scaler.fit_transform(state)
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.scaler.transform(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if len(self.memory) > batch_size:
                    self.replay()
            
            if self.epsilon > epsilon_min:
                self.epsilon *= epsilon_decay
            
            logger.info(f"Episode {e+1}/{num_episodes} completed. Epsilon: {self.epsilon}")

            if e % 10 == 0:
                self.update_target_network()

    def act(self, state):
        """Choose an action based on the current state."""
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()  # Explore
        act_values = self.q_network.predict(state)
        return np.argmax(act_values[0])  # Exploit

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in replay buffer."""
        self.memory.add(state, action, reward, next_state, done)
        logger.debug(f"Stored experience: {(state, action, reward, next_state, done)}")

    def replay(self):
        """Train the Q-network using experiences from the replay buffer."""
        batch, importance_weights, indices = self.memory.sample(batch_size, prioritized_replay_beta)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        target_qvalues = self.q_network.predict(states)
        next_qvalues = self.target_network.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            if done:
                target_qvalues[i][action] = reward
            else:
                target_qvalues[i][action] = reward + gamma * np.amax(next_qvalues[i])

        self.q_network.fit(states, target_qvalues, sample_weight=importance_weights, verbose=0)
        self.memory.update_priorities(indices, np.abs(target_qvalues[np.arange(batch_size), actions] - self.q_network.predict(states)))
        logger.debug("Replayed experiences for training.")

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta):
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]
        importance_weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        importance_weights /= importance_weights.max()
        return batch, importance_weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

class NoisyDense(Dense):
    def __init__(self, units, sigma=noise_sigma, **kwargs):
        super().__init__(units, **kwargs)
        self.sigma = sigma

    def build(self, input_shape):
        super().build(input_shape)
        self.noise_weight = self.add_weight(shape=self.kernel.shape, initializer='random_normal', trainable=True)
        self.noise_bias = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        noisy_weights = self.kernel + self.sigma * self.noise_weight
        noisy_biases = self.bias + self.sigma * self.noise_bias
        return tf.nn.dense(inputs, noisy_weights, noisy_biases)

class DuelingLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        state_value, action_advantage = inputs
        action_advantage = action_advantage - tf.reduce_mean(action_advantage, axis=1, keepdims=True)
        outputs = state_value + action_advantage
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1]

# Potential errors and solutions:
# 1. **Shape Mismatch**: Ensure the input shape to the models matches the environment's observation space.
#    - Solution: Check and adjust the input shape definitions in `_build_shared_layers`.
# 2. **Replay Buffer Overflow**: Ensure the memory buffer does not exceed its maximum length.
#    - Solution: Use `deque` with a fixed `maxlen` to automatically handle buffer size.
# 3. **Exploration-Exploitation Trade-off**: Ensure epsilon decays correctly and allows for sufficient exploration.
#    - Solution: Adjust `epsilon_decay`, `epsilon_min`, and initial `epsilon` values as needed.
# 4. **Training Stability**: Ensure the models are trained with stable targets and learning rates.
#    - Solution: Use target network updates and appropriate learning rate for the optimizer.


