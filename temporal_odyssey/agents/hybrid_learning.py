import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import logging
import random
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridLearningAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)  # Fixed-size buffer to store experiences
        self.batch_size = 64
        self.scaler = StandardScaler()

        # Initialize models
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.supervised_model = self._build_supervised_model()
        self.update_target_network()

        # Initialize optimizer
        self.optimizer = Adam(learning_rate=self.learning_rate)
        logger.info("HybridLearningAgent initialized.")

    def _build_q_network(self):
        """
        Build the Q-network using Keras.
        """
        inputs = Input(shape=self.env.observation_space.shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(self.env.action_space.n, activation='linear')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        logger.info("Q-network built.")
        return model

    def _build_supervised_model(self):
        """
        Build the supervised learning model using Keras.
        """
        inputs = Input(shape=self.env.observation_space.shape)
        x = Flatten()(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.env.action_space.n, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        logger.info("Supervised learning model built.")
        return model

    def update_target_network(self):
        """
        Update the target network with weights from the Q-network.
        """
        self.target_network.set_weights(self.q_network.get_weights())
        logger.info("Target network updated.")

    def remember(self, state, action, reward, next_state, done):
        """
        Store experiences in memory.
        """
        self.memory.append((state, action, reward, next_state, done))
        logger.info("Experience remembered.")

    def act(self, state):
        """
        Select an action using an epsilon-greedy strategy.
        """
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()
            logger.info("Random action selected.")
        else:
            q_values = self.q_network.predict(np.expand_dims(state, axis=0))
            action = np.argmax(q_values[0])
            logger.info("Action selected from Q-network.")
        return action

    def replay(self):
        """
        Train the Q-network using experiences from memory.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)

        q_values = self.q_network.predict(states)
        target_q_values = self.target_network.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_q_values[i])

        self.q_network.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logger.info("Q-network trained on minibatch.")

    def supervised_train(self, data, labels):
        """
        Train the supervised learning model using labeled data.
        """
        self.supervised_model.fit(data, labels, epochs=10, verbose=1)
        logger.info("Supervised learning model trained.")

    def hybrid_train(self, episodes):
        """
        Train the agent using both reinforcement learning and supervised learning.
        """
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()
            logger.info(f"Episode: {episode+1}, Total Reward: {total_reward}")

    def evaluate(self, episodes):
        """
        Evaluate the agent's performance.
        """
        total_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.q_network.predict(np.expand_dims(state, axis=0))[0])
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)
            logger.info(f"Evaluation Episode: {episode+1}, Total Reward: {total_reward}")
        return np.mean(total_rewards)

# Example usage
if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v1')  # Replace with the custom TimeTravelEnv
    agent = HybridLearningAgent(env)
    
    # Train the agent using hybrid learning techniques
    agent.hybrid_train(1000)

    # Evaluate the agent's performance
    mean_reward = agent.evaluate(100)
    print(f"Mean Reward over 100 episodes: {mean_reward}")

    # Supervised training example
    # Assuming `data` and `labels` are preprocessed and available
    # data, labels = preprocess_data()
    # agent.supervised_train(data, labels)

