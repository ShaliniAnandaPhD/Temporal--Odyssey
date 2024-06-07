import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.n

        # Experience replay memory
        self.memory = deque(maxlen=2000)

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        # Q-network
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.state_size))
        model.add(Dropout(0.2))  # Dropout layer for regularization
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()  # Explore
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.expand_dims(next_state, axis=0)
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            state = np.expand_dims(state, axis=0)
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.update_epsilon()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            for time in range(500):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"episode: {e}/{episodes}, score: {time}, e: {self.epsilon:.2}")
                    break
                self.replay()

    def test(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for time in range(500):
                action = np.argmax(self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0])
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    print(f"episode: {e}/{episodes}, score: {time}, total reward: {total_reward}")
                    break

if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env)

    # Train the agent
    agent.train(1000)

    # Test the agent
    agent.test(100)

    # Save the model
    agent.save("dqn_model.h5")

    # Load the model
    agent.load("dqn_model.h5")

    # Test the loaded model
    agent.test(10)

# Additional utility functions for more comprehensive training and testing

def run_experiment(env_name, episodes=1000, test_episodes=100, model_name="dqn_model.h5"):
    env = gym.make(env_name)
    agent = DQNAgent(env)
    agent.train(episodes)
    agent.test(test_episodes)
    agent.save(model_name)
    agent.load(model_name)
    agent.test(10)

def hyperparameter_tuning():
    env = gym.make('CartPole-v1')
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [32, 64, 128]
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"Testing with learning rate: {lr}, batch size: {bs}")
            agent = DQNAgent(env)
            agent.learning_rate = lr
            agent.batch_size = bs
            agent.train(500)
            agent.test(50)

# Run experiments
run_experiment('CartPole-v1')
hyperparameter_tuning()

# Logging and monitoring
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_training_progress(agent, episode, total_episodes):
    logger.info(f"Episode: {episode}/{total_episodes}, Epsilon: {agent.epsilon:.2f}, Score: {agent.score}")

def log_testing_results(agent, episode, total_episodes, total_reward):
    logger.info(f"Test Episode: {episode}/{total_episodes}, Total Reward: {total_reward}")

# Visualization of results
import matplotlib.pyplot as plt

def plot_results(scores, title="Training Progress"):
    plt.plot(scores)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

# Example usage of logging and plotting
scores = []
for e in range(100):
    state = env.reset()
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            scores.append(total_reward)
            log_training_progress(agent, e, 100)
            break
        agent.replay()

plot_results(scores)

