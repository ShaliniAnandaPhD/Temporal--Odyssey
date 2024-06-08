import numpy as np
import gym
from gym import spaces
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentEnv(gym.Env):
    def __init__(self, env_name, num_agents):
        """
        Initialize the MultiAgentEnv wrapper.

        Args:
            env_name (str): Name of the single-agent environment to wrap.
            num_agents (int): Number of agents in the environment.
        """
        self.env = gym.make(env_name)
        self.num_agents = num_agents

        # Define action and observation spaces as tuples of the single-agent spaces
        self.action_space = spaces.Tuple([self.env.action_space for _ in range(num_agents)])
        self.observation_space = spaces.Tuple([self.env.observation_space for _ in range(num_agents)])
        logger.info(f"MultiAgentEnv initialized with {num_agents} agents.")

    def reset(self):
        """
        Reset the environment and return initial observations for all agents.

        Returns:
            list: List of initial observations for each agent.
        """
        obs = self.env.reset()
        return [obs for _ in range(self.num_agents)]

    def step(self, actions):
        """
        Take a step in the environment with actions from all agents.

        Args:
            actions (list): List of actions for each agent.

        Returns:
            tuple: Observations, rewards, dones, and info for all agents.
        """
        obs, reward, done, info = self.env.step(actions[0])
        return (
            [obs for _ in range(self.num_agents)],
            [reward for _ in range(self.num_agents)],
            [done for _ in range(self.num_agents)],
            [info for _ in range(self.num_agents)]
        )

def train_multi_agent(env, agents, num_episodes):
    """
    Train multiple agents in the given environment.

    Args:
        env (MultiAgentEnv): The multi-agent environment.
        agents (list): List of agent instances.
        num_episodes (int): Number of training episodes.
    """
    for episode in range(num_episodes):
        states = env.reset()
        done = [False for _ in range(env.num_agents)]
        total_rewards = [0 for _ in range(env.num_agents)]

        while not all(done):
            # Each agent selects an action based on its current state
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones, infos = env.step(actions)

            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                total_rewards[i] += rewards[i]

            states = next_states
            done = dones

        logger.info(f"Episode {episode+1}/{num_episodes} completed with total rewards: {total_rewards}")

# Example usage
if __name__ == "__main__":
    from hybrid_learning import HybridLearningAgent

    env_name = 'CartPole-v1'
    num_agents = 2
    num_episodes = 1000

    env = MultiAgentEnv(env_name, num_agents)
    agents = [HybridLearningAgent(env) for _ in range(num_agents)]

    train_multi_agent(env, agents, num_episodes)
