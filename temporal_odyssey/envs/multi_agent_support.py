import numpy as np
import gym
from gym import spaces
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentEnv(gym.Env):
    def __init__(self, env_name, num_agents, reward_scheme=None, observation_space=None):
        """
        Initialize the MultiAgentEnv wrapper.

        Args:
            env_name (str): Name of the single-agent environment to wrap.
            num_agents (int): Number of agents in the environment.
            reward_scheme (dict): Dictionary defining reward schemes for agents.
            observation_space (spaces.Space): Observation space for partial observability.
        """
        self.env = gym.make(env_name)
        self.num_agents = num_agents
        self.reward_scheme = reward_scheme
        self.observation_space = observation_space if observation_space else self.env.observation_space

        # Define action and observation spaces as tuples of the single-agent spaces
        self.action_space = spaces.Tuple([self.env.action_space for _ in range(num_agents)])
        self.observation_space = spaces.Tuple([self.observation_space for _ in range(num_agents)])

        logger.info(f"MultiAgentEnv initialized with {num_agents} agents.")

    def reset(self):
        """
        Reset the environment and return initial observations for all agents.

        Returns:
            list: List of initial observations for each agent.
        """
        try:
            obs = self.env.reset()
            return [obs for _ in range(self.num_agents)]
        except Exception as e:
            logger.error(f"Error during reset: {e}")
            raise

    def step(self, actions):
        """
        Take a step in the environment with actions from all agents.

        Args:
            actions (list): List of actions for each agent.

        Returns:
            tuple: Observations, rewards, dones, and info for all agents.
        """
        try:
            obs, reward, done, info = self.env.step(actions[0])
            
            # Process rewards for each agent based on reward_scheme if provided
            rewards = self._process_rewards(reward)

            return (
                [obs for _ in range(self.num_agents)],
                rewards,
                [done for _ in range(self.num_agents)],
                [info for _ in range(self.num_agents)]
            )
        except Exception as e:
            logger.error(f"Error during step: {e}")
            raise

    def _process_rewards(self, base_reward):
        """
        Process rewards for each agent based on reward_scheme.

        Args:
            base_reward (float): Base reward from the environment.

        Returns:
            list: List of rewards for each agent.
        """
        if self.reward_scheme:
            return [self.reward_scheme.get(agent, base_reward) for agent in range(self.num_agents)]
        return [base_reward for _ in range(self.num_agents)]

    def send_message(self, sender_id, receiver_id, message):
        """
        Send a message from one agent to another.

        Args:
            sender_id (int): ID of the sending agent.
            receiver_id (int): ID of the receiving agent.
            message (str): Message content.
        """
        logger.info(f"Agent {sender_id} sent message to Agent {receiver_id}: {message}")
        # Implement message sending logic here

    def receive_message(self, receiver_id):
        """
        Receive a message for an agent.

        Args:
            receiver_id (int): ID of the receiving agent.

        Returns:
            str: Received message content.
        """
        # Implement message receiving logic here
        logger.info(f"Agent {receiver_id} received a message.")
        return "Sample message"

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
            try:
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
            except Exception as e:
                logger.error(f"Error during training: {e}")
                raise

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


