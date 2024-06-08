import os
import json
import gzip
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Telemetry:
    def __init__(self, log_dir="logs", compress=True):
        """
        Initialize the Telemetry class.

        Args:
            log_dir (str): Directory to save telemetry logs.
            compress (bool): Whether to compress the telemetry data.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.compress = compress
        self.data = defaultdict(list)
        logger.info("Telemetry initialized.")

    def log(self, agent_id, episode, step, reward, state, action, done):
        """
        Log telemetry data for a single agent.

        Args:
            agent_id (str): Identifier for the agent.
            episode (int): Episode number.
            step (int): Step number within the episode.
            reward (float): Reward received at this step.
            state (array): Current state.
            action (int): Action taken.
            done (bool): Whether the episode is done.
        """
        self.data[agent_id].append({
            "episode": episode,
            "step": step,
            "reward": reward,
            "state": state.tolist(),
            "action": action,
            "done": done
        })
        logger.debug(f"Logged data for agent {agent_id}, episode {episode}, step {step}.")

    def save(self):
        """
        Save the telemetry data to disk.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"telemetry_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        if self.compress:
            filepath += ".gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(self.data, f)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f)
        logger.info(f"Telemetry data saved to {filepath}.")

    def load(self, filepath):
        """
        Load telemetry data from disk.

        Args:
            filepath (str): Path to the telemetry log file.
        """
        if filepath.endswith(".gz"):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        logger.info(f"Telemetry data loaded from {filepath}.")

    def visualize(self, agent_id=None):
        """
        Visualize telemetry data for the specified agent.

        Args:
            agent_id (str): Identifier for the agent. If None, visualize data for all agents.
        """
        if agent_id:
            self._plot_agent_data(agent_id)
        else:
            for agent in self.data.keys():
                self._plot_agent_data(agent)

    def _plot_agent_data(self, agent_id):
        """
        Plot telemetry data for a single agent.

        Args:
            agent_id (str): Identifier for the agent.
        """
        agent_data = self.data[agent_id]
        df = pd.DataFrame(agent_data)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(df['step'], df['reward'], label='Reward')
        plt.title(f'Agent {agent_id} - Reward per Step')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(df['step'], df['done'].astype(int), label='Done')
        plt.title(f'Agent {agent_id} - Done per Step')
        plt.xlabel('Step')
        plt.ylabel('Done')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def aggregate_data(self, agent_id=None):
        """
        Aggregate telemetry data across multiple episodes or runs.

        Args:
            agent_id (str): Identifier for the agent. If None, aggregate data for all agents.

        Returns:
            DataFrame: Aggregated telemetry data.
        """
        if agent_id:
            return self._aggregate_agent_data(agent_id)
        else:
            aggregated_data = pd.DataFrame()
            for agent in self.data.keys():
                agent_data = self._aggregate_agent_data(agent)
                agent_data['agent_id'] = agent
                aggregated_data = pd.concat([aggregated_data, agent_data])
            return aggregated_data

    def _aggregate_agent_data(self, agent_id):
        """
        Aggregate telemetry data for a single agent.

        Args:
            agent_id (str): Identifier for the agent.

        Returns:
            DataFrame: Aggregated telemetry data for the agent.
        """
        agent_data = self.data[agent_id]
        df = pd.DataFrame(agent_data)
        aggregation = {
            'reward': ['sum', 'mean', 'std'],
            'done': 'sum'
        }
        aggregated_df = df.groupby('episode').agg(aggregation)
        aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
        return aggregated_df

    def filter_data(self, agent_id, criteria):
        """
        Filter telemetry data based on specific criteria.

        Args:
            agent_id (str): Identifier for the agent.
            criteria (dict): Dictionary containing filter criteria.

        Returns:
            DataFrame: Filtered telemetry data.
        """
        agent_data = self.data[agent_id]
        df = pd.DataFrame(agent_data)
        query = ' & '.join([f"{key} {value}" for key, value in criteria.items()])
        filtered_df = df.query(query)
        return filtered_df

    def real_time_monitoring(self, agent_id):
        """
        Real-time monitoring of the agent's performance.

        Args:
            agent_id (str): Identifier for the agent.
        """
        raise NotImplementedError("Real-time monitoring is not yet implemented.")

# Example usage
if __name__ == "__main__":
    telemetry = Telemetry()
    telemetry.log("agent_1", 1, 1, 1.0, np.array([0.1, 0.2, 0.3]), 0, False)
    telemetry.log("agent_1", 1, 2, 1.0, np.array([0.1, 0.2, 0.3]), 1, True)
    telemetry.save()
    telemetry.load("logs/telemetry_20211010_123456.json.gz")
    telemetry.visualize("agent_1")
    aggregated_data = telemetry.aggregate_data("agent_1")
    print(aggregated_data)
    filtered_data = telemetry.filter_data("agent_1", {"reward": ">0.5"})
    print(filtered_data)

