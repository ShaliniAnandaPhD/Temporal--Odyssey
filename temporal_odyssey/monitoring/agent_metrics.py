import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentMetrics:
    def __init__(self):
        self.metrics = {}
        self.all_agents_metrics = {}
        logger.info("AgentMetrics initialized.")

    def update_metrics(self, agent_id, metric_name, value):
        """
        Update a specific metric for an agent.

        Args:
            agent_id (str): Identifier for the agent.
            metric_name (str): Name of the metric to update.
            value (float): Value to update the metric with.
        """
        if agent_id not in self.metrics:
            self.metrics[agent_id] = {}
        if metric_name not in self.metrics[agent_id]:
            self.metrics[agent_id][metric_name] = []
        self.metrics[agent_id][metric_name].append(value)
        logger.info(f"Metric {metric_name} updated for agent {agent_id} with value {value}.")

    def aggregate_metrics(self, agent_id):
        """
        Aggregate metrics for a specific agent.

        Args:
            agent_id (str): Identifier for the agent.
        """
        if agent_id not in self.metrics:
            logger.warning(f"No metrics found for agent {agent_id}.")
            return

        aggregated_metrics = {metric: np.mean(values) for metric, values in self.metrics[agent_id].items()}
        self.all_agents_metrics[agent_id] = aggregated_metrics
        logger.info(f"Aggregated metrics for agent {agent_id}: {aggregated_metrics}")

    def aggregate_all_metrics(self):
        """
        Aggregate metrics across all agents.
        """
        all_metrics = pd.DataFrame.from_dict(self.metrics, orient='index')
        self.all_agents_metrics['overall'] = all_metrics.mean().to_dict()
        logger.info(f"Aggregated metrics for all agents: {self.all_agents_metrics['overall']}")

    def visualize_agent_performance(self, agent_id):
        """
        Visualize the performance metrics for a specific agent.

        Args:
            agent_id (str): Identifier for the agent.
        """
        if agent_id not in self.metrics:
            logger.warning(f"No metrics found for agent {agent_id}.")
            return

        df = pd.DataFrame.from_dict(self.metrics[agent_id], orient='index').T
        df.plot(kind='line', figsize=(12, 6))
        plt.title(f'Performance Metrics for Agent {agent_id}')
        plt.xlabel('Episodes')
        plt.ylabel('Values')
        plt.show()
        logger.info(f"Performance metrics visualized for agent {agent_id}.")

    def visualize_all_agents_performance(self):
        """
        Visualize the aggregated performance metrics across all agents.
        """
        df = pd.DataFrame.from_dict(self.all_agents_metrics, orient='index')
        df.plot(kind='bar', figsize=(12, 6))
        plt.title('Aggregated Performance Metrics for All Agents')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.show()
        logger.info("Aggregated performance metrics visualized for all agents.")

    def update_multiple_agents(self, agents_metrics):
        """
        Update metrics for multiple agents.

        Args:
            agents_metrics (dict): Dictionary containing agent IDs as keys and another dictionary of metrics as values.
        """
        for agent_id, metrics in agents_metrics.items():
            for metric_name, value in metrics.items():
                self.update_metrics(agent_id, metric_name, value)

    def compute_additional_metrics(self, agent_id):
        """
        Compute additional performance metrics relevant to reinforcement learning.

        Args:
            agent_id (str): Identifier for the agent.
        """
        if agent_id not in self.metrics:
            logger.warning(f"No metrics found for agent {agent_id}.")
            return

        rewards = self.metrics[agent_id].get('reward', [])
        episode_lengths = self.metrics[agent_id].get('episode_length', [])
        if rewards:
            self.update_metrics(agent_id, 'average_reward', np.mean(rewards))
            self.update_metrics(agent_id, 'total_reward', np.sum(rewards))
        if episode_lengths:
            self.update_metrics(agent_id, 'average_episode_length', np.mean(episode_lengths))

        logger.info(f"Additional metrics computed for agent {agent_id}.")

    def statistical_test(self, agent_id1, agent_id2, metric, test='t-test'):
        """
        Perform statistical tests to compare metrics between two agents.

        Args:
            agent_id1 (str): Identifier for the first agent.
            agent_id2 (str): Identifier for the second agent.
            metric (str): Name of the metric to compare.
            test (str): Type of statistical test ('t-test' or 'anova').

        Returns:
            float: p-value from the statistical test.
        """
        if agent_id1 not in self.metrics or agent_id2 not in self.metrics:
            logger.warning(f"Metrics not found for one or both agents: {agent_id1}, {agent_id2}.")
            return None

        data1 = self.metrics[agent_id1].get(metric, [])
        data2 = self.metrics[agent_id2].get(metric, [])

        if not data1 or not data2:
            logger.warning(f"Insufficient data for metric {metric} for agents: {agent_id1}, {agent_id2}.")
            return None

        if test == 't-test':
            _, p_value = stats.ttest_ind(data1, data2)
        elif test == 'anova':
            _, p_value = stats.f_oneway(data1, data2)
        else:
            raise ValueError("Invalid test type. Choose 't-test' or 'anova'.")

        logger.info(f"Statistical test ({test}) performed with p-value: {p_value}")
        return p_value

    def log_to_mlflow(self):
        """
        Log the metrics to MLflow.
        """
        for agent_id, metrics in self.all_agents_metrics.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{agent_id}_{key}", value)
                elif isinstance(value, np.ndarray):
                    mlflow.log_artifact(pd.DataFrame(value).to_csv(index=False), f"{agent_id}_{key}.csv")
                else:
                    mlflow.log_param(f"{agent_id}_{key}", value)
        logger.info("Metrics logged to MLflow.")

# Example usage
if __name__ == "__main__":
    agent_metrics = AgentMetrics()

    # Example data
    agents_data = {
        'agent_1': {'reward': 10, 'episode_length': 5},
        'agent_2': {'reward': 15, 'episode_length': 6},
        'agent_1': {'reward': 20, 'episode_length': 7},
        'agent_2': {'reward': 25, 'episode_length': 8},
    }

    # Update multiple agents
    agent_metrics.update_multiple_agents(agents_data)

    # Compute additional metrics
    agent_metrics.compute_additional_metrics('agent_1')
    agent_metrics.compute_additional_metrics('agent_2')

    # Aggregate and visualize metrics
    agent_metrics.aggregate_metrics('agent_1')
    agent_metrics.aggregate_metrics('agent_2')
    agent_metrics.visualize_agent_performance('agent_1')
    agent_metrics.visualize_agent_performance('agent_2')
    agent_metrics.aggregate_all_metrics()
    agent_metrics.visualize_all_agents_performance()

    # Perform statistical test
    agent_metrics.statistical_test('agent_1', 'agent_2', 'reward')

    # Log metrics to MLflow
    agent_metrics.log_to_mlflow()


