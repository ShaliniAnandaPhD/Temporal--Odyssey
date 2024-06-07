import logging
import csv
import os
from datetime import datetime
from collections import defaultdict


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Telemetry:
    def __init__(self, filepath="telemetry.csv"):
        """
        Initialize the Telemetry class.

        Parameters:
        filepath (str): The path to the CSV file where telemetry data will be stored.
        """
        self.filepath = filepath
        self.fields = ["timestamp", "episode", "step", "state", "action", "reward", "next_state", "done"]
        self.data = defaultdict(list)
        self._initialize_file()

    def _initialize_file(self):
        """
        Initialize the CSV file and write the header if the file does not exist.
        """
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=self.fields)
                    writer.writeheader()
                    logger.info(f"Telemetry file '{self.filepath}' created with header.")
            except Exception as e:
                logger.error(f"Failed to create telemetry file: {e}")

    def record(self, episode, step, state, action, reward, next_state, done):
        """
        Record telemetry data for each step.

        Parameters:
        episode (int): The current episode number.
        step (int): The current step number.
        state (object): The current state of the environment.
        action (int): The action taken by the agent.
        reward (float): The reward received after taking the action.
        next_state (object): The next state of the environment.
        done (bool): Whether the episode has ended.
        """
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "episode": episode,
            "step": step,
            "state": str(state),
            "action": action,
            "reward": reward,
            "next_state": str(next_state),
            "done": done
        }
        for key, value in record.items():
            self.data[key].append(value)
        logger.info(f"Telemetry data recorded for episode {episode}, step {step}.")

    def save(self):
        """
        Save the recorded telemetry data to the CSV file.
        """
        try:
            with open(self.filepath, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                for i in range(len(self.data["timestamp"])):
                    row = {field: self.data[field][i] for field in self.fields}
                    writer.writerow(row)
            logger.info(f"Telemetry data saved to '{self.filepath}'.")
            self._clear_data()
        except Exception as e:
            logger.error(f"Failed to save telemetry data: {e}")

    def _clear_data(self):
        """
        Clear the in-memory telemetry data.
        """
        self.data = defaultdict(list)
        logger.info("In-memory telemetry data cleared.")

    def load(self):
        """
        Load telemetry data from the CSV file.

        Returns:
        list of dict: The list of telemetry data records.
        """
        try:
            with open(self.filepath, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                data = list(reader)
                logger.info(f"Telemetry data loaded from '{self.filepath}'.")
                return data
        except Exception as e:
            logger.error(f"Failed to load telemetry data: {e}")
            return []

    def get_statistics(self):
        """
        Calculate and log basic statistics from the telemetry data.

        Returns:
        dict: A dictionary containing average reward and step count per episode.
        """
        try:
            data = self.load()
            if not data:
                return {}

            episode_rewards = defaultdict(list)
            episode_steps = defaultdict(int)

            for row in data:
                episode = int(row["episode"])
                reward = float(row["reward"])
                episode_rewards[episode].append(reward)
                episode_steps[episode] += 1

            avg_rewards = {ep: sum(rewards)/len(rewards) for ep, rewards in episode_rewards.items()}
            avg_steps = {ep: steps for ep, steps in episode_steps.items()}

            statistics = {
                "average_rewards": avg_rewards,
                "average_steps": avg_steps
            }
            logger.info(f"Telemetry statistics calculated: {statistics}")
            return statistics
        except Exception as e:
            logger.error(f"Failed to calculate telemetry statistics: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    telemetry = Telemetry()

    # Simulate recording telemetry data
    for episode in range(10):
        for step in range(50):
            state = {"position": (step, step)}
            action = step % 4
            reward = step
            next_state = {"position": (step + 1, step + 1)}
            done = step == 49
            telemetry.record(episode, step, state, action, reward, next_state, done)

    # Save the telemetry data
    telemetry.save()

    # Load the telemetry data
    data = telemetry.load()
    print(data)

    # Get statistics from the telemetry data
    statistics = telemetry.get_statistics()
    print(statistics)
