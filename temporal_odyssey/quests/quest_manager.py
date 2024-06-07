# temporal_odyssey/quests/quest_manager.py

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Quest:
    def __init__(self, name, description, objectives, rewards):
        """
        Initialize a new quest.

        Parameters:
        name (str): The name of the quest.
        description (str): A brief description of the quest.
        objectives (list): A list of QuestObjective instances that define the quest objectives.
        rewards (list): A list of QuestReward instances that define the quest rewards.
        """
        self.name = name
        self.description = description
        self.objectives = objectives
        self.rewards = rewards
        self.completed = False
        self.start_time = None
        self.end_time = None
        logger.info(f"Quest '{self.name}' initialized.")

    def start(self):
        """
        Start the quest.
        """
        self.start_time = datetime.now()
        logger.info(f"Quest '{self.name}' started at {self.start_time}.")

    def check_completion(self, agent):
        """
        Check if all objectives of the quest are completed.

        Parameters:
        agent (object): The agent attempting the quest.
        
        Returns:
        bool: True if all objectives are completed, False otherwise.
        """
        for objective in self.objectives:
            if not objective.is_completed(agent):
                return False
        self.completed = True
        self.end_time = datetime.now()
        logger.info(f"Quest '{self.name}' completed at {self.end_time}.")
        return True

    def grant_rewards(self, agent):
        """
        Grant rewards to the agent upon quest completion.

        Parameters:
        agent (object): The agent who completed the quest.
        """
        for reward in self.rewards:
            reward.grant(agent)
        logger.info(f"Rewards for quest '{self.name}' granted to agent.")

class QuestObjective:
    def __init__(self, description, completion_check):
        """
        Initialize a new quest objective.

        Parameters:
        description (str): A brief description of the objective.
        completion_check (function): A function that checks if the objective is completed.
        """
        self.description = description
        self.completion_check = completion_check
        logger.info(f"QuestObjective '{self.description}' initialized.")

    def is_completed(self, agent):
        """
        Check if the objective is completed by the agent.

        Parameters:
        agent (object): The agent attempting the objective.
        
        Returns:
        bool: True if the objective is completed, False otherwise.
        """
        result = self.completion_check(agent)
        logger.info(f"QuestObjective '{self.description}' completion check: {result}")
        return result

class QuestReward:
    def __init__(self, reward_type, value):
        """
        Initialize a new quest reward.

        Parameters:
        reward_type (str): The type of the reward (e.g., "score", "item").
        value (any): The value of the reward.
        """
        self.reward_type = reward_type
        self.value = value
        logger.info(f"QuestReward of type '{self.reward_type}' with value '{self.value}' initialized.")

    def grant(self, agent):
        """
        Grant the reward to the agent based on the reward type and value.

        Parameters:
        agent (object): The agent receiving the reward.
        """
        if self.reward_type == "score":
            agent.score += self.value
            logger.info(f"Granted {self.value} score to agent.")
        elif self.reward_type == "item":
            agent.inventory.add_item(self.value)
            logger.info(f"Granted item '{self.value}' to agent.")
        else:
            logger.warning(f"Unknown reward type '{self.reward_type}'.")

class QuestManager:
    def __init__(self):
        """
        Initialize the QuestManager.
        """
        self.quests = []
        logger.info("QuestManager initialized.")

    def add_quest(self, quest):
        """
        Add a new quest to the quest manager.

        Parameters:
        quest (Quest): The quest to be added.
        """
        self.quests.append(quest)
        logger.info(f"Quest '{quest.name}' added to QuestManager.")

    def remove_quest(self, quest):
        """
        Remove a quest from the quest manager.

        Parameters:
        quest (Quest): The quest to be removed.
        """
        self.quests.remove(quest)
        logger.info(f"Quest '{quest.name}' removed from QuestManager.")

    def get_available_quests(self, agent):
        """
        Get the list of available quests for the agent.

        Parameters:
        agent (object): The agent seeking quests.
        
        Returns:
        list: A list of available quests.
        """
        available_quests = [quest for quest in self.quests if not quest.completed and self._is_quest_available(quest, agent)]
        logger.info(f"Available quests for agent: {[quest.name for quest in available_quests]}")
        return available_quests

    def update_quests(self, agent):
        """
        Update the state of quests based on the agent's actions and progress.

        Parameters:
        agent (object): The agent whose progress is being updated.
        """
        for quest in self.quests:
            if not quest.completed and quest.check_completion(agent):
                quest.grant_rewards(agent)
                logger.info(f"Quest '{quest.name}' updated for agent.")

    def _is_quest_available(self, quest, agent):
        """
        Check if the quest is available for the agent.

        Parameters:
        quest (Quest): The quest to be checked.
        agent (object): The agent seeking quests.
        
        Returns:
        bool: True if the quest is available, False otherwise.
        """
        # Implement quest availability logic based on the agent's state, location, or other factors.
        return True  # Placeholder for real logic

    def adapt_quests(self, agent):
        """
        Adapt quests based on the agent's skills, history, and current state.

        Parameters:
        agent (object): The agent whose quests are being adapted.
        """
        for quest in self.quests:
            if not quest.completed:
                # Example adaptation: Increase rewards for more skilled agents
                if agent.level > 5:
                    for reward in quest.rewards:
                        if reward.reward_type == "score":
                            reward.value *= 1.5
                            logger.info(f"Increased reward for quest '{quest.name}' for skilled agent.")

# Example usage
if __name__ == "__main__":
    class Agent:
        def __init__(self):
            self.score = 0
            self.level = 1
            self.inventory = Inventory()

        def level_up(self):
            self.level += 1

    class Inventory:
        def __init__(self):
            self.items = []

        def add_item(self, item):
            self.items.append(item)

    def example_completion_check(agent):
        return agent.score >= 10

    agent = Agent()
    quest_manager = QuestManager()

    # Create example quests
    quest1 = Quest(
        name="First Quest",
        description="Complete the first task",
        objectives=[QuestObjective("Score 10 points", example_completion_check)],
        rewards=[QuestReward("score", 100)]
    )

    quest_manager.add_quest(quest1)

    # Example agent interactions
    available_quests = quest_manager.get_available_quests(agent)
    for quest in available_quests:
        quest.start()
        agent.score = 10  # Simulate completing the objective
        quest_manager.update_quests(agent)

    # Adapt quests based on agent's skills
    agent.level_up()
    agent.level_up()
    agent.level_up()
    agent.level_up()
    agent.level_up()
    agent.level_up()
    quest_manager.adapt_quests(agent)

