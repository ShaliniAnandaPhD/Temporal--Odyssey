import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Quest:
    def __init__(self, name, description, objectives, rewards):
        self.name = name
        self.description = description
        self.objectives = objectives
        self.rewards = rewards
        self.completed = False
        logger.info(f"Quest '{self.name}' created with {len(self.objectives)} objectives and {len(self.rewards)} rewards.")
    
    def check_completion(self, agent):
        """Check if all objectives of the quest are completed by the agent."""
        for objective in self.objectives:
            if not objective.is_completed(agent):
                return False
        self.completed = True
        logger.info(f"Quest '{self.name}' completed by agent.")
        return True
    
    def grant_rewards(self, agent):
        """Grant rewards to the agent upon quest completion."""
        for reward in self.rewards:
            reward.grant(agent)
        logger.info(f"Rewards for quest '{self.name}' granted to agent.")

class QuestObjective:
    def __init__(self, description, check_function):
        self.description = description
        self.check_function = check_function
        logger.info(f"QuestObjective created: {self.description}")
    
    def is_completed(self, agent):
        """Check if the objective is completed by the agent."""
        completed = self.check_function(agent)
        if completed:
            logger.info(f"Objective '{self.description}' completed by agent.")
        return completed

class QuestReward:
    def __init__(self, reward_type, value):
        self.reward_type = reward_type
        self.value = value
        logger.info(f"QuestReward created: {self.reward_type} - {self.value}")
    
    def grant(self, agent):
        """Grant the reward to the agent based on the reward type and value."""
        if self.reward_type == "score":
            agent.score += self.value
            logger.info(f"Granted score reward: {self.value} to agent. New score: {agent.score}")
        elif self.reward_type == "item":
            agent.inventory.add_item(self.value)
            logger.info(f"Granted item reward: {self.value} to agent's inventory.")
        # Add more reward types and corresponding grant logic as needed

class QuestManager:
    def __init__(self):
        self.quests = []
        logger.info("QuestManager initialized.")
    
    def add_quest(self, quest):
        self.quests.append(quest)
        logger.info(f"Quest '{quest.name}' added to QuestManager.")
    
    def remove_quest(self, quest):
        self.quests.remove(quest)
        logger.info(f"Quest '{quest.name}' removed from QuestManager.")
    
    def get_available_quests(self, agent):
        """Get the list of available quests for the agent based on their current state and progress."""
        available_quests = [quest for quest in self.quests if not quest.completed and self._is_quest_available(quest, agent)]
        logger.info(f"{len(available_quests)} available quests retrieved for agent.")
        return available_quests
    
    def update_quests(self, agent):
        """Update the state of quests based on the agent's actions and progress."""
        for quest in self.quests:
            if not quest.completed and quest.check_completion(agent):
                quest.grant_rewards(agent)
    
    def _is_quest_available(self, quest, agent):
        """Check if the quest is available for the agent based on specific criteria."""
        # Implement quest availability logic based on the agent's state, location, or other factors
        logger.debug(f"Checking availability for quest '{quest.name}' for agent.")
        return True
    
    def list_all_quests(self):
        """List all quests managed by the QuestManager."""
        for quest in self.quests:
            status = "completed" if quest.completed else "incomplete"
            logger.info(f"Quest: {quest.name}, Status: {status}")

    def get_quest_by_name(self, name):
        """Retrieve a quest by its name."""
        for quest in self.quests:
            if quest.name == name:
                logger.info(f"Quest '{name}' found in QuestManager.")
                return quest
        logger.warning(f"Quest '{name}' not found in QuestManager.")
        return None
    
    def save_quests(self, filepath):
        """Save the current state of quests to a file."""
        with open(filepath, 'w') as file:
            for quest in self.quests:
                file.write(f"{quest.name},{quest.description},{quest.completed}\n")
                for objective in quest.objectives:
                    file.write(f"Objective: {objective.description}\n")
                for reward in quest.rewards:
                    file.write(f"Reward: {reward.reward_type},{reward.value}\n")
        logger.info(f"Quest data saved to {filepath}")
    
    def load_quests(self, filepath, agent):
        """Load quests from a file and update the QuestManager."""
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        current_quest = None
        for line in lines:
            if line.startswith("Quest:"):
                parts = line.strip().split(',')
                name = parts[0].split(":")[1].strip()
                description = parts[1].strip()
                completed = parts[2].strip() == "True"
                current_quest = Quest(name, description, [], [])
                current_quest.completed = completed
                self.add_quest(current_quest)
            elif line.startswith("Objective:"):
                description = line.strip().split(":")[1].strip()
                objective = QuestObjective(description, lambda agent: False)  # Placeholder check function
                current_quest.objectives.append(objective)
            elif line.startswith("Reward:"):
                parts = line.strip().split(',')
                reward_type = parts[0].split(":")[1].strip()
                value = int(parts[1].strip())
                reward = QuestReward(reward_type, value)
                current_quest.rewards.append(reward)
        logger.info(f"Quest data loaded from {filepath}")

# Example Agent class for testing purposes
class Agent:
    def __init__(self):
        self.score = 0
        self.inventory = Inventory()
    
class Inventory:
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        self.items.append(item)
        logger.info(f"Item '{item}' added to inventory.")

# Example usage
if __name__ == "__main__":
    quest_manager = QuestManager()

    # Create some example quests
    quest1 = Quest(
        "First Quest",
        "Complete the first task",
        [QuestObjective("Find the hidden key", lambda agent: agent.score >= 10)],
        [QuestReward("score", 100)]
    )

    quest2 = Quest(
        "Second Quest",
        "Collect 5 magic stones",
        [QuestObjective("Collect 5 stones", lambda agent: len(agent.inventory.items) >= 5)],
        [QuestReward("item", "Magic Sword")]
    )

    quest_manager.add_quest(quest1)
    quest_manager.add_quest(quest2)

    agent = Agent()

    # Simulate agent progress
    agent.score = 10
    agent.inventory.add_item("Stone")
    agent.inventory.add_item("Stone")
    agent.inventory.add_item("Stone")
    agent.inventory.add_item("Stone")
    agent.inventory.add_item("Stone")

    # Update quests based on agent's progress
    quest_manager.update_quests(agent)

    # List all quests
    quest_manager.list_all_quests()

    # Save and load quests
    quest_manager.save_quests("quests.txt")
    quest_manager.load_quests("quests.txt", agent)
