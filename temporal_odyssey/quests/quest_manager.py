import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestManager:
    def __init__(self, quest_data_file="quests.json", agent_state_file="agent_state.json"):
        """
        Initialize the QuestManager class.

        Parameters:
        quest_data_file (str): Path to the JSON file containing quest data.
        agent_state_file (str): Path to the JSON file containing the agent's state.
        """
        self.quest_data_file = quest_data_file
        self.agent_state_file = agent_state_file
        self.quests = []
        self.load_quests()
        self.load_agent_state()
        logger.info("QuestManager initialized.")

    def load_quests(self):
        """
        Load quests from the JSON file.
        """
        try:
            with open(self.quest_data_file, 'r') as file:
                self.quests = json.load(file)
                logger.info("Quests loaded.")
        except FileNotFoundError:
            self.quests = []
            logger.warning("Quest data file not found. Initialized with empty quest list.")

    def save_quests(self):
        """
        Save the quests to the JSON file.
        """
        with open(self.quest_data_file, 'w') as file:
            json.dump(self.quests, file, indent=4)
            logger.info("Quests saved.")

    def load_agent_state(self):
        """
        Load the agent's state from the JSON file.
        """
        try:
            with open(self.agent_state_file, 'r') as file:
                self.agent_state = json.load(file)
                logger.info("Agent state loaded.")
        except FileNotFoundError:
            self.agent_state = {
                "completed_quests": [],
                "current_quests": [],
                "skills": {},
                "level": 1,
                "inventory": {},
                "quest_progress": {}
            }
            logger.warning("Agent state file not found. Initialized with default state.")

    def save_agent_state(self):
        """
        Save the agent's state to the JSON file.
        """
        with open(self.agent_state_file, 'w') as file:
            json.dump(self.agent_state, file, indent=4)
            logger.info("Agent state saved.")

    def update_agent_state(self, updates):
        """
        Update the agent's state with the provided updates.

        Parameters:
        updates (dict): Dictionary containing the updates to apply to the agent's state.
        """
        self.agent_state.update(updates)
        self.save_agent_state()
        logger.info(f"Agent state updated with: {updates}")

    def check_quest_availability(self):
        """
        Check available quests based on the agent's state and prerequisites.
        """
        available_quests = []
        for quest in self.quests:
            if quest["id"] not in self.agent_state["completed_quests"] and self._check_prerequisites(quest):
                available_quests.append(quest)
        logger.info(f"Available quests: {available_quests}")
        return available_quests

    def _check_prerequisites(self, quest):
        """
        Check if the agent meets the prerequisites for a quest.

        Parameters:
        quest (dict): The quest to check prerequisites for.

        Returns:
        bool: True if the agent meets the prerequisites, False otherwise.
        """
        prerequisites = quest.get("prerequisites", {})
        for key, value in prerequisites.items():
            if self.agent_state.get(key) < value:
                return False
        return True

    def accept_quest(self, quest_id):
        """
        Accept a quest and add it to the agent's current quests.

        Parameters:
        quest_id (int): The ID of the quest to accept.
        """
        for quest in self.quests:
            if quest["id"] == quest_id and quest not in self.agent_state["current_quests"]:
                self.agent_state["current_quests"].append(quest)
                self.agent_state["quest_progress"][quest_id] = {"started_at": datetime.now().isoformat(), "progress": {}}
                self.save_agent_state()
                logger.info(f"Quest accepted: {quest}")
                return True
        logger.warning(f"Quest with ID {quest_id} not found or already accepted.")
        return False

    def complete_quest(self, quest_id):
        """
        Complete a quest and provide the rewards to the agent.

        Parameters:
        quest_id (int): The ID of the quest to complete.
        """
        for quest in self.agent_state["current_quests"]:
            if quest["id"] == quest_id:
                self.agent_state["current_quests"].remove(quest)
                self.agent_state["completed_quests"].append(quest_id)
                self._apply_rewards(quest)
                self.save_agent_state()
                logger.info(f"Quest completed: {quest}")
                return True
        logger.warning(f"Quest with ID {quest_id} not found in current quests.")
        return False

    def _apply_rewards(self, quest):
        """
        Apply the rewards of a completed quest to the agent's state.

        Parameters:
        quest (dict): The completed quest.
        """
        rewards = quest.get("rewards", {})
        for key, value in rewards.items():
            if key in self.agent_state:
                self.agent_state[key] += value
            else:
                self.agent_state[key] = value
        logger.info(f"Rewards applied: {rewards}")

    def update_quest_progress(self, quest_id, progress_updates):
        """
        Update the progress of a current quest.

        Parameters:
        quest_id (int): The ID of the quest to update.
        progress_updates (dict): The progress updates to apply.
        """
        if quest_id in self.agent_state["quest_progress"]:
            self.agent_state["quest_progress"][quest_id]["progress"].update(progress_updates)
            self.save_agent_state()
            logger.info(f"Quest progress updated for quest ID {quest_id}: {progress_updates}")
        else:
            logger.warning(f"Quest with ID {quest_id} not found in current quest progress.")

    def generate_dynamic_quest(self):
        """
        Generate a dynamic quest based on the agent's actions, decisions, or the current state of the game world.
        """
        # Example: Generate a quest based on the agent's current level and skills
        quest = {
            "id": len(self.quests) + 1,
            "name": f"Dynamic Quest {len(self.quests) + 1}",
            "description": "A quest generated dynamically based on your current state.",
            "objectives": {"collect": 5, "eliminate": 3},
            "prerequisites": {"level": self.agent_state["level"]},
            "rewards": {"experience": 100, "gold": 50}
        }
        self.quests.append(quest)
        self.save_quests()
        logger.info(f"Dynamic quest generated: {quest}")
        return quest

    def get_quest_dialog(self, quest_id):
        """
        Get the dialog for a quest.

        Parameters:
        quest_id (int): The ID of the quest to get dialog for.

        Returns:
        str: The quest dialog.
        """
        for quest in self.quests:
            if quest["id"] == quest_id:
                return quest.get("dialog", "No dialog available for this quest.")
        logger.warning(f"Quest with ID {quest_id} not found.")
        return "Quest not found."

# Example usage
if __name__ == "__main__":
    quest_manager = QuestManager()

    # Accept a quest
    quest_manager.accept_quest(1)

    # Update quest progress
    quest_manager.update_quest_progress(1, {"collect": 3})

    # Complete a quest
    quest_manager.complete_quest(1)

    # Generate a dynamic quest
    quest_manager.generate_dynamic_quest()

    # Get quest dialog
    dialog = quest_manager.get_quest_dialog(1)
    print("Quest Dialog:", dialog)


