import json
import logging
import random
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicStory:
    def __init__(self, agent_state_file="agent_state.json"):
        """
        Initialize the DynamicStory class.

        Parameters:
        agent_state_file (str): Path to the JSON file containing the agent's state.
        """
        self.agent_state_file = agent_state_file
        self.story_events = []
        self.story_progress = []
        self.load_agent_state()
        logger.info("DynamicStory initialized.")

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
                "location": "start",
                "time_in_game": 0
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

    def add_story_event(self, event):
        """
        Add a story event to the story events list.

        Parameters:
        event (dict): Dictionary containing the event details.
        """
        self.story_events.append(event)
        logger.info(f"Story event added: {event}")

    def trigger_event(self, condition):
        """
        Trigger a story event based on the specified condition.

        Parameters:
        condition (str): Condition to check for triggering the event.
        """
        for event in self.story_events:
            if event["condition"] == condition:
                self.story_progress.append(event)
                self.story_events.remove(event)
                self.update_agent_state(event["updates"])
                logger.info(f"Story event triggered: {event}")
                return event
        logger.info("No matching event found for condition.")
        return None

    def get_story_summary(self):
        """
        Get a summary of the story progress.

        Returns:
        list: List of story events that have occurred.
        """
        return self.story_progress

    def create_event(self, description, condition, updates, consequences=None):
        """
        Create a new story event.

        Parameters:
        description (str): Description of the event.
        condition (str): Condition that triggers the event.
        updates (dict): Updates to apply to the agent's state when the event is triggered.
        consequences (dict): Optional consequences of the event.

        Returns:
        dict: The created story event.
        """
        event = {
            "description": description,
            "condition": condition,
            "updates": updates,
            "consequences": consequences or {}
        }
        self.add_story_event(event)
        return event

    def create_branching_event(self, description, condition, branches):
        """
        Create a branching story event.

        Parameters:
        description (str): Description of the event.
        condition (str): Condition that triggers the event.
        branches (dict): Dictionary containing branches and their respective updates.

        Returns:
        dict: The created branching story event.
        """
        event = {
            "description": description,
            "condition": condition,
            "branches": branches
        }
        self.add_story_event(event)
        return event

    def choose_branch(self, event, choice):
        """
        Choose a branch for a branching story event.

        Parameters:
        event (dict): The branching story event.
        choice (str): The chosen branch.

        Returns:
        dict: The updates for the chosen branch.
        """
        if choice in event["branches"]:
            updates = event["branches"][choice]
            self.update_agent_state(updates)
            self.story_progress.append(event)
            self.story_events.remove(event)
            logger.info(f"Branch chosen: {choice} for event: {event}")
            return updates
        logger.warning(f"Invalid choice: {choice} for event: {event}")
        return {}

    def add_timed_event(self, description, condition, updates, deadline):
        """
        Add a timed story event.

        Parameters:
        description (str): Description of the event.
        condition (str): Condition that triggers the event.
        updates (dict): Updates to apply to the agent's state when the event is triggered.
        deadline (datetime): Deadline for the event.
        """
        event = {
            "description": description,
            "condition": condition,
            "updates": updates,
            "deadline": deadline
        }
        self.add_story_event(event)

    def check_timed_events(self):
        """
        Check for any timed events that should be triggered.
        """
        current_time = datetime.now()
        for event in self.story_events:
            if "deadline" in event and current_time >= event["deadline"]:
                self.story_progress.append(event)
                self.story_events.remove(event)
                self.update_agent_state(event["updates"])
                logger.info(f"Timed event triggered: {event}")

    def generate_dynamic_story(self):
        """
        Generate a dynamic story based on the agent's actions, decisions, and the current state of the game world.
        """
        # Example: Generate a random event based on the agent's location
        if self.agent_state["location"] == "forest":
            event = self.create_event(
                description="You encounter a mysterious figure in the forest.",
                condition="location_forest",
                updates={"location": "cave", "experience": 50},
                consequences={"message": "The figure leads you to a hidden cave."}
            )
            self.trigger_event(event["condition"])

        # Check for any timed events
        self.check_timed_events()

# Example usage
if __name__ == "__main__":
    dynamic_story = DynamicStory()

    # Create an example event
    dynamic_story.create_event(
        description="You find a hidden treasure chest.",
        condition="location_forest",
        updates={"gold": 100},
        consequences={"message": "You gain 100 gold."}
    )

    # Create a branching event
    dynamic_story.create_branching_event(
        description="You meet a fork in the road.",
        condition="location_forest",
        branches={
            "left": {"location": "village", "experience": 20},
            "right": {"location": "mountain", "experience": 30}
        }
    )

    # Trigger the event
    dynamic_story.trigger_event("location_forest")

    # Choose a branch
    branching_event = dynamic_story.story_events[-1]
    dynamic_story.choose_branch(branching_event, "left")

    # Generate dynamic story
    dynamic_story.generate_dynamic_story()

    # Get story summary
    summary = dynamic_story.get_story_summary()
    print("Story Summary:", summary)

