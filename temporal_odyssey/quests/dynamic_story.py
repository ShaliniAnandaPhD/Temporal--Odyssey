import logging
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicStory:
    def __init__(self):
        """
        Initialize the DynamicStory class.
        """
        self.story_events = defaultdict(list)
        self.current_story = []
        self.story_progress = 0
        logger.info("DynamicStory initialized.")

    def create_story_event(self, description, impact):
        """
        Create a new story event.

        Parameters:
        description (str): The description of the story event.
        impact (dict): The impact of the story event on the agent (e.g., {"health": -10, "experience": 5}).
        """
        event = {"description": description, "impact": impact}
        self.story_events[len(self.story_events)].append(event)
        logger.info(f"Story event created: {description} with impact {impact}.")

    def generate_dynamic_story(self, num_events):
        """
        Generate a dynamic story with a specified number of events.

        Parameters:
        num_events (int): The number of events in the story.
        """
        self.current_story = []
        for _ in range(num_events):
            event_id = random.choice(list(self.story_events.keys()))
            event = random.choice(self.story_events[event_id])
            self.current_story.append(event)
        logger.info(f"Dynamic story generated with {num_events} events.")

    def get_current_event(self):
        """
        Get the current story event.

        Returns:
        dict: The current story event.
        """
        if self.story_progress < len(self.current_story):
            return self.current_story[self.story_progress]
        else:
            return None

    def advance_story(self):
        """
        Advance to the next story event.
        """
        self.story_progress += 1
        if self.story_progress >= len(self.current_story):
            logger.info("Story has reached its end.")
        else:
            logger.info(f"Advanced to story event {self.story_progress}.")

    def visualize_story(self):
        """
        Visualize the generated story using a plot.
        """
        if not self.current_story:
            logger.warning("No story to visualize.")
            return
        
        descriptions = [event["description"] for event in self.current_story]
        impacts = [event["impact"] for event in self.current_story]

        fig, ax = plt.subplots(figsize=(12, 8))
        for i, (desc, imp) in enumerate(zip(descriptions, impacts)):
            ax.text(0.5, 1 - i*0.1, f"{i+1}. {desc} (Impact: {imp})", fontsize=12, va='center', ha='center', transform=ax.transAxes)

        ax.axis('off')
        plt.title('Dynamic Story Visualization')
        plt.show()
        logger.info("Story visualized.")

    def apply_story_impact(self, agent):
        """
        Apply the impact of the current story event to the agent.

        Parameters:
        agent (object): The agent to apply the impact to.
        """
        event = self.get_current_event()
        if event:
            for attribute, change in event["impact"].items():
                if hasattr(agent, attribute):
                    setattr(agent, attribute, getattr(agent, attribute) + change)
                    logger.info(f"Applied impact {change} to agent's {attribute}.")
                else:
                    logger.warning(f"Agent does not have attribute {attribute}.")
            self.advance_story()
        else:
            logger.warning("No current event to apply impact from.")

# Example usage
if __name__ == "__main__":
    class Agent:
        def __init__(self):
            self.health = 100
            self.experience = 0

    agent = Agent()
    dynamic_story = DynamicStory()

    # Create story events
    dynamic_story.create_story_event("You encounter a wild beast", {"health": -20, "experience": 10})
    dynamic_story.create_story_event("You find a hidden treasure", {"health": 0, "experience": 50})
    dynamic_story.create_story_event("You meet a wise sage", {"health": 10, "experience": 20})

    # Generate and visualize a dynamic story
    dynamic_story.generate_dynamic_story(5)
    dynamic_story.visualize_story()

    # Apply the story impacts to the agent
    for _ in range(5):
        event = dynamic_story.get_current_event()
        if event:
            print(f"Current Event: {event['description']} with impact {event['impact']}")
            dynamic_story.apply_story_impact(agent)
        else:
            print("Story has ended.")
            break

    # Display agent's final state
    print(f"Agent's final health: {agent.health}, experience: {agent.experience}")
