import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPCBehavior:
    def perform_interaction(self, agent):
        """
        Perform interaction with the agent.
        """
        pass

    def update(self, agent):
        """
        Update behavior based on agent's actions and progress.
        """
        pass

class FriendlyBehavior(NPCBehavior):
    def perform_interaction(self, agent):
        """
        Friendly interaction logic.
        """
        print("The NPC smiles warmly at you.")
        logger.info("Friendly interaction performed.")

    def update(self, agent):
        """
        Update logic for friendly behavior.
        """
        # Implement specific update logic for friendly behavior
        logger.info("Friendly behavior updated.")

class MysteriousBehavior(NPCBehavior):
    def perform_interaction(self, agent):
        """
        Mysterious interaction logic.
        """
        print("The NPC speaks in riddles.")
        logger.info("Mysterious interaction performed.")

    def update(self, agent):
        """
        Update logic for mysterious behavior.
        """
        # Implement specific update logic for mysterious behavior
        logger.info("Mysterious behavior updated.")

# Example usage
if __name__ == "__main__":
    class Agent:
        pass

    agent = Agent()

    friendly_behavior = FriendlyBehavior()
    mysterious_behavior = MysteriousBehavior()

    friendly_behavior.perform_interaction(agent)
    mysterious_behavior.perform_interaction(agent)

    friendly_behavior.update(agent)
    mysterious_behavior.update(agent)
