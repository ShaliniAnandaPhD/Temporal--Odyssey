

import logging
from temporal_odyssey.npcs.npc_behaviors import FriendlyBehavior, MysteriousBehavior

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPC:
    def __init__(self, name, description, dialogue, behavior):
        self.name = name
        self.description = description
        self.dialogue = dialogue
        self.behavior = behavior
        logger.info(f"NPC '{self.name}' created with behavior '{self.behavior.__class__.__name__}'.")

    def interact(self, agent):
        """
        Handle interaction logic between the NPC and the agent.
        """
        print(f"{self.name}: {self.dialogue}")
        self.behavior.perform_interaction(agent)
        logger.info(f"{self.name} interacted with agent. Behavior: {self.behavior.__class__.__name__}")

class Merchant(NPC):
    def __init__(self, name, description, dialogue, inventory, behavior):
        super().__init__(name, description, dialogue, behavior)
        self.inventory = inventory
        logger.info(f"Merchant '{self.name}' with inventory created.")

    def trade(self, agent):
        """
        Implement the trading logic between the merchant and the agent.
        """
        print(f"{self.name}: Welcome to my shop! Here are the available items:")
        for item in self.inventory:
            print(f"- {item.name} (Price: {item.price})")
        # Simulate a trade
        chosen_item = self.inventory[0]  # This is just a placeholder
        if agent.gold >= chosen_item.price:
            agent.gold -= chosen_item.price
            agent.inventory.add_item(chosen_item)
            print(f"You bought {chosen_item.name} for {chosen_item.price} gold.")
            logger.info(f"Agent bought {chosen_item.name} from {self.name}.")
        else:
            print("You don't have enough gold.")
            logger.warning(f"Agent could not afford {chosen_item.name}.")

class QuestGiver(NPC):
    def __init__(self, name, description, dialogue, quests, behavior):
        super().__init__(name, description, dialogue, behavior)
        self.quests = quests
        logger.info(f"QuestGiver '{self.name}' with quests created.")

    def offer_quests(self, agent):
        """
        Implement the quest offering logic between the quest giver and the agent.
        """
        print(f"{self.name}: I have the following quests available for you:")
        for quest in self.quests:
            print(f"- {quest.name}: {quest.description}")
        # Simulate quest acceptance
        accepted_quest = self.quests[0]  # This is just a placeholder
        agent.accept_quest(accepted_quest)
        print(f"You accepted the quest: {accepted_quest.name}")
        logger.info(f"Agent accepted quest '{accepted_quest.name}' from {self.name}.")

class NPCManager:
    def __init__(self):
        self.npcs = []
        logger.info("NPCManager initialized.")

    def add_npc(self, npc):
        self.npcs.append(npc)
        logger.info(f"NPC '{npc.name}' added to NPCManager.")

    def remove_npc(self, npc):
        self.npcs.remove(npc)
        logger.info(f"NPC '{npc.name}' removed from NPCManager.")

    def get_npc_by_name(self, name):
        for npc in self.npcs:
            if npc.name == name:
                logger.info(f"NPC '{name}' found in NPCManager.")
                return npc
        logger.warning(f"NPC '{name}' not found in NPCManager.")
        return None

    def update_npcs(self, agent):
        """
        Update the state of NPCs based on the agent's actions and progress.
        """
        for npc in self.npcs:
            if isinstance(npc, Merchant):
                logger.debug(f"Updating Merchant '{npc.name}' based on agent's trade actions.")
            elif isinstance(npc, QuestGiver):
                logger.debug(f"Updating QuestGiver '{npc.name}' based on agent's quest progress.")
            npc.behavior.update(agent)

    def list_all_npcs(self):
        """
        List all NPCs managed by the NPCManager.
        """
        for npc in self.npcs:
            logger.info(f"NPC: {npc.name}, Description: {npc.description}")

    def save_npcs(self, filepath):
        """
        Save the current state of NPCs to a file.
        """
        with open(filepath, 'w') as file:
            for npc in self.npcs:
                file.write(f"{npc.name},{npc.description},{npc.dialogue},{npc.behavior.__class__.__name__}\n")
                if isinstance(npc, Merchant):
                    for item in npc.inventory:
                        file.write(f"Item:{item.name},{item.price}\n")
                elif isinstance(npc, QuestGiver):
                    for quest in npc.quests:
                        file.write(f"Quest:{quest.name},{quest.description}\n")
        logger.info(f"NPC data saved to {filepath}")

    def load_npcs(self, filepath):
        """
        Load NPCs from a file and update the NPCManager.
        """
        with open(filepath, 'r') as file:
            lines = file.readlines()

        current_npc = None
        for line in lines:
            if not line.startswith("Item:") and not line.startswith("Quest:"):
                name, description, dialogue, behavior_name = line.strip().split(',')
                behavior = eval(f"{behavior_name}()")
                if "Merchant" in description:
                    current_npc = Merchant(name, description, dialogue, [], behavior)
                elif "QuestGiver" in description:
                    current_npc = QuestGiver(name, description, dialogue, [], behavior)
                else:
                    current_npc = NPC(name, description, dialogue, behavior)
                self.add_npc(current_npc)
            elif line.startswith("Item:"):
                name, price = line.strip().split(',')[1:]
                item = Item(name, int(price))
                current_npc.inventory.append(item)
            elif line.startswith("Quest:"):
                name, description = line.strip().split(',')[1:]
                quest = Quest(name, description, [], [])
                current_npc.quests.append(quest)
        logger.info(f"NPC data loaded from {filepath}")

class Item:
    def __init__(self, name, price):
        self.name = name
        self.price = price

class Agent:
    def __init__(self):
        self.gold = 100
        self.inventory = Inventory()

    def accept_quest(self, quest):
        print(f"Quest '{quest.name}' accepted.")
        logger.info(f"Agent accepted quest '{quest.name}'.")

class Inventory:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)
        logger.info(f"Item '{item.name}' added to inventory.")

# Example usage
if __name__ == "__main__":
    from temporal_odyssey.quests.quest_manager import Quest, QuestReward, QuestObjective

    npc_manager = NPCManager()

    # Create some example NPCs
    merchant_inventory = [Item("Health Potion", 10), Item("Mana Potion", 15)]
    quest_giver_quests = [
        Quest("First Quest", "Complete the first task", [QuestObjective("Find the hidden key", lambda agent: agent.score >= 10)], [QuestReward("score", 100)]),
        Quest("Second Quest", "Collect 5 magic stones", [QuestObjective("Collect 5 stones", lambda agent: len(agent.inventory.items) >= 5)], [QuestReward("item", "Magic Sword")])
    ]

    merchant = Merchant("Bob the Merchant", "A friendly merchant", "Would you like to buy something?", merchant_inventory, FriendlyBehavior())
    quest_giver = QuestGiver("Alice the Quest Giver", "A mysterious quest giver", "I have some tasks for you.", quest_giver_quests, MysteriousBehavior())

    npc_manager.add_npc(merchant)
    npc_manager.add_npc(quest_giver)

    agent = Agent()

    # Simulate interactions
    merchant.interact(agent)
    merchant.trade(agent)
    quest_giver.interact(agent)
    quest_giver.offer_quests(agent)

    # List all NPCs
    npc_manager.list_all_npcs()

    # Save and load NPCs
    npc_manager.save_npcs("npcs.txt")
    npc_manager.load_npcs("npcs.txt")

