import random


class NPCDialogueManager:
    def __init__(self):
        self.dialogues = {
            "greetings": [
                "Hello, {player_name}! What brings you here?",
                "Greetings! How can I assist you today, {player_name}?",
                "Hey there, {player_name}! Need any help?",
                "Welcome, {player_name}! What can I do for you?"
            ],
            "farewells": [
                "Goodbye, {player_name}! Safe travels.",
                "Farewell, {player_name}! Until next time.",
                "See you around, {player_name}!",
                "Take care, {player_name}! Come back soon."
            ],
            "quests": {
                "fetch_quest": [
                    "I need you to gather 10 healing herbs from the forest.",
                    "Could you bring me 5 pieces of enchanted wood from the Dark Woods?",
                    "I'm looking for someone to collect 3 magical stones from the cave.",
                    "Please find and bring back 2 ancient artifacts from the ruins."
                ],
                "escort_quest": [
                    "Can you escort me to the village safely?",
                    "I need protection on my way to the market, will you help?",
                    "Please guide me through the haunted forest.",
                    "I need someone to accompany me to the castle."
                ],
                "defeat_monsters": [
                    "There are monsters causing trouble near the village. Can you take care of them?",
                    "We've spotted some dangerous creatures in the forest. Can you defeat them?",
                    "Help us by eliminating the beasts in the cave.",
                    "The town is under threat from marauding monsters. Will you defend us?"
                ]
            },
            "random_facts": [
                "Did you know this village was built over ancient ruins?",
                "Legend says there's a hidden treasure in the forest.",
                "People say the old castle is haunted.",
                "I've heard rumors of a powerful artifact buried nearby."
            ],
            "general_talk": [
                "The weather has been quite strange lately, hasn't it?",
                "Have you heard the latest news from the kingdom?",
                "The market is bustling with activity today.",
                "I could use a drink after all this excitement."
            ]
        }

    def format_dialogue(self, dialogue, player_name):
        if "{player_name}" in dialogue:
            return dialogue.format(player_name=player_name)
        return dialogue

    def get_random_dialogue(self, category, player_name=None, quest_type=None):
        if category == "quests" and quest_type:
            dialogues = self.dialogues.get(category, {}).get(quest_type, [])
        else:
            dialogues = self.dialogues.get(category, [])
        if not dialogues:
            return "I don't have anything to say right now."
        dialogue = random.choice(dialogues)
        return self.format_dialogue(dialogue, player_name)

    def get_dialogue(self, dialogue_type, player_name=None, quest_type=None):
        if dialogue_type == "greeting":
            return self.get_random_dialogue("greetings", player_name)
        elif dialogue_type == "farewell":
            return self.get_random_dialogue("farewells", player_name)
        elif dialogue_type == "quest":
            return self.get_random_dialogue("quests", player_name, quest_type)
        elif dialogue_type == "fact":
            return self.get_random_dialogue("random_facts")
        elif dialogue_type == "talk":
            return self.get_random_dialogue("general_talk")
        else:
            return "I'm sorry, I don't understand what you're asking for."

# Example usage:
if __name__ == "__main__":
    dialogue_manager = NPCDialogueManager()

    player_name = "John"

    print(dialogue_manager.get_dialogue("greeting", player_name))
    print(dialogue_manager.get_dialogue("quest", player_name, "fetch_quest"))
    print(dialogue_manager.get_dialogue("fact"))
    print(dialogue_manager.get_dialogue("talk"))
    print(dialogue_manager.get_dialogue("farewell", player_name))
