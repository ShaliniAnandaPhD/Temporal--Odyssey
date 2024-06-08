import json

# Constants
LEVEL_UP_THRESHOLD = 100

class Skill:
    def __init__(self, name, category, level=1, experience=0, prerequisites=None, actions=None):
        """
        Initialize a new skill with a name, category, level, experience points, prerequisites, and actions.

        Parameters:
        name (str): The name of the skill.
        category (str): The category or type of the skill.
        level (int): The current level of the skill. Default is 1.
        experience (int): The current experience points of the skill. Default is 0.
        prerequisites (dict): A dictionary of prerequisite skills and their required levels. Default is None.
        actions (list): A list of actions or abilities associated with the skill. Default is None.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Skill name must be a non-empty string.")
        if not isinstance(category, str) or not category:
            raise ValueError("Skill category must be a non-empty string.")
        if not isinstance(level, int) or level < 1:
            raise ValueError("Skill level must be a positive integer.")
        if not isinstance(experience, int) or experience < 0:
            raise ValueError("Skill experience must be a non-negative integer.")

        self.name = name
        self.category = category
        self.level = level
        self.experience = experience
        self.prerequisites = prerequisites or {}
        self.actions = actions or []

    def gain_experience(self, amount):
        """
        Increase the experience points of the skill by a specified amount.
        If the experience points exceed the threshold, level up the skill.

        Parameters:
        amount (int): The amount of experience points to add.
        """
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("Experience amount must be a positive integer.")
        
        self.experience += amount
        if self.experience >= self.level * LEVEL_UP_THRESHOLD:
            self.level_up()

    def level_up(self):
        """
        Increase the level of the skill and reset experience points.
        """
        self.level += 1
        self.experience = 0
        print(f"{self.name} leveled up to level {self.level}!")

class AgentSkills:
    def __init__(self):
        """
        Initialize the AgentSkills class with a dictionary to store various skills.
        """
        self.skills = {}

    def add_skill(self, skill):
        """
        Add a new skill to the agent's skill set.

        Parameters:
        skill (Skill): The skill to add.
        """
        if not isinstance(skill, Skill):
            raise ValueError("Argument must be an instance of Skill.")
        
        if skill.name not in self.skills:
            self.skills[skill.name] = skill
            print(f"Added new skill: {skill.name}")
        else:
            print(f"Skill {skill.name} already exists.")

    def remove_skill(self, skill_name):
        """
        Remove an existing skill from the agent's skill set.

        Parameters:
        skill_name (str): The name of the skill to remove.
        """
        if skill_name in self.skills:
            del self.skills[skill_name]
            print(f"Removed skill: {skill_name}")
        else:
            print(f"Skill {skill_name} does not exist.")

    def modify_skill(self, skill_name, new_level=None, new_experience=None):
        """
        Modify an existing skill's level or experience.

        Parameters:
        skill_name (str): The name of the skill to modify.
        new_level (int): The new level to set. Default is None.
        new_experience (int): The new experience points to set. Default is None.
        """
        if skill_name in self.skills:
            if new_level is not None:
                if not isinstance(new_level, int) or new_level < 1:
                    raise ValueError("Skill level must be a positive integer.")
                self.skills[skill_name].level = new_level
            if new_experience is not None:
                if not isinstance(new_experience, int) or new_experience < 0:
                    raise ValueError("Skill experience must be a non-negative integer.")
                self.skills[skill_name].experience = new_experience
            print(f"Modified skill: {skill_name}")
        else:
            print(f"Skill {skill_name} does not exist.")

    def gain_skill_experience(self, skill_name, amount):
        """
        Increase the experience points for a specified skill.

        Parameters:
        skill_name (str): The name of the skill to gain experience.
        amount (int): The amount of experience points to add.
        """
        if skill_name in self.skills:
            self.skills[skill_name].gain_experience(amount)
        else:
            print(f"Skill {skill_name} does not exist. Add the skill first.")

    def get_skill_level(self, skill_name):
        """
        Get the current level of a specified skill.

        Parameters:
        skill_name (str): The name of the skill.

        Returns:
        int: The current level of the skill.
        """
        if skill_name in self.skills:
            return self.skills[skill_name].level
        else:
            print(f"Skill {skill_name} does not exist.")
            return None

    def get_skill_experience(self, skill_name):
        """
        Get the current experience points of a specified skill.

        Parameters:
        skill_name (str): The name of the skill.

        Returns:
        int: The current experience points of the skill.
        """
        if skill_name in self.skills:
            return self.skills[skill_name].experience
        else:
            print(f"Skill {skill_name} does not exist.")
            return None

    def get_all_skills(self):
        """
        Retrieve a dictionary of all skills and their details.

        Returns:
        dict: A dictionary of all skills with their details.
        """
        return {name: {"level": skill.level, "experience": skill.experience, "category": skill.category, "prerequisites": skill.prerequisites, "actions": skill.actions}
                for name, skill in self.skills.items()}

    def save_skills(self, filepath):
        """
        Save the agent's skills to a file.

        Parameters:
        filepath (str): The path to save the skills.
        """
        try:
            with open(filepath, 'w') as file:
                json.dump(self.get_all_skills(), file, indent=4)
            print(f"Skills saved to {filepath}")
        except IOError as e:
            print(f"Error saving skills: {e}")

    def load_skills(self, filepath):
        """
        Load the agent's skills from a file.

        Parameters:
        filepath (str): The path to load the skills from.
        """
        try:
            with open(filepath, 'r') as file:
                skills_data = json.load(file)
            for name, details in skills_data.items():
                skill = Skill(name, details["category"], details["level"], details["experience"], details["prerequisites"], details["actions"])
                self.skills[name] = skill
            print(f"Skills loaded from {filepath}")
        except IOError as e:
            print(f"Error loading skills: {e}")

# Example usage
if __name__ == "__main__":
    agent_skills = AgentSkills()
    try:
        crafting = Skill("Crafting", "Crafting Skills")
        combat = Skill("Combat", "Combat Skills")
        agent_skills.add_skill(crafting)
        agent_skills.add_skill(combat)
        agent_skills.gain_skill_experience("Crafting", 50)
        agent_skills.gain_skill_experience("Crafting", 60)  # Should trigger a level up
        print(f"Crafting Level: {agent_skills.get_skill_level('Crafting')}")
        print(f"Crafting Experience: {agent_skills.get_skill_experience('Crafting')}")
        agent_skills.gain_skill_experience("Combat", 20)
        print(f"Combat Level: {agent_skills.get_skill_level('Combat')}")
        print(f"Combat Experience: {agent_skills.get_skill_experience('Combat')}")

        # Adding new skills for different eras
        alchemy = Skill("Alchemy", "Alchemy Skills")
        archery = Skill("Archery", "Combat Skills")
        agent_skills.add_skill(alchemy)
        agent_skills.add_skill(archery)
        agent_skills.gain_skill_experience("Alchemy", 120)  # Should trigger a level up
        print(f"Alchemy Level: {agent_skills.get_skill_level('Alchemy')}")
        print(f"Alchemy Experience: {agent_skills.get_skill_experience('Alchemy')}")
        agent_skills.gain_skill_experience("Archery", 30)
        print(f"Archery Level: {agent_skills.get_skill_level('Archery')}")
        print(f"Archery Experience: {agent_skills.get_skill_experience('Archery')}")

        # Save and load skills
        agent_skills.save_skills("skills.json")
        agent_skills.load_skills("skills.json")
    except ValueError as e:
        print(f"Error: {e}")

