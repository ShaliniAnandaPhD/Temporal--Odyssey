class Skill:
    def __init__(self, name, level=1, experience=0):
        """
        Initialize a new skill with a name, level, and experience points.

        Parameters:
        name (str): The name of the skill.
        level (int): The current level of the skill. Default is 1.
        experience (int): The current experience points of the skill. Default is 0.
        """
        self.name = name
        self.level = level
        self.experience = experience
    
    def gain_experience(self, amount):
        """
        Increase the experience points of the skill by a specified amount.
        If the experience points exceed the threshold, level up the skill.

        Parameters:
        amount (int): The amount of experience points to add.
        """
        self.experience += amount
        if self.experience >= self.level * 100:  # Example threshold for leveling up
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
    
    def add_skill(self, skill_name):
        """
        Add a new skill to the agent's skill set.

        Parameters:
        skill_name (str): The name of the skill to add.
        """
        if skill_name not in self.skills:
            self.skills[skill_name] = Skill(skill_name)
            print(f"Added new skill: {skill_name}")
        else:
            print(f"Skill {skill_name} already exists.")
    
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

# Example usage
if __name__ == "__main__":
    agent_skills = AgentSkills()
    agent_skills.add_skill("Crafting")
    agent_skills.add_skill("Combat")
    agent_skills.gain_skill_experience("Crafting", 50)
    agent_skills.gain_skill_experience("Crafting", 60)  # Should trigger a level up
    print(f"Crafting Level: {agent_skills.get_skill_level('Crafting')}")
    print(f"Crafting Experience: {agent_skills.get_skill_experience('Crafting')}")
    agent_skills.gain_skill_experience("Combat", 20)
    print(f"Combat Level: {agent_skills.get_skill_level('Combat')}")
    print(f"Combat Experience: {agent_skills.get_skill_experience('Combat')}")
