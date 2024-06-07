import json

class BehaviorConfig:
    def __init__(self, config_file="behavior_config.json"):
        """
        Initialize the BehaviorConfig class.

        Parameters:
        config_file (str): Path to the JSON configuration file.
        """
        self.config_file = config_file
        self.behaviors = self.load_config()

    def load_config(self):
        """
        Load the behavior configuration from the JSON file.

        Returns:
        dict: Dictionary containing behavior configurations.
        """
        try:
            with open(self.config_file, 'r') as file:
                behaviors = json.load(file)
                print(f"Loaded behavior configuration from {self.config_file}")
                return behaviors
        except FileNotFoundError:
            print(f"Configuration file {self.config_file} not found. Using default configuration.")
            return {}

    def save_config(self):
        """
        Save the current behavior configuration to the JSON file.
        """
        with open(self.config_file, 'w') as file:
            json.dump(self.behaviors, file, indent=4)
            print(f"Saved behavior configuration to {self.config_file}")

    def set_behavior(self, behavior_name, behavior_settings):
        """
        Set or update a specific behavior configuration.

        Parameters:
        behavior_name (str): Name of the behavior to set or update.
        behavior_settings (dict): Dictionary containing the settings for the behavior.
        """
        self.behaviors[behavior_name] = behavior_settings
        print(f"Set behavior '{behavior_name}' with settings: {behavior_settings}")
        self.save_config()

    def get_behavior(self, behavior_name):
        """
        Retrieve the configuration for a specific behavior.

        Parameters:
        behavior_name (str): Name of the behavior to retrieve.

        Returns:
        dict: Dictionary containing the settings for the behavior.
        """
        return self.behaviors.get(behavior_name, {})

    def list_behaviors(self):
        """
        List all configured behaviors.

        Returns:
        list: List of behavior names.
        """
        return list(self.behaviors.keys())

# Example usage
if __name__ == "__main__":
    config = BehaviorConfig()

    # Set a new behavior
    crafting_behavior = {
        "gathering_speed": 1.5,
        "resource_preference": "wood",
        "tool_usage": "axe"
    }
    config.set_behavior("Crafting", crafting_behavior)

    # Retrieve and print a behavior
    crafting_config = config.get_behavior("Crafting")
    print(f"Crafting Config: {crafting_config}")

    # List all behaviors
    all_behaviors = config.list_behaviors()
    print(f"All Behaviors: {all_behaviors}")

    # Example of saving and loading configuration
    combat_behavior = {
        "attack_style": "aggressive",
        "weapon_preference": "sword",
        "defense_strategy": "shield"
    }
    config.set_behavior("Combat", combat_behavior)
    print(f"Combat Config: {config.get_behavior('Combat')}")
    config.save_config()
    config.load_config()
