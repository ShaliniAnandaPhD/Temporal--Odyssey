import json
import logging
from cryptography.fernet import Fernet  # For encryption

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BehaviorConfig:
    def __init__(self, config_file="behavior_config.json", default_config=None, encryption_key=None):
        """
        Initialize the BehaviorConfig class.

        Parameters:
        config_file (str): Path to the JSON configuration file.
        default_config (dict): Default configuration values.
        encryption_key (bytes): Key for encrypting/decrypting the configuration file.
        """
        self.config_file = config_file
        self.default_config = default_config or {}
        self.encryption_key = encryption_key
        self.behaviors = self.load_config()

    def load_config(self):
        """
        Load the behavior configuration from the JSON file.

        Returns:
        dict: Dictionary containing behavior configurations.
        """
        try:
            with open(self.config_file, 'r') as file:
                data = file.read()
                if self.encryption_key:
                    data = self._decrypt(data)
                behaviors = json.loads(data)
                self._validate_config(behaviors)
                logger.info(f"Loaded behavior configuration from {self.config_file}")
                return behaviors
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_file} not found. Using default configuration.")
            return self.default_config
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON configuration file: {e}")
            return self.default_config

    def save_config(self):
        """
        Save the current behavior configuration to the JSON file.
        """
        try:
            data = json.dumps(self.behaviors, indent=4)
            if self.encryption_key:
                data = self._encrypt(data)
            with open(self.config_file, 'w') as file:
                file.write(data)
                logger.info(f"Saved behavior configuration to {self.config_file}")
        except IOError as e:
            logger.error(f"Error saving configuration file: {e}")

    def set_behavior(self, behavior_name, behavior_settings):
        """
        Set or update a specific behavior configuration.

        Parameters:
        behavior_name (str): Name of the behavior to set or update.
        behavior_settings (dict): Dictionary containing the settings for the behavior.
        """
        if not self._validate_behavior_settings(behavior_settings):
            logger.warning(f"Invalid behavior settings for '{behavior_name}'. Skipping update.")
            return

        self.behaviors[behavior_name] = behavior_settings
        logger.info(f"Set behavior '{behavior_name}' with settings: {behavior_settings}")
        self.save_config()

    def get_behavior(self, behavior_name):
        """
        Retrieve the configuration for a specific behavior.

        Parameters:
        behavior_name (str): Name of the behavior to retrieve.

        Returns:
        dict: Dictionary containing the settings for the behavior.
        """
        behavior_config = self.behaviors.get(behavior_name)
        if behavior_config is None:
            logger.warning(f"Behavior '{behavior_name}' not found in configuration.")
            return self.default_config.get(behavior_name, {})
        return behavior_config

    def list_behaviors(self):
        """
        List all configured behaviors.

        Returns:
        list: List of behavior names.
        """
        return list(self.behaviors.keys())

    def _validate_behavior_settings(self, behavior_settings):
        """
        Validate the behavior settings.

        Parameters:
        behavior_settings (dict): Dictionary containing the settings for the behavior.

        Returns:
        bool: True if the settings are valid, False otherwise.
        """
        # Example validation checks:
        if "gathering_speed" in behavior_settings and behavior_settings["gathering_speed"] <= 0:
            return False
        if "resource_preference" in behavior_settings and behavior_settings["resource_preference"] not in ["wood", "stone", "iron"]:
            return False
        return True

    def _validate_config(self, config):
        """
        Validate the entire configuration.

        Parameters:
        config (dict): Dictionary containing the configuration.
        """
        # Implement any necessary validation logic
        pass

    def _encrypt(self, data):
        """
        Encrypt the configuration data.

        Parameters:
        data (str): Data to encrypt.

        Returns:
        str: Encrypted data.
        """
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data.encode()).decode()

    def _decrypt(self, data):
        """
        Decrypt the configuration data.

        Parameters:
        data (str): Data to decrypt.

        Returns:
        str: Decrypted data.
        """
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(data.encode()).decode()

# Example usage
if __name__ == "__main__":
    default_config = {
        "Crafting": {
            "gathering_speed": 1.0,
            "resource_preference": "wood",
            "tool_usage": "axe"
        }
    }
    encryption_key = Fernet.generate_key()
    config = BehaviorConfig(default_config=default_config, encryption_key=encryption_key)

    # Set a new behavior
    crafting_behavior = {
        "gathering_speed": 1.5,
        "resource_preference": "stone",
        "tool_usage": "pickaxe"
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

