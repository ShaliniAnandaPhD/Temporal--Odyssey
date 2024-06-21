import random

class EnvironmentGenerator:
    def __init__(self):
        """
        Initialize the EnvironmentGenerator class.
        This class is responsible for generating various types of dynamic environments.
        """
        self.environments = []  # List to store all generated environments

    def generate_dynamic_environment(self):
        """
        Generate a dynamic environment.
        
        This method randomly selects attributes such as terrain type, weather condition,
        and obstacles to create a unique environment. The generated environment is then 
        added to the list of environments and returned.

        Returns:
        dict: The dynamically generated environment containing terrain, weather, and obstacles.
        """
        # Predefined lists of attributes for generating an environment
        terrain_types = ["forest", "desert", "mountain", "swamp"]
        weather_conditions = ["sunny", "rainy", "snowy", "stormy"]
        obstacles = ["rocks", "trees", "rivers", "lakes"]

        # Randomly select terrain, weather, and obstacles for the environment
        terrain = random.choice(terrain_types)
        weather = random.choice(weather_conditions)
        selected_obstacles = random.choices(obstacles, k=random.randint(1, 3))

        # Create the environment dictionary
        environment = {
            "terrain": terrain,
            "weather": weather,
            "obstacles": selected_obstacles
        }

        # Add the generated environment to the list of environments
        self.environments.append(environment)
        
        # Log the generated environment for debugging purposes
        print(f"Generated Environment: Terrain - {terrain}, Weather - {weather}, Obstacles - {selected_obstacles}")

        return environment

    def generate_multiple_environments(self, count=5):
        """
        Generate multiple dynamic environments.

        This method generates a specified number of environments and returns them as a list.

        Parameters:
        count (int): The number of environments to generate. Default is 5.

        Returns:
        list: A list of dynamically generated environments.
        """
        environments = []
        for _ in range(count):
            environments.append(self.generate_dynamic_environment())
        return environments

    def get_environment_by_index(self, index):
        """
        Retrieve a previously generated environment by its index.

        Parameters:
        index (int): The index of the environment to retrieve.

        Returns:
        dict: The environment at the specified index.
        """
        if 0 <= index < len(self.environments):
            return self.environments[index]
        else:
            print(f"No environment found at index {index}.")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize the environment generator
    env_generator = EnvironmentGenerator()

    # Generate a single dynamic environment
    environment = env_generator.generate_dynamic_environment()
    print("Single Generated Environment:", environment)

    # Generate multiple dynamic environments
    environments = env_generator.generate_multiple_environments(count=3)
    print("Multiple Generated Environments:", environments)

    # Retrieve a specific environment by its index
    specific_environment = env_generator.get_environment_by_index(1)
    print("Retrieved Environment at Index 1:", specific_environment)
