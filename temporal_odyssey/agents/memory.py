class Memory:
    def __init__(self, capacity=100):
        """
        Initialize the Memory class.

        Parameters:
        capacity (int): The maximum number of memories to store. Default is 100.
        """
        self.capacity = capacity
        self.memory = []
        print(f"Memory initialized with capacity: {self.capacity}")

    def remember(self, event):
        """
        Store an event in memory. If memory is full, the oldest event is removed.

        Parameters:
        event (dict): A dictionary representing the event to remember.
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)  # Remove the oldest memory
        self.memory.append(event)
        print(f"Event remembered: {event}")

    def recall(self, criteria):
        """
        Recall events from memory that match the given criteria.

        Parameters:
        criteria (dict): A dictionary with keys and values to match in the memory.

        Returns:
        list: A list of remembered events that match the criteria.
        """
        recalled_events = [event for event in self.memory if all(event.get(k) == v for k, v in criteria.items())]
        print(f"Recalled events matching criteria {criteria}: {recalled_events}")
        return recalled_events

    def forget(self, criteria):
        """
        Forget events from memory that match the given criteria.

        Parameters:
        criteria (dict): A dictionary with keys and values to match in the memory.
        """
        self.memory = [event for event in self.memory if not all(event.get(k) == v for k, v in criteria.items())]
        print(f"Forgot events matching criteria {criteria}")

    def list_memory(self):
        """
        List all events currently stored in memory.

        Returns:
        list: A list of all remembered events.
        """
        print(f"Current memory: {self.memory}")
        return self.memory

# Example usage
if __name__ == "__main__":
    agent_memory = Memory(capacity=5)

    # Remember some events
    agent_memory.remember({"type": "interaction", "npc": "Bob", "outcome": "positive"})
    agent_memory.remember({"type": "interaction", "npc": "Alice", "outcome": "negative"})
    agent_memory.remember({"type": "discovery", "location": "forest", "item": "herb"})
    agent_memory.remember({"type": "battle", "enemy": "goblin", "outcome": "win"})
    agent_memory.remember({"type": "interaction", "npc": "Charlie", "outcome": "neutral"})
    
    # List current memory
    agent_memory.list_memory()

    # Recall specific events
    agent_memory.recall({"type": "interaction", "npc": "Bob"})
    agent_memory.recall({"outcome": "win"})

    # Forget specific events
    agent_memory.forget({"type": "interaction", "npc": "Alice"})

    # List current memory after forgetting some events
    agent_memory.list_memory()

    # Add more events to test memory capacity
    agent_memory.remember({"type": "exploration", "location": "cave", "finding": "treasure"})
    agent_memory.remember({"type": "interaction", "npc": "Daisy", "outcome": "positive"})

    # List current memory to see capacity handling
    agent_memory.list_memory()
