# Temporal Odyssey

**Temporal Odyssey** is an immersive reinforcement learning project that guides you through various eras inspired by H.G. Wells' "The Time Machine." As an agent, you navigate through distinct time periods, facing unique challenges and dynamic environments that test your adaptability and survival skills.

## Key Features

- **Immersive Time Travel Experience**
  - Explore meticulously crafted time periods.
  - Each era has its own atmosphere, challenges, and rewards.
  - Travel from the primitive past to a technologically advanced future.

- **Dynamic Environments**
  - Interact with evolving environments that react to your actions.
  - Shape the world around you through your interactions with objects and characters.
  
- **Versatile Agent Capabilities**
  - Perform era-specific actions:
    - Navigate treacherous terrains.
    - Gather resources, craft tools, and build shelters in the primitive past.
    - Trade goods and make strategic decisions in the present.
    - Scavenge supplies and navigate dangers in dystopian futures.
  
- **Intelligent Reward System**
  - Sophisticated rewards encourage exploration and survival.
  - Actions and decisions have consequences, shaping your path through history.
  
- **Adaptive Challenges**
  - Face obstacles that evolve with your learning progress.
  - Ensure continuous growth and replayability.
  
- **Advanced Learning Techniques**
  - Utilize cutting-edge reinforcement learning methods like PPO, A3C, transfer learning, and meta-learning.
  - Leverage knowledge from one era to adapt quickly to another.
  
- **NPC Interactions**
  - Engage with non-player characters across different eras.
  - Gather information, trade items, and form alliances to aid your journey.
  
- **Quest System**
  - Embark on quests and missions within each era.
  - Complete objectives, unravel mysteries, and earn rewards to unlock new possibilities.

## Installation

To start your journey through Temporal Odyssey, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/ShaliniAnandaPhD/Temporal-Odyssey.git
   ```

2. **Navigate to the project directory:**
   ```sh
   cd Temporal-Odyssey
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up the virtual environment:**
   - Create a new virtual environment:
     ```sh
     python -m venv venv
     ```
   - Activate the virtual environment:
     - For Windows:
       ```sh
       venv\Scripts\activate
       ```
     - For macOS and Linux:
       ```sh
       source venv/bin/activate
       ```

## Getting Started

To begin your adventure through time:

1. **Run the Temporal Odyssey environment and train your agent:**

   ```python
   from temporal_odyssey.envs.time_travel_env import TimeTravelEnv
   from temporal_odyssey.agents.ppo_agent import PPOAgent
   from temporal_odyssey.agents.a3c_agent import A3CAgent
   from temporal_odyssey.models.transfer_learning import TransferLearning
   from temporal_odyssey.models.meta_learning import MetaLearning

   env = TimeTravelEnv()
   ppo_agent = PPOAgent(env)
   a3c_agent = A3CAgent(env)
   transfer_learning = TransferLearning()
   meta_learning = MetaLearning()

   # Train the agents using advanced techniques
   ppo_agent.train(transfer_learning, meta_learning)
   a3c_agent.train(transfer_learning, meta_learning)
   ```

2. **Customize your agent and experiment with different learning algorithms.**

## Project Structure

- **`temporal_odyssey/envs/`**
  - **`time_travel_env.py`**: Defines the `TimeTravelEnv` class, representing the time travel environment.

- **`temporal_odyssey/agents/`**
  - **`dqn_agent.py`**: Implements the DQN agent with advanced techniques.
  - **`ppo_agent.py`**: Implements the PPO agent.
  - **`a3c_agent.py`**: Implements the A3C agent.

- **`temporal_odyssey/models/`**
  - **`transfer_learning.py`**: Implements transfer learning functionality.
  - **`meta_learning.py`**: Implements meta-learning techniques.

- **`temporal_odyssey/quests/`**
  - **`quest_manager.py`**: Defines the `QuestManager` class for managing quests and missions.

- **`temporal_odyssey/npcs/`**
  - **`npc_manager.py`**: Defines the `NPCManager` class for managing NPCs and their interactions.

- **`README.md`**: Overview and instructions for getting started.

- **`requirements.txt`**: Dependencies required to run the project.

## Contributing

- **Contributions**:
  - Welcome contributions from the community.
  - Open an issue on GitHub for ideas, suggestions, or bug reports.
  - Fork the repository and submit a pull request with your changes.

## License

- **MIT License**:
  - Temporal Odyssey is released under the [MIT License](https://opensource.org/licenses/MIT).
  - You are free to use, modify, and distribute the project as per the license terms.

Prepare to embark on an extraordinary journey through time with Temporal Odyssey. Explore, learn, and adapt as you navigate the challenges and wonders of different eras. Let the adventure begin!

[GitHub Repository](https://github.com/ShaliniAnandaPhD/Temporal--Odyssey)
