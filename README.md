# Temporal Odyssey

Temporal Odyssey is an immersive reinforcement learning project inspired by H.G. Wells' classic novel "The Time Machine." This project transports the agent into three distinct eras: the primitive past, the complex present, and the dystopian future. As the agent navigates through these time periods, it encounters unique challenges, dynamic environments, and evolving societal structures.

## Key Features

- **Immersive Time Travel Experience:** Embark on a journey through three meticulously crafted time periods, each with its own distinctive atmosphere, challenges, and rewards.
- **Dynamic Environments:** Explore rich, ever-changing environments that react to the agent's actions and evolve over time, creating a truly interactive and immersive experience.
- **Versatile Agent Capabilities:** Engage in a wide range of actions, including movement, interaction with objects and characters, and the ability to traverse through time to adapt to different eras.
- **Intelligent Reward System:** Benefit from a sophisticated reward mechanism that encourages exploration, promotes survival instincts, and drives the agent towards achieving specific goals within each time period.
- **Adaptive Challenges:** Face increasingly difficult obstacles and challenges that adapt to the agent's learning progress, ensuring a constant sense of growth and achievement.
- **Cutting-Edge Learning Techniques:** Harness the power of advanced reinforcement learning techniques, such as transfer learning and hierarchical learning, to enable the agent to learn efficiently and effectively across different time periods.

## Getting Started

To embark on your Temporal Odyssey, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/ShaliniAnandaPhD/Temporal--Odyssey.git
   ```

2. Navigate to the project directory:
   ```
   cd Temporal-Odyssey
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the virtual environment:
   - Create a new virtual environment:
     ```
     python -m venv venv
     ```
   - Activate the virtual environment:
     - For Windows:
       ```
       venv\Scripts\activate
       ```
     - For macOS and Linux:
       ```
       source venv/bin/activate
       ```

5. Run the Temporal Odyssey environment and train the agent:
   ```
   from temporal_odyssey.envs.time_travel_env import TimeTravelEnv
   from temporal_odyssey.agents.dqn_agent import DQNAgent

   env = TimeTravelEnv()
   agent = DQNAgent(env)
   agent.train()
   ```

## Contributing

We welcome contributions from the community to enhance Temporal Odyssey. If you have any ideas, suggestions, or bug reports, please open an issue on the GitHub repository. If you'd like to contribute code, you can fork the repository and submit a pull request with your changes.

## License

Temporal Odyssey is released under the [MIT License](https://opensource.org/licenses/MIT), granting you the freedom to use, modify, and distribute the project as per the terms of the license.

Prepare to embark on an extraordinary journey through time with Temporal Odyssey. Explore, learn, and adapt as you navigate the challenges and wonders of different eras. Let the adventure begin!
