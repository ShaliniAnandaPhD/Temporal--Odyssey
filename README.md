**Temporal Odyssey** is an immersive reinforcement learning project inspired by H.G. Wells' "The Time Machine." Navigate through distinct time periods, facing unique challenges and dynamic environments that test your adaptability and survival skills.

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Concepts](#concepts)
- [Environment Configuration and Models](#environment-configuration-and-models)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Citation](#citation)

## Key Features

- Immersive time travel experience across multiple eras
- Dynamic, evolving environments
- Multimodal agent capabilities with era-specific actions
- Intelligent reward system
- Adaptive challenges
- Advanced learning techniques (PPO, A3C, transfer learning, meta-learning)
- NPC interactions and quest system

## Prerequisites

- Python 3.7+
- `pip`
- `virtualenv` (recommended)
- Certain dependencies might have specific version requirements. Check `requirements.txt` for details.

## Installation

### Clone the repository

```bash
git clone https://github.com/ShaliniAnandaPhD/Temporal-Odyssey.git
cd Temporal-Odyssey
```

### Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install the required dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

To run a basic simulation:

1. Navigate to the project directory:

    ```bash
    cd temporal_odyssey
    ```

2. Run the main script:

    ```bash
    python main.py
    ```

This will start a basic simulation using default settings. You'll see output describing the agent's actions and rewards as it navigates through different time periods. Expect to see how the agent adapts and learns in dynamic environments with era-specific challenges.

## Project Structure

- **temporal_odyssey/**
  - **envs/**: Environment definitions
    - `time_travel_env.py`: Main time travel environment
  - **agents/**: Agent implementations
    - `ppo_agent.py`: Proximal Policy Optimization agent
    - `a3c_agent.py`: Asynchronous Advantage Actor-Critic agent
  - **models/**: Learning models
    - `transfer_learning.py`: Transfer learning implementation
    - `meta_learning.py`: Meta-learning techniques
  - **quests/**: Quest and story management
  - **npcs/**: Non-player character implementations
  - **monitoring/**: Performance monitoring and telemetry
  - `main.py`: Entry point for running simulations

- **docs/**: Documentation and guides
- **tests/**: Test cases and testing framework
- **ui/**: User interface components

## Advanced Usage

To customize the simulation or use specific learning techniques:

```python
from temporal_odyssey.envs.time_travel_env import TimeTravelEnv
from temporal_odyssey.agents.ppo_agent import PPOAgent
from temporal_odyssey.models.transfer_learning import TransferLearning

# Create environment and agent
env = TimeTravelEnv(start_era="prehistoric", difficulty="hard")
agent = PPOAgent(env, learning_rate=0.001)

# Initialize transfer learning
transfer_model = TransferLearning()

# Train the agent
agent.train(num_episodes=1000, transfer_model=transfer_model)

# Test the agent
agent.test(num_episodes=100)
```

For more detailed examples and API documentation, refer to the `docs/` directory.

## Concepts

- **PPO (Proximal Policy Optimization)**: An advanced reinforcement learning algorithm that balances exploration and exploitation.
- **A3C (Asynchronous Advantage Actor-Critic)**: A deep reinforcement learning algorithm that uses asynchronous gradient descent for more efficient learning.

## Environment Configuration and Models

### Environment Configuration

The Temporal Odyssey environment is fully simulated and doesn't require external datasets. To configure the environment:

1. Navigate to the `config/` directory in your project root.
2. Modify `environment_config.yaml` to adjust parameters such as:
   - Time periods available
   - Difficulty levels
   - Reward structures
   - NPC behavior patterns

Example:

```yaml
time_periods:
  - prehistoric
  - medieval
  - modern
  - future
difficulty: medium
reward_scaling: 1.0
npc_density: 0.5
```

### Models

Temporal Odyssey uses reinforcement learning models that are trained within the simulated environment:

1. To start with a fresh model:

   ```bash
   python temporal_odyssey/models/initialize_model.py --agent_type ppo
   ```

   This will create a new model in the `models/` directory.

2. To continue training from a checkpoint:

   ```bash
   python temporal_odyssey/models/train_model.py --agent_type ppo --checkpoint models/ppo_latest.pth
   ```

3. To evaluate a trained model:

   ```bash
   python temporal_odyssey/models/evaluate_model.py --agent_type ppo --model_path models/ppo_best.pth
   ```

Note: The environment and models are generated and evolved during the reinforcement learning process. No pre-trained models or external datasets are required to get started.

## Troubleshooting

### Common Issues:

1. **Environment Setup Errors**:
   - Ensure all dependencies are correctly installed using `pip install -r requirements.txt`.
   - Check that your Python version is 3.7+ as specified in the prerequisites.

2. **Runtime Errors**:
   - Verify that the configuration files in `config/` are correctly formatted and contain valid parameters.
   - Ensure all paths in your scripts are correct relative to the project root.

3. **GPU-related Issues**:
   - If using GPU acceleration, check that CUDA is properly installed and compatible with your PyTorch version.
   - Try running on CPU if GPU issues persist: add `--device cpu` to your run commands.

4. **Model Training Problems**:
   - If training is unstable, try adjusting hyperparameters in `config/training_config.yaml`.
   - For out-of-memory errors, reduce batch size or model complexity.

5. **Rendering Issues**:
   - If the environment fails to render, check that all required visualization libraries are installed.
   - Ensure your system supports the rendering backend (e.g., OpenGL).

For more detailed troubleshooting:
- Check the logs in the `logs/` directory for error messages and stack traces.
- Refer to `docs/troubleshooting.md` for an extensive guide on resolving common issues.
- If problems persist, please open an issue on our GitHub repository with a detailed description of the error and your system configuration.

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

To run tests:

```bash
python -m unittest tests/test_env.py
```

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

- [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1810.06339)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Meta-Learning: A Survey](https://arxiv.org/abs/1810.03548)
- [A Survey on Transfer Learning](https://arxiv.org/abs/0907.0209)

## Ongoing work

Please refer to the Notion page. 


## Citation

If you use this project in your research, please cite it as follows:

```
@misc{ananda2024temporalodyssey,
  author = {Shalini Ananda},
  title = {Temporal Odyssey: Embark on a Journey Through Time},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ShaliniAnandaPhD/Temporal-Odyssey}}
}
```
