import numpy as np
from temporal_odyssey.envs.time_travel_env import TimeTravelEnv
from temporal_odyssey.agents.ppo_agent import PPOAgent  # Assuming PPOAgent is an advanced agent

def main():
    # Initialize the environment
    env = TimeTravelEnv()
    
    # Initialize the agent (using PPOAgent for a more sophisticated example)
    agent = PPOAgent(env)

    # Reset the environment to get the initial state
    initial_state = env.reset()
    print("Initial state:", initial_state)

    # Run the simulation for a set number of steps
    num_steps = 10  # You can adjust this to run for more steps

    for step in range(num_steps):
        # Agent decides on an action based on the current state
        action = agent.choose_action(initial_state)

        # Take a step in the environment
        next_state, reward, done = env.step(action)

        # Print out the results of the step
        print(f"Step {step + 1}:")
        print("Next state:", next_state)
        print("Reward:", reward)
        print("Done:", done)

        # Update the current state
        initial_state = next_state

        # If the environment says we're done, break the loop
        if done:
            break

    # Example visual, auditory, and textual data
    raw_visual_data = np.random.random((480, 640, 3))
    raw_auditory_data = np.random.random(22050)  # 1 second of audio at 22.05kHz
    raw_textual_data = ["This is a test sentence."]

    # Preprocess data using environment methods
    processed_visual_data = env.preprocess_visual_data(raw_visual_data)
    processed_auditory_data = env.preprocess_auditory_data(raw_auditory_data)
    processed_textual_data = env.preprocess_textual_data(raw_textual_data)

    # Print the shapes of the processed data
    print("Processed visual data shape:", processed_visual_data.shape)
    print("Processed auditory data shape:", processed_auditory_data.shape)
    print("Processed textual data shape:", processed_textual_data.shape)

if __name__ == "__main__":
    main()
