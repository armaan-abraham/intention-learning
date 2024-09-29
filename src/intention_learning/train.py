# Main training script

import numpy as np
from pathlib import Path

from agent import TD3Agent
from judge import Judge
from data import DataHandler
from img import ImageHandler


def main():
    """Main training loop for the RL agent interacting with the environment and using the judge model."""
    # Environment and training parameters
    NUM_ENVS = 8
    MAX_TIMESTEPS = int(1e5)
    JUDGMENT_FREQUENCY = 10  # Judge every k steps
    SEED = 42
    NUM_TIMESTEPS_PER_TRAIN = 10000

    # Initialize the managers
    data_handler = DataHandler(
        num_envs=NUM_ENVS, seed=SEED, replay_buffer_size=int(2e6)
    )
    image_handler = ImageHandler()

    # Get environment specifications
    state_dim, action_dim, action_low, action_high = (
        data_handler.get_environment_specs()
    )

    # Initialize agent, judge, and replay buffer
    agent = TD3Agent(state_dim, action_dim, action_low, action_high)
    judge = Judge(model_path=Path("path/to/judge/model"), auth_token="YOUR_AUTH_TOKEN")

    total_timesteps = 0

    # Reset environments and get initial states
    states = data_handler.reset()

    while total_timesteps < MAX_TIMESTEPS:
        # Collect experience
        states, new_timesteps = collect_experience(
            data_handler=data_handler,
            agent=agent,
            judge=judge,
            num_timesteps_per_train=NUM_TIMESTEPS_PER_TRAIN,
            judgment_frequency=JUDGMENT_FREQUENCY,
            states=states,
            total_timesteps=total_timesteps,
        )
        total_timesteps += new_timesteps

        # Train the agent
        agent.train(data_handler, iterations=100, batch_size=256)

    # Save the trained agent
    agent.save(directory=Path("path/to/save/models"))

    # Close environments
    data_handler.close()


def collect_experience(
    data_handler: DataHandler,
    agent: TD3Agent,
    judge: Judge,
    image_handler: ImageHandler,
    num_timesteps_per_train: int,
    judgment_frequency: int,
    states: np.ndarray,
    total_timesteps: int,
):
    """Collect experiences from the environment and store them in the replay buffer."""
    timesteps_collected = 0

    for step in range(num_timesteps_per_train):
        actions = select_actions(agent, states, total_timesteps)

        next_states, rewards_ground_truth, dones = data_handler.step_environments(
            actions
        )

        # Obtain rewards from judge at specified frequency
        if step % judgment_frequency == 0:
            rewards = data_handler.get_rewards(judge, image_handler)
        else:
            rewards = np.zeros(data_handler.get_num_envs())

        data_handler.save_experiences(
            states, actions, next_states, rewards, rewards_ground_truth, dones
        )

        states = next_states
        timesteps_collected += data_handler.get_num_envs()

    return states, timesteps_collected


def select_actions(agent: TD3Agent, states: np.ndarray, total_timesteps: int):
    """Select actions with exploration noise."""
    # STILL TODO
    pass


if __name__ == "__main__":
    main()
