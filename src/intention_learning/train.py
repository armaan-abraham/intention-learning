import numpy as np
import torch
from pathlib import Path

from agent import Agent
from data import DataHandler, ExperienceBuffer, CyclingEnvHandler
from terminal import TerminalModel, TerminalNetwork

# Environment and training parameters
NUM_ENVS = 500
MAX_TIMESTEPS = int(1e8)
NUM_TIMESTEPS_PER_TRAIN = int(5e4)
NUM_TIMESTEPS_PER_SAVE = NUM_TIMESTEPS_PER_TRAIN * 10
TRAIN_BATCH_SIZE = int(5e4)
TRAINING_ITERATIONS = int(1e3)
EXPERIENCE_BUFFER_SIZE = int(5e7)
TERMINAL_MODEL_ID = "e3359666"

def main():
    """Main training loop for the RL agent interacting with the environment and using the judge model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_handler = CyclingEnvHandler(num_envs=NUM_ENVS, device=device)
    experience_buffer = ExperienceBuffer(max_size=EXPERIENCE_BUFFER_SIZE, device=device)
    # Initialize the managers
    data_handler = DataHandler(
        env_handler=env_handler,
        experience_buffer=experience_buffer,
    )
    terminal_network = data_handler.load_model(TERMINAL_MODEL_ID, TerminalNetwork)
    terminal_model = TerminalModel(data_handler, device, network=terminal_network)

    # Get environment specifications
    state_dim, action_dim, action_low, action_high = (
        data_handler.get_environment_specs()
    )

    # Initialize agent, judge, and replay buffer
    agent = Agent(state_dim, action_dim, action_low, action_high)

    # Reset environments and get initial states
    states = data_handler.reset()

    total_timesteps = 0
    while total_timesteps < MAX_TIMESTEPS:
        # Collect experience
        states, new_timesteps = collect_experience(
            data_handler=data_handler,
            agent=agent,
            terminal_model=terminal_model,
            num_timesteps_per_train=NUM_TIMESTEPS_PER_TRAIN,
            states=states,
            total_timesteps=total_timesteps,
        )
        total_timesteps += new_timesteps

        # Train the agent
        agent.train(data_handler, iterations=100, batch_size=256)

        if total_timesteps % NUM_TIMESTEPS_PER_SAVE == 0:
            print(f"Saving")
            data_handler.save_model(agent.actor)
            data_handler.save_model(agent.critic)
            data_handler.save_model(agent.critic_target)
            data_handler.save_buffer(experience_buffer)


def collect_experience(
    data_handler: DataHandler,
    agent: Agent,
    terminal_model: TerminalModel,
    num_timesteps_per_train: int,
    states: np.ndarray,
    total_timesteps: int,
):
    """Collect experiences from the environment and store them."""
    timesteps_collected = 0

    while timesteps_collected < num_timesteps_per_train:
        actions = select_actions(data_handler, agent, states, total_timesteps)

        next_states, dones = data_handler.step_environments(actions)

        next_states_legacy = next_states.copy()
        next_states_legacy[:, 2] *= 8
        rewards = terminal_model.get_rewards(next_states_legacy)

        data_handler.save_experiences(
            states, actions, next_states, rewards, rewards_ground_truth, dones
        )

        states = next_states
        timesteps_collected += NUM_ENVS

    return states, timesteps_collected


def select_actions(data_handler: DataHandler, agent: Agent, states: torch.Tensor, total_timesteps: int):
    if total_timesteps <= TRAINING_BEGIN * REPLAY_BUFFER_BATCH_SIZE:
        actions = data_handler.select_random_actions()
    else:
        actions = agent.select_action_with_noise(states).cpu().numpy()
        agent.decay_noise()
    assert actions.shape == (NUM_ENVS,)
    return actions

if __name__ == "__main__":
    main()
