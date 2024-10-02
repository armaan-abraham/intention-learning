import numpy as np
import torch
from agent import Agent
from terminal import TerminalModel, TerminalNetwork

from data import CyclingEnvHandler, DataHandler, ExperienceBuffer

# Environment and training parameters
NUM_ENVS = 500
MAX_TIMESTEPS = int(5e8)
NUM_TIMESTEPS_PER_TRAIN = int(1e5)
NUM_TIMESTEPS_PER_SAVE = NUM_TIMESTEPS_PER_TRAIN * 10
TRAIN_BATCH_SIZE = int(1e5)
TRAINING_ITERATIONS = int(1e3)
TRAINING_BEGIN = 2
EXPERIENCE_BUFFER_SIZE = int(1e8)
TERMINAL_MODEL_ID = "e3359666"


class Logger:
    def __init__(self):
        from collections import deque

        self.reward_buffer = deque(maxlen=int(1e4))

    def add_rewards(self, rewards: torch.Tensor):
        self.reward_buffer.extend(list(rewards.detach().cpu()))

    def mean_reward(self) -> float:
        return np.mean(self.reward_buffer)

    def print_mean_reward(self):
        print(f"Mean reward: {self.mean_reward()}")


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
    terminal_network = data_handler.load_model(TERMINAL_MODEL_ID, TerminalNetwork).to(
        device
    )
    terminal_network.eval()
    terminal_model = TerminalModel(data_handler, device, network=terminal_network)
    logger = Logger()

    # Get environment specifications
    state_dim, action_dim, action_low, action_high = (
        data_handler.get_environment_specs()
    )

    # Initialize agent, judge, and replay buffer
    agent = Agent(state_dim, action_dim, action_low, action_high, device)

    # Reset environments and get initial states
    states = data_handler.reset_envs()

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
            logger=logger,
        )
        total_timesteps += new_timesteps

        # Train the agent
        if total_timesteps >= TRAINING_BEGIN * NUM_TIMESTEPS_PER_TRAIN:
            agent.train(
                data_handler,
                iterations=TRAINING_ITERATIONS,
                batch_size=TRAIN_BATCH_SIZE,
            )

        if total_timesteps % NUM_TIMESTEPS_PER_SAVE == 0:
            print("Saving")
            data_handler.save_model(agent.actor)
            data_handler.save_model(agent.actor_target)
            data_handler.save_model(agent.critic1)
            data_handler.save_model(agent.critic1_target)
            data_handler.save_model(agent.critic2)
            data_handler.save_model(agent.critic2_target)
            data_handler.save_buffer(experience_buffer)


def collect_experience(
    data_handler: DataHandler,
    agent: Agent,
    terminal_model: TerminalModel,
    num_timesteps_per_train: int,
    states: np.ndarray,
    total_timesteps: int,
    logger: Logger,
):
    """Collect experiences from the environment and store them."""
    timesteps_collected = 0

    while timesteps_collected < num_timesteps_per_train:
        actions = select_actions(data_handler, agent, states, total_timesteps)

        next_states, dones = data_handler.step_environments(actions)

        next_states_legacy = next_states.clone()
        next_states_legacy[:, 2] *= 8
        with torch.no_grad():
            rewards = terminal_model.predict(next_states_legacy)

        data_handler.save_experiences(states, actions, next_states, rewards, dones)
        logger.add_rewards(rewards)
        states = next_states
        timesteps_collected += NUM_ENVS

    logger.print_mean_reward()
    return states, timesteps_collected


def select_actions(
    data_handler: DataHandler, agent: Agent, states: torch.Tensor, total_timesteps: int
):
    if total_timesteps <= TRAINING_BEGIN * NUM_TIMESTEPS_PER_TRAIN:
        actions = data_handler.select_random_actions()
    else:
        actions = agent.select_action_with_noise(states)
        agent.decay_noise()
    assert actions.shape == (NUM_ENVS, 1), f"{actions.shape} != {(NUM_ENVS, 1)}"
    return actions


if __name__ == "__main__":
    main()
