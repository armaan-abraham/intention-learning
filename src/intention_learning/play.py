import argparse
import os
from pathlib import Path

import gymnasium as gym
import torch
from PIL import Image

from intention_learning.agent import Actor

DATA_DIR = Path(__file__).parent.parent.parent / "data"
GIFS_DIR = DATA_DIR / "videos"
MODEL_DIR = DATA_DIR / "models"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Play environment and optionally save animations."
    )
    parser.add_argument("id", type=str, help="ID of the model to play")
    args = parser.parse_args()

    # Create gifs folder if saving is enabled
    os.makedirs(GIFS_DIR, exist_ok=True)

    # Load the environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # Initialize the actor network
    actor = Actor(state_dim, action_dim, action_low, action_high, device)

    model_path = MODEL_DIR / args.id / "actor_target.pth"

    # Load the trained weights
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()  # Set the network to evaluation mode

    # Play episodes
    num_episodes = 10
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frames = []

        while not done:
            state[2] /= 8.0
            # Select action
            state_tensor = torch.Tensor(state).to(device)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()

            # Take action in the environment
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state

            # Render and optionally save frame
            frame = env.render()
            frames.append(Image.fromarray(frame))

        # Save episode as gif if enabled
        gif_path = GIFS_DIR / f"episode_{episode+1}.gif"
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=10, loop=0
        )
        print(f"Saved episode animation to {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
