import argparse
import base64
import math

import torch

from intention_learning.data import IMAGES_DIR, DataHandler, JudgmentBuffer, StateBuffer
from intention_learning.img import ImageHandler
from intention_learning.judge import Judge, validate_judgments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the buffers and data handler
state_buffer = StateBuffer(max_size=1000, device=device)
judgment_buffer = JudgmentBuffer(max_size=1000, device=device)
data_handler = DataHandler(state_buffer=state_buffer, judgment_buffer=judgment_buffer)

# Initialize the image handler and judge
image_handler = ImageHandler(image_dim=450)
judge = Judge(image_handler=image_handler, data_handler=data_handler)


def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None


def wrap_image_content(image_b64: str) -> str:
    return f"data:image/jpeg;base64,{image_b64}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-encode-image",
        action="store_true",
        help="Test the encode_image_for_prompt function",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Batch size for judgments"
    )
    args = parser.parse_args()

    if args.test_encode_image:
        # Generate a random state
        angle = torch.rand(2) * 2 * torch.pi
        velocity = torch.rand(2) * 16 - 8
        states = torch.stack([torch.cos(angle), torch.sin(angle), velocity], dim=1)

        # Create an image from the state
        states = states.cpu().numpy()
        img = image_handler.overlay_states_on_img(states[0], states[1])

        # Save the original image
        original_img_path = IMAGES_DIR / "original_image.png"
        img.save(original_img_path)

    else:
        # Modified code for judging state pairs with batching
        n_pairs = 50
        n_states = n_pairs * 2
        angles = torch.rand(n_states) * 2 * torch.pi
        velocities = torch.rand(n_states) * 16 - 8
        states = torch.stack([torch.cos(angles), torch.sin(angles), velocities], dim=1)

        # Store states in the data handler
        data_handler.store_states(states)

        # Create pairs of states
        states1 = states[::2]
        states2 = states[1::2]

        # Judge the state pairs in batches
        batch_size = args.batch_size
        n_batches = math.ceil(n_pairs / batch_size)

        all_judgments = []
        all_images = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_pairs)

            batch_states1 = states1[start_idx:end_idx]
            batch_states2 = states2[start_idx:end_idx]

            batch_judgments = judge.judge(batch_states1, batch_states2)

            all_judgments.extend(batch_judgments)

        judgments = all_judgments

        is_valid_judgment = validate_judgments(
            states1, states2, torch.Tensor(judgments)[:, None]
        )
        print(f"percent valid: {is_valid_judgment.sum() / is_valid_judgment.numel()}")
