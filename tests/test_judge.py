import torch
from intention_learning.judge import Judge, PromptAndParser
from intention_learning.img import ImageHandler
from intention_learning.data import DataHandler, StateBuffer, JudgmentBuffer, IMAGES_DIR
import os
import argparse
import base64
from io import BytesIO
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the buffers and data handler
state_buffer = StateBuffer(max_size=1000, device=device)
judgment_buffer = JudgmentBuffer(max_size=1000, device=device)
data_handler = DataHandler(state_buffer=state_buffer, judgment_buffer=judgment_buffer)

# Initialize the image handler and judge
image_handler = ImageHandler(image_dim=1500)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-encode-image', action='store_true', help='Test the encode_image_for_prompt function')
    args = parser.parse_args()

    if args.test_encode_image:
        # Generate a random state
        angle = torch.rand(2) * 2 * torch.pi
        velocity = torch.rand(2) * 16 - 8
        states = torch.stack([torch.cos(angle), torch.sin(angle), velocity], dim=1)

        # Store the state
        data_handler.store_states(states)

        # Create an image from the state
        states = states.cpu().numpy()
        img = image_handler.overlay_states_on_img(states[0], states[1])

        # Save the original image
        original_img_path = IMAGES_DIR / 'original_image.png'
        img.save(original_img_path)

        # Encode the image using encode_image_for_prompt
        encoded_image_data = judge.encode_image_for_prompt(img)

        # Encode the image using wrap_image_content
        wrapped_image_data = wrap_image_content(encode_image(original_img_path))

        print(encoded_image_data)
        print(wrapped_image_data)

        assert encoded_image_data == wrapped_image_data

    else:
        # Existing code for judging state pairs
        # Generate random states
        n_pairs = 10
        n_states = n_pairs * 2
        angles = torch.rand(n_states) * 2 * torch.pi
        velocities = torch.rand(n_states) * 16 - 8
        states = torch.stack([torch.cos(angles), torch.sin(angles), velocities], dim=1)
        
        # Store states in the data handler
        data_handler.store_states(states)
        
        # Create pairs of states
        states1 = states[::2]
        states2 = states[1::2]
        
        # Judge the state pairs
        judgments, images = judge.judge(states1, states2)
        
        # Save images and print predictions
        for i, (judgment, img) in enumerate(zip(judgments, images)):
            # Define the image path
            img_path = IMAGES_DIR / f"judgment_{i}.png"
            
            # Save the image to the appropriate directory
            img.save(img_path)
            
            # Print out the prediction
            print(f"Pair {i}: Judgment: {judgment}")
            print(f"Image saved to {img_path}\n")

