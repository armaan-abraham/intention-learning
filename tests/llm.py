# download the model
from huggingface_hub import snapshot_download
from pathlib import Path
import os

auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

mistral_models_path = Path.home().joinpath("mistral_models", "Pixtral")
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="mistralai/Pixtral-12B-2409",
    allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
    local_dir=mistral_models_path,
    token=auth_token,
)

# load the model
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    TextChunk,
    ImageURLChunk,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
model = Transformer.from_folder(mistral_models_path)

# Run the model
import base64
import os

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

def wrap_image_content(image_b64: str) -> str:
    return f"data:image/jpeg;base64,{image_b64}" 


from pathlib import Path


IMAGE_DIR = Path(__file__).parent.parent / "images"
image_files = IMAGE_DIR.glob("*.png")
prompt = """In this image, there is an elf with a green hat, a brown stool, and
a gift box in a 4x4 grid of squares. With the bottom left square being the
origin (position (1, 1)), respond with the coordinates of the elf in the image as
(x, y).  For example if the elf were in the top right square, you would respond
with (4, 4) and if he were in the top left square you would respond with
(1, 4). Respond ONLY with the coordinates and no other text."""

for image_file in image_files:
    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=[ImageURLChunk(image_url=wrap_image_content(encode_image(IMAGE_DIR / image_file))), TextChunk(text=prompt)])
        ]
    )

    encoded = tokenizer.encode_chat_completion(completion_request)

    images = encoded.images
    tokens = encoded.tokens

    out_tokens, _ = generate(
        [tokens],
        model,
        images=[images],
        max_tokens=256,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.decode(out_tokens[0])

    print(f"{image_file}: {result}")
