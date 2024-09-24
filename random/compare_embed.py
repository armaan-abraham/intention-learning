# download the model
import numpy as np
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

chat_tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
text_tokenizer = chat_tokenizer.instruct_tokenizer.tokenizer
model = Transformer.from_folder(mistral_models_path)

# Run the model
import base64
import os


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


from pathlib import Path
import torch


def get_embedding(encoded_text, encoded_image=[]):
    image_torch = (
        [torch.tensor(encoded_image[0], device=model.device, dtype=model.dtype)]
        if encoded_image
        else []
    )
    text_torch = torch.tensor(encoded_text, device=model.device, dtype=torch.long)
    with torch.no_grad():
        output = model.forward_partial(
            text_torch, images=image_torch, seqlens=[len(encoded_text)]
        )
    out_np = output.float().cpu().numpy()
    embedding = out_np[-1]
    return embedding


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


IMAGE_DIR = Path(__file__).parent.parent / "images"
image_files = list(IMAGE_DIR.glob("*.png"))
image_prompt = """In this image, there is an elf with a green hat, a brown stool, and
a gift box in a 4x4 grid of squares with (1,1) in the bottom left. In one
sentence, describe the location of the elf."""

template = """In a 4x4 coordinate system with (1,1) in the bottom left, an elf
with a green hat is located at {}. In one sentence, describe the location of the
elf."""
coords = ["(4,4)", "(1,4)", "(4,1)", "(1,1)", "(2,2)"]
text_embeddings = []
for coord in coords:
    completion_request = ChatCompletionRequest(
        messages=[UserMessage(content=[TextChunk(text=template.format(coord))])]
    )
    encoded = chat_tokenizer.encode_chat_completion(completion_request)
    text_embeddings.append(get_embedding(encoded.tokens, encoded.images))

image_embeddings = []

for image_file in image_files:
    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    ImageURLChunk(
                        image_url=wrap_image_content(encode_image(image_file))
                    ),
                    TextChunk(text=image_prompt),
                ]
            )
        ]
    )

    encoded = chat_tokenizer.encode_chat_completion(completion_request)

    image_embeddings.append(get_embedding(encoded.tokens, encoded.images))


for text, text_embed in zip(coords, text_embeddings):
    print()
    for file, image_embed in zip(image_files, image_embeddings):
        print(f"{text} {file}: {cosine_similarity(text_embed, image_embed)}")
