# %%
from huggingface_hub import snapshot_download
from pathlib import Path
import os

auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
assert auth_token is not None, "HUGGING_FACE_HUB_TOKEN is not set"

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

import torch
import numpy as np

tokenizer = MistralTokenizer.from_file(
    f"{mistral_models_path}/tekken.json"
).instruct_tokenizer.tokenizer
model = Transformer.from_folder(mistral_models_path).eval()


# Function to get embeddings
def get_embedding(input_text=None, input_image_url=None):
    encoded = tokenizer.encode(input_text, bos=True, eos=False)

    images = [encoded.images]
    images_torch = [
        [
            torch.tensor(im, device=model.device, dtype=model.dtype)
            for im in images_for_sample
        ]
        for images_for_sample in images
    ]
    flattened_images = sum(images_torch, [])
    encoded_torch = torch.tensor(encoded, device=model.device, dtype=torch.long)
    with torch.no_grad():
        output = model.forward_partial(
            encoded_torch, images=flattened_images, seqlens=[len(encoded)]
        )

    out_np = output.float().cpu().numpy()
    embedding = out_np[-1]
    return embedding


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


# Run the model
# url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"

texts = [
    "Yosemite",
    "Object-oriented programming",
    "National Park",
    "Park",
    "Osama bin laden",
    "Fire",
]

# Run the model
embeddings = [
    get_embedding(input_text=f"Describe this in detail: {text}") for text in texts
]

# Get embeddings
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        print(
            f"{texts[i]} -- {texts[j]}: {round(cosine_similarity(embeddings[i], embeddings[j]), 3)}"
        )
