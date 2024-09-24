# download the model
from huggingface_hub import snapshot_download
from pathlib import Path
import os

auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
assert auth_token is not None, "HUGGING_FACE_HUB_TOKEN is not set"

mistral_models_path = Path.home().joinpath("mistral_models", "Mistral-7B-v0.3")
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.3",
    local_dir=mistral_models_path,
    token=auth_token,
)

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

import torch
import numpy as np

tokenizer = MistralTokenizer.from_file(
    f"{mistral_models_path}/tokenizer.model"
).instruct_tokenizer.tokenizer
model = Transformer.from_folder(mistral_models_path, dtype=torch.bfloat16).eval()


# Function to get embeddings
def get_embedding(input_text=None):
    encoded = tokenizer.encode(input_text, bos=True, eos=True)

    encoded_torch = torch.tensor(encoded).to(model.device)
    with torch.no_grad():
        output = model.forward_partial(encoded_torch, seqlens=[len(encoded)])

    return output.float()[-1].cpu().detach().numpy()


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


texts = [
    "Yosemite",
    "Object-oriented programming",
    "National Park",
    "Park",
    "Osama bin laden",
    "Fire",
]

# Run the model
embeddings = [get_embedding(input_text=text) for text in texts]


# Get embeddings
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        print(
            f"{texts[i]} -- {texts[j]}: {round(cosine_similarity(embeddings[i], embeddings[j]), 3)}"
        )
