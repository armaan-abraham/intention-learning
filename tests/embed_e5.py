import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import numpy as np

auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
assert auth_token is not None, "HUGGING_FACE_HUB_TOKEN is not set"

mistral_models_path = Path.home().joinpath("mistral_models", "e5-mistral-7b-instruct")
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="intfloat/e5-mistral-7b-instruct",
    local_dir=mistral_models_path,
    token=auth_token,
)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# Each query must come with a one-sentence instruction that describes the task
input_texts = [
    "Yosemite",
    "Object-oriented programming",
    "National Park",
    "Park",
    "Osama bin laden",
    "Fire",
]

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')

max_length = 4096
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
embeddings = embeddings.cpu().numpy()

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

for i in range(len(input_texts)):
    for j in range(i + 1, len(input_texts)):
        print(f"Cosine similarity between {input_texts[i]} and {input_texts[j]}: {cosine_similarity(embeddings[i], embeddings[j])}")