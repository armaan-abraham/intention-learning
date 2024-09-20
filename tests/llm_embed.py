# download the model
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
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    TextChunk,
    ImageURLChunk,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import torch
import numpy as np

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
model = Transformer.from_folder(mistral_models_path).eval()


# Function to get embeddings
def get_embedding(input_text=None, input_image_url=None):
    content = []
    if input_image_url:
        content.append(ImageURLChunk(image_url=input_image_url))
    if input_text:
        content.append(TextChunk(text=input_text))
    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=content)
        ]
    )
    encoded = tokenizer.encode_chat_completion(completion_request)
    prompts = [encoded.tokens]
    images = [encoded.images]
    images_torch = [
        [torch.tensor(im, device=model.device, dtype=model.dtype) for im in images_for_sample]
        for images_for_sample in images
    ]
    flattened_images = sum(images_torch, []) 
    # TODO: eos id?
    seqlens = [len(x) for x in prompts]
    prompts_torch = torch.tensor(sum(prompts, []), device=model.device, dtype=torch.long)
    with torch.no_grad():
        output = model.forward_partial(prompts_torch, seqlens=seqlens, images=flattened_images)
    
    out_np = output.float().cpu().numpy()
    assert out_np.shape[0] == len(prompts[0])
    return out_np[-1]



def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Run the model
url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
text1 = "Yosemite national park"
text2 = "George Soros"
embedding_text1 = get_embedding(input_text=text1)
embedding_text2 = get_embedding(input_text=text2)
embedding_image = get_embedding(input_image_url=url)

# Get embeddings
print("text1, text2", cosine_similarity(embedding_text1, embedding_text2))
print("text1, image", cosine_similarity(embedding_text1, embedding_image))
print("text2, image", cosine_similarity(embedding_text2, embedding_image))
