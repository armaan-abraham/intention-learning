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
url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
prompt = "Describe the image."

completion_request = ChatCompletionRequest(
    messages=[
        UserMessage(content=[ImageURLChunk(image_url=url), TextChunk(text=prompt)])
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
    temperature=0.35,
    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
)
result = tokenizer.decode(out_tokens[0])
