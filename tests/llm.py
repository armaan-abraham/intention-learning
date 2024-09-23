# %%
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

# %%
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

import base64
import os

# %%

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


IMAGE_DIR = Path(__file__).parent.parent / "paired_images"
image_files = list(IMAGE_DIR.glob("*.png"))
prompt = """In which frame is the elf closer to the gift box?""".replace("\n", " ")

truth = []
predict = []
results = []

for image_file in image_files:
    file1, file2 = image_file.name.split("_")
    format_coord = lambda filename: f"({filename.replace('-', ',').replace('.png', '')})"
    coord1, coord2 = format_coord(file1), format_coord(file2)

    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    ImageURLChunk(
                        image_url=wrap_image_content(
                            encode_image(IMAGE_DIR / image_file)
                        )
                    ),
                    TextChunk(text=prompt),
                ]
            )
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

    results.append(result)

    # answer = result.split("'")[1]
    # remove any style and whitespace
    # answer = answer.replace("*", "").replace("~", "").replace("_", "").replace(" ", "")
    # predict.append(answer)
    # print(image_file, result)
    

# %%

for image_file, result in zip(image_files, results):
    print(image_file.name)
    print(result)
    print()
# %%
