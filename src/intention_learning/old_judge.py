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
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
tokenizer.instruct_tokenizer.tokenizer.special_token_policy = SpecialTokenPolicy.IGNORE
model = Transformer.from_folder(mistral_models_path, max_batch_size=10)

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


IMAGE_DIR = Path(__file__).parent.parent.parent / "images" / "overlaid_images_with_goal"
image_files = sorted(list(IMAGE_DIR.glob("*.png")), key=lambda x: int(x.stem.split("_")[-1]))
prompt = """You are a judge for the classic pendulum control problem. The
objective is to balance the pendulum upright. You are shown two pendulums. Both
pendulums pivot around the same point in the center of the image. The red
pendulum is pendulum **a** and the blue pendulum is pendulum **b**. Is any
pendulum clearly closer to the goal than the other? No need to think about
strategies that may be involved in getting the pendulum upright. Just respond
with the pendulum that best exemplifies an upright pendulum. Respond with one of
three answers, and surround your answer with curly brackets: {a}, {b}, or
{none}. Respond with {none} if neither pendulum really exemplifies the goal or
if both pendulums exemplify the goal to roughly the same degree. Respond only
with your answer surrounded by curly brackets and nothing else.
""".replace("\n", " ").replace("  ", " ")

truth = []
predict = []
results = []

BATCH_SIZE = 10  # Define the batch size

# Collect batches of image files
batches = [image_files[i:i + BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE)]

for batch in batches:
    completion_requests = []

    for image_file in batch:
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
        completion_requests.append(completion_request)

    # Encode all requests in the batch
    encoded_batches = [tokenizer.encode_chat_completion(req) for req in completion_requests]

    images = [encoded.images for encoded in encoded_batches]
    tokens = [encoded.tokens for encoded in encoded_batches]

    # Generate responses for the batch
    out_tokens, _ = generate(
        tokens,
        model,
        images=images,
        max_tokens=256,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )

    for i, out_token in enumerate(out_tokens):
        result = tokenizer.decode(out_token)
        results.append(result)
        print(batch[i].name)
        print(result)
        try:
            print(correct_answers[i])
        except:
            pass
        print("-" * 100)

# %%

correct_answers = results
# %%
