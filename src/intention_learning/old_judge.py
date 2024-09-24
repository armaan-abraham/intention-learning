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
model = Transformer.from_folder(mistral_models_path, max_batch_size=10)

import base64
import os

# %%

@torch.inference_mode()
def generate_(
    encoded_prompts: List[List[int]],
    model: Transformer,
    images: List[List[np.ndarray]] = [],
    *,
    max_tokens: int,
    temperature: float,
    chunk_size: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    images_torch: List[List[torch.Tensor]] = []
    if images:
        assert chunk_size is None
        images_torch = [
            [torch.tensor(im, device=model.device, dtype=model.dtype) for im in images_for_sample]
            for images_for_sample in images
        ]

    model = model.eval()
    B, V = len(encoded_prompts), model.args.vocab_size

    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # Bookkeeping
    logprobs: List[List[float]] = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    flattened_images: List[torch.Tensor] = sum(images_torch, [])

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            images=flattened_images,
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache,
        )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(
            0,
            torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1,
        )
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    assert last_token_prelogits is not None
    for _ in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        if eos_id is not None:
            is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tensors.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * B, cache=cache)
        assert last_token_prelogits.shape == (B, V)

    generated_tokens: List[List[int]]
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = []

    return generated_tokens, logprobs

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
image_files = list(IMAGE_DIR.glob("*.png"))
prompt = """You are an evaluation model for the task of ranking arrows based on
how close they point to a certain goal direction. In this case, the goal is to
make the arrow point upward. Which arrow, a (the red arrow) or b (the blue
arrow), is closer to this goal?  Respond only with this winning arrow surrounded
by asterisks, (*a* or *b*).
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
