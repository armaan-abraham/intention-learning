# Judge model for evaluating image pairs
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    TextChunk,
    ImageURLChunk,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import os
from intention_learning.img import ImageHandler

VALID_JUDGMENTS = [-1, 0, 1]

class Judge:
    """A judge model that evaluates pairs of images."""

    def __init__(self, model_path, auth_token, image_handler: ImageHandler):
        """Initialize the judge model with the specified model path.

        Args:
            model_path (str or pathlib.Path): The path to the model directory.
            auth_token (str): The authentication token for accessing protected resources.
        """
        from huggingface_hub import snapshot_download
        from pathlib import Path
        import os

        self.auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        mistral_models_path = Path.home().joinpath("mistral_models", "Pixtral")
        mistral_models_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id="mistralai/Pixtral-12B-2409",
            allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
            local_dir=mistral_models_path,
            token=auth_token,
        )
        self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
        self.model = Transformer.from_folder(mistral_models_path).eval()
        self.image_handler = image_handler


    def sample_and_judge(self, data_handler: DataHandler, n_pairs: int = 1e2, batch_size: int = 20, save=True):
        """Sample states from the data handler and judge them."""
        states = data_handler.sample_past_states(n_pairs * 2)
        pairs = list(zip(states[::2], states[1::2]))
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        judgments = []
        for batch in batches:
            judgments.extend(self.judge(batch))
        judgments = torch.concat(judgments)
        if save:
            data_handler.save_judgments(judgments)
        else:
            return judgments

    def judge(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor[int]:
        prompt_imgs = self.image_handler.create_overlaid_images(states1, states2)
        prompt_txt = """You are an evaluation model for the task of ranking arrows based on
        how close they point to a certain goal direction. In this case, the goal is to
        make the arrow point upward. Which arrow, a (the red arrow) or b (the blue
        arrow), is closer to this goal?  Respond only with this winning arrow surrounded
        by asterisks, (*a* or *b*).
        """
        prompt_txt = prompt_txt.replace("\n", " ").replace("\t", "").replace("  ", " ").strip()
        encoded_requests = [self.get_encoded_completion_request(prompt_txt, prompt_img) for prompt_img in prompt_imgs]
        responses = self.llm_respond(encoded_requests, max_tokens=6)
        # TODO: encode common part of input once, and store it. Make sure that
        # this is the same across calls of this function. 
        judgments = self.parse_responses(responses)
        return judgments


    def get_encoded_completion_request(self, prompt_txt: str, prompt_img: PIL.Image.Image):
        """Build the prompt to send to the LLM.

        Args:
            prompt_text (str): The textual prompt describing the intention.

        Returns:
            str: The complete prompt including any necessary formatting.
        """
        prompt_img = self.encode_image_for_prompt(prompt_img)
        completion_request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        ImageURLChunk(
                            image_url=prompt_img
                        ),
                        TextChunk(text=prompt_txt),
                    ]
                )
            ]
        )
        encoded = self.tokenizer.encode_chat_completion(completion_request)
        return encoded

    @torch.inference_mode()
    def llm_respond(self, encoded_requests, max_tokens):
        encoded_txts = [request.tokens for request in encoded_requests]
        images = [request.images for request in encoded_requests]
        images_torch: List[List[torch.Tensor]] = []
        images_torch = [
            [torch.tensor(im, device=self.model.device, dtype=self.model.dtype) for im in images_for_sample]
            for images_for_sample in images
        ]
        B, V = len(encoded_txts), self.model.args.vocab_size
        seqlens = [len(x) for x in encoded_txts]
        # TODO: cache



        

    def encode_image_for_prompt(self, image: PIL.Image.Image) -> str:
        """Encode the image to a base64 string for LLM input."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"

    def parse_response(self, response):
        """Parse the LLM response to extract the decision.

        Args:
            response (str): The response from the LLM.

        Returns:
            int: The parsed result (1 if 'b' is better, -1 if 'a' is better, 0 if tie or unclear).
        """
        pass