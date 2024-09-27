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
import enum
from intention_learning.data import DataHandler


class Judge:
    """A judge model that evaluates pairs of images."""

    def __init__(self, model_path, auth_token, image_handler: ImageHandler, max_batch_size: int = 40):
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
        # TODO: encode common part of input once, and store it. Make sure that
        # this is the same across calls of this function. 

        snapshot_download(
            repo_id="mistralai/Pixtral-12B-2409",
            allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
            local_dir=mistral_models_path,
            token=auth_token,
        )
        self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
        self.max_batch_size = max_batch_size
        self.transformer = Transformer.from_folder(mistral_models_path, max_batch_size=max_batch_size).eval()
        self.image_handler = image_handler


    def sample_and_judge(self, data_handler: DataHandler, n_pairs: int = 1e2, batch_size: int = 20, save=True) -> torch.Tensor:
        """Sample states from the data handler and judge them."""
        states = data_handler.sample_past_states(n_pairs * 2)
        states1 = states[::2]
        states2 = states[1::2]
        judgments = []
        # loop over batches of states
        for i in range(0, len(states1), batch_size):
            judgments.extend(self.judge(states1[i:i + batch_size], states2[i:i + batch_size]))
        judgments = torch.tensor(judgments)
        if save:
            data_handler.save_judgments(judgments)
        return judgments

    def judge(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor[int]:
        """ DOES NOT ALLOW TIES. 1 FOR STATE 2, 0 FOR STATE 1. """
        prompt_imgs = self.image_handler.create_overlaid_images(states1, states2)
        prompt_txt = """You are a judge for the classic pendulum control problem. The
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
        """.replace("\n", " ").replace("  ", " ").replace("\t", "").strip()
        requests = [self.get_completion_request(prompt_txt, prompt_img) for prompt_img in prompt_imgs]
        responses = self.llm_respond(requests, max_tokens=8)
        judgments = self.parse_responses(responses)
        return judgments


    def get_completion_request(self, prompt_txt: str, prompt_img: PIL.Image.Image):
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
        return completion_request

    @torch.inference_mode()
    def llm_respond(self, requests: List[ChatCompletionRequest], max_tokens: int, batch_size: int = 20):
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} is larger than the maximum batch size {self.max_batch_size}"
        encoded_requests = [self.tokenizer.encode_chat_completion(request) for request in requests]
        tokens = [request.tokens for request in encoded_requests]
        images = [request.images for request in encoded_requests]

        responses = []
        for i in range(0, len(tokens), batch_size):
            responses_batch, _ = generate(
                tokens[i:i + batch_size],
                self.transformer,
                images=images[i:i + batch_size],
                max_tokens=max_tokens,
                temperature=0.0,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

            for i, response in enumerate(responses_batch):
                result = self.tokenizer.decode(response)





        

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


class PromptAndParser:
    """A prompt and parser for the judge model."""
    def __init__(self, prompt: str, parser: Callable[[str], int]):
        self.prompt = prompt
        self.parser = parser
