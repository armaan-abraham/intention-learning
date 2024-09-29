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
from typing import List
import torch
import PIL
import io
import os
from intention_learning.img import ImageHandler
from intention_learning.data import DataHandler
import base64


class PromptAndParser:
    prompt = (
        """You are a judge for the classic pendulum control problem. The
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
    """.replace("\n", " ")
        .replace("  ", " ")
        .replace("\t", "")
        .strip()
    )

    def clean_response(self, response: str) -> str:
        return response.replace(" ", "").replace("\n", "")

    def parse_response(self, response: str) -> int | None:
        cleaned = self.clean_response(response)
        if cleaned == "{a}":
            return 0
        elif cleaned == "{b}":
            return 1
        elif cleaned == "{none}":
            return None
        else:
            raise ValueError(f"Invalid response: {response}")


class Judge:
    """A judge model that evaluates pairs of states."""

    def __init__(
        self,
        image_handler: ImageHandler,
        data_handler: DataHandler,
        prompt_and_parser: PromptAndParser = PromptAndParser(),
        auth_token: str = None,
        max_batch_size: int = 40,
    ):
        """Initialize the judge model with the specified model path.

        Args:
            model_path (str or pathlib.Path): The path to the model directory.
            auth_token (str): The authentication token for accessing protected resources.
        """

        self.auth_token = (
            os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if auth_token is None
            else auth_token
        )

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
        self.tokenizer = MistralTokenizer.from_file(
            f"{mistral_models_path}/tekken.json"
        )
        self.max_batch_size = max_batch_size
        self.transformer = Transformer.from_folder(
            mistral_models_path, max_batch_size=max_batch_size
        ).eval()
        self.image_handler = image_handler
        self.prompt_and_parser = prompt_and_parser
        self.data_handler = data_handler

    def sample_and_judge(
        self,
        n_pairs: int = 1e2,
        batch_size: int = 20,
        save=True,
        discard_ties: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample states from the data handler and judge them."""
        states = self.data_handler.sample_past_states(int(n_pairs * 2))
        states1 = states[::2]
        states2 = states[1::2]
        judgments = []
        # loop over batches of states
        for i in range(0, len(states1), batch_size):
            judgments.extend(
                self.judge(states1[i : i + batch_size], states2[i : i + batch_size])
            )
        if discard_ties:
            valid_idx = [j is not None for j in judgments]
            judgments = [judgments[i] for i in valid_idx]
            states1 = states1[valid_idx]
            states2 = states2[valid_idx]
        judgments = torch.tensor(judgments, dtype=torch.int8)[:, None]
        if save:
            self.data_handler.save_judgments(states1, states2, judgments)
        return states1, states2, judgments

    def judge(self, states1: torch.Tensor, states2: torch.Tensor) -> list[int | None]:
        """DOES NOT ALLOW TIES. 1 FOR STATE 2, 0 FOR STATE 1."""
        prompt_imgs = [
            self.image_handler.overlay_states_on_img(s1, s2)
            for s1, s2 in zip(states1.cpu().numpy(), states2.cpu().numpy())
        ]
        requests = [
            self.get_completion_request(self.prompt_and_parser.prompt, prompt_img)
            for prompt_img in prompt_imgs
        ]
        responses = self.llm_respond(requests, max_tokens=8)
        judgments = [
            self.prompt_and_parser.parse_response(response) for response in responses
        ]
        return judgments, prompt_imgs

    def get_completion_request(
        self, prompt_txt: str, prompt_img: PIL.Image.Image
    ) -> ChatCompletionRequest:
        """Build the prompt to send to the LLM.

        Args:
            prompt_text (str): The textual prompt describing the intention.

        Returns:
            str: The complete prompt including any necessary formatting.
        """
        return ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        ImageURLChunk(
                            image_url=self.encode_image_for_prompt(prompt_img)
                        ),
                        TextChunk(text=prompt_txt),
                    ]
                )
            ]
        )

    @torch.inference_mode()
    def llm_respond(
        self,
        requests: List[ChatCompletionRequest],
        max_tokens: int,
        batch_size: int = 20,
    ) -> List[str]:
        assert (
            batch_size <= self.max_batch_size
        ), f"Batch size {batch_size} is larger than the maximum batch size {self.max_batch_size}"
        encoded_requests = [
            self.tokenizer.encode_chat_completion(request) for request in requests
        ]
        tokens = [request.tokens for request in encoded_requests]
        images = [request.images for request in encoded_requests]

        responses = []
        for i in range(0, len(tokens), batch_size):
            responses_batch, _ = generate(
                tokens[i : i + batch_size],
                self.transformer,
                images=images[i : i + batch_size],
                max_tokens=max_tokens,
                temperature=0.0,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )

            for i, response in enumerate(responses_batch):
                result = self.tokenizer.decode(response)
                responses.append(result)

        return responses

    def encode_image_for_prompt(self, image: PIL.Image.Image) -> str:
        """Encode the image to a base64 string for LLM input."""
        buffered = io.BytesIO()
        image.save(buffered, format="png")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

def validate_judgments(
    states1: torch.Tensor, states2: torch.Tensor, judgments: torch.Tensor
) -> bool:
    terminal_1 = -torch.arctan2(states1[:, 1], states1[:, 0]) ** 2
    terminal_2 = -torch.arctan2(states2[:, 1], states2[:, 0]) ** 2
    return judgments[:, 0] == (terminal_2 > terminal_1).to(torch.int8)
