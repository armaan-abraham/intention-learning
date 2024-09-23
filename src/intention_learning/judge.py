# Judge model for evaluating image pairs

class Judge:
    """A judge model that evaluates pairs of images to provide a reward signal."""

    def __init__(self, model_path, auth_token):
        """Initialize the judge model with the specified model path.

        Args:
            model_path (str or pathlib.Path): The path to the model directory.
            auth_token (str): The authentication token for accessing protected resources.
        """
        pass

    def encode_image(self, image):
        """Encode the image to a base64 string for LLM input.

        Args:
            image (PIL.Image.Image): The image to encode.

        Returns:
            str: The base64-encoded image string.
        """
        pass

    def wrap_image_content(self, image_b64):
        """Wrap the base64 image string in the appropriate data URI format.

        Args:
            image_b64 (str): The base64-encoded image string.

        Returns:
            str: The data URI string for the image.
        """
        pass

    def build_prompt(self, prompt_text):
        """Build the prompt to send to the LLM.

        Args:
            prompt_text (str): The textual prompt describing the intention.

        Returns:
            str: The complete prompt including any necessary formatting.
        """
        pass

    def judge_pair(self, img1, img2):
        """Judge the pair of images and return a reward.

        Args:
            img1 (PIL.Image.Image): The first image.
            img2 (PIL.Image.Image): The second image.
            prompt_text (str): The prompt describing the intention.

        Returns:
            int: The reward signal based on the judge's evaluation (1, 0, or -1).
        """
        prompt = "Which image shows the pendulum pointing more upward?"
        # STILL TODO
        pass

    def parse_response(self, response):
        """Parse the LLM response to extract the decision.

        Args:
            response (str): The response from the LLM.

        Returns:
            int: The parsed result (1 if 'b' is better, -1 if 'a' is better, 0 if tie or unclear).
        """
        pass