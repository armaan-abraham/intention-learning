# Image handling functions


class ImageHandler:
    def resize_image(self, img, target_width):
        """Resize the image while maintaining aspect ratio to the target width.

        Args:
            img (PIL.Image.Image): The original image.
            target_width (int): The desired width of the resized image.

        Returns:
            PIL.Image.Image: The resized image.
        """
        pass

    def render_state(self, state):
        """Render the state into an image.

        Args:
            state (np.ndarray): The state to render.

        Returns:
            PIL.Image.Image: The rendered image.
        """
        pass

    def create_labeled_pair(self, img1, img2, target_width=150):
        """Create a side-by-side image of two images with labels 'a' and 'b'.

        Args:
            img1 (PIL.Image.Image): The first image.
            img2 (PIL.Image.Image): The second image.
            target_width (int, optional): The width to which both images will be resized. Defaults to 150.

        Returns:
            PIL.Image.Image: The combined image with labels.
        """
        pass

    def save_image(self, img, path):
        """Save the image to the specified path.

        Args:
            img (PIL.Image.Image): The image to save.
            path (str or pathlib.Path): The path where to save the image.
        """
        pass