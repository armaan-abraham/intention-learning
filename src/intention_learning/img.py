# Image handling functions

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from itertools import permutations


class ImageHandler:
    def __init__(self, image_dim=900, grayscale=True):
        self.image_dim = image_dim
        self.grayscale = grayscale

    def render_state(self, state: np.ndarray) -> Image.Image:
        """Render the pendulum state into an image.

        Args:
            state (np.ndarray): The state to render, with shape (3,).
            image_size (tuple, optional): Size of the output image. Defaults to (200, 200).

        Returns:
            PIL.Image.Image: The rendered image of the pendulum.
        """

        # Extract the angle θ from the state
        cos_theta, sin_theta = state[0], state[1]
        theta = np.arctan2(sin_theta, cos_theta)

        # Define pendulum parameters
        origin = (self.image_dim // 2, self.image_dim // 2)  # Center of the image
        length = self.image_dim // 2 - self.image_dim // 8  # Length of the pendulum rod

        # Calculate the pendulum bob position
        x_end = origin[0] + length * np.sin(theta)
        y_end = origin[1] + length * np.cos(theta)

        # Create a blank image and get a drawing context
        image = Image.new("RGB", (self.image_dim, self.image_dim), "white")
        draw = ImageDraw.Draw(image)

        pendulum_width = int(self.image_dim // 17)

        # Draw the pendulum rod
        draw.line([origin, (x_end, y_end)], fill="black", width=pendulum_width)

        # Add a circle at the origin
        # circle_radius = 5
        # draw.ellipse([origin[0] - circle_radius, origin[1] - circle_radius, origin[0] + circle_radius, origin[1] + circle_radius], fill="black")

        # Draw triangle pointer at the end of the pendulum
        arrow_length = int(pendulum_width * 2)  # Length of the arrow tip
        arrow_width = int(pendulum_width * 2)   # Width of the arrow base

        # Calculate the tip point of the triangle
        tip_x = x_end + arrow_length * np.sin(theta)
        tip_y = y_end + arrow_length * np.cos(theta)

        # Calculate the base points of the triangle
        theta_perp = theta + np.pi / 2  # Perpendicular angle

        base_left_x = x_end + (arrow_width / 2) * np.sin(theta_perp)
        base_left_y = y_end + (arrow_width / 2) * np.cos(theta_perp)

        base_right_x = x_end - (arrow_width / 2) * np.sin(theta_perp)
        base_right_y = y_end - (arrow_width / 2) * np.cos(theta_perp)

        # Draw the triangle pointer
        draw.polygon(
            [(tip_x, tip_y), (base_left_x, base_left_y), (base_right_x, base_right_y)],
            fill="black"
        )

        # flip the image
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # Convert the image to grayscale
        if self.grayscale:
            image = image.convert("L")

        return image

    def render_goal(self) -> Image.Image:
        """Render the goal state into an image.

        Args:
            goal_state (np.ndarray): The goal state to render, with shape (3,).

        Returns:
            PIL.Image.Image: The rendered image of the goal state.
        """
        return self.render_state(np.array([1.0, 0.0, 0.0]))
        

    def create_labeled_pair(self, s1: np.ndarray | Image.Image, s2: np.ndarray | Image.Image) -> Image.Image:
        """Create an image with the goal on the far left and two images labeled 'a' and 'b'.

        Args:
            s1 (np.ndarray | Image.Image): The first image.
            s2 (np.ndarray | Image.Image): The second image.

        Returns:
            PIL.Image.Image: The combined image with labels.
        """
        if isinstance(s1, np.ndarray):
            img1 = self.render_state(s1)
        else:
            img1 = s1

        if isinstance(s2, np.ndarray):
            img2 = self.render_state(s2)
        else:
            img2 = s2

        # Render the goal image
        goal_img = self.render_goal()

        # Define padding and margin
        top_padding = int(0.1 * self.image_dim)
        margin = int(0.05 * self.image_dim)

        # Calculate dimensions
        total_width = self.image_dim * 3 + margin * 2
        total_height = self.image_dim + top_padding

        # Create a new image with space for all images, labels, and margin
        combined_img = Image.new('RGB', (total_width, total_height), color='black')

        # Paste the images
        combined_img.paste(goal_img, (0, top_padding))
        combined_img.paste(img1, (self.image_dim + margin, top_padding))
        combined_img.paste(img2, (2 * self.image_dim + 2 * margin, top_padding))

        # Add labels
        font = ImageFont.load_default(size=int(self.image_dim / 15))
        draw = ImageDraw.Draw(combined_img)
        draw.text((self.image_dim // 2, 5), "goal", fill="white", font=font, anchor="mt")
        draw.text((self.image_dim + margin + self.image_dim // 2, 5), "a", fill="white", font=font, anchor="mt")
        draw.text((2 * self.image_dim + 2 * margin + self.image_dim // 2, 5), "b", fill="white", font=font, anchor="mt")

        # Add lines to separate the images
        # sep1_x = self.image_dim + margin // 2
        # sep2_x = 2 * self.image_dim + margin + margin // 2
        # line_width = 5
        # draw.line([(sep1_x, 0), (sep1_x, total_height)], fill="white", width=5)
        # draw.line([(sep2_x, 0), (sep2_x, total_height)], fill="white", width=5)

        if self.grayscale:
            combined_img = combined_img.convert("L")

        return combined_img

    def save_image(self, img, path):
        """Save the image to the specified path.

        Args:
            img (PIL.Image.Image): The image to save.
            path (str or pathlib.Path): The path where to save the image.
        """
        pass


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent.parent.parent / "images"
    

    # Create an instance of ImageHandler
    image_handler = ImageHandler(grayscale=True)

    # Define a few pendulum states to render
    angles = [0, np.pi/2, np.pi/2 + 0.05, np.pi/2 + 0.1]
    states = [
        np.array([np.cos(angle), np.sin(angle), 0.0]) for angle in angles
    ]

    # Specify the directory to save paired images
    paired_images_dir = out_dir / "paired_images"
    paired_images_dir.mkdir(parents=True, exist_ok=True)

    from itertools import permutations

    # Iterate over all unique permutations of states
    for idx, (s1, s2) in enumerate(permutations(states, 2), start=1):
        if np.all(s1 == s2):
            continue

        # Create the paired image
        paired_image = image_handler.create_labeled_pair(s1, s2)

        # Define the filename for the paired image
        image_filename = f"pair_{idx}.png"
        image_path = paired_images_dir / image_filename

        # Save the paired image directly
        paired_image.save(image_path)

    print(f"Paired images have been saved to {paired_images_dir}")

