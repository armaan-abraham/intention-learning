# Image handling functions

from itertools import permutations
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont


class ImageHandler:
    def __init__(self, image_dim=450):
        self.image_dim = image_dim

    def render_state(
        self,
        state: np.ndarray,
        color: str = "black",
        opacity: int = 255,
        width: float = 0.1,
        length_sub: float = 0.01,
        corner_radius: int = 10,
    ) -> Image.Image:
        """Render the pendulum state as a rounded rectangle with a black circle at its base.

        Returns:
            PIL.Image.Image: The rendered image of the pendulum.
        """
        # Extract the angle θ from the state
        cos_theta, sin_theta = state[0], state[1]
        theta = np.arctan2(sin_theta, cos_theta) - np.pi / 2
        theta_degrees = np.degrees(
            -theta
        )  # Negative because PIL rotates counterclockwise

        pendulum_width = int(self.image_dim * width)

        # Define pendulum parameters
        origin = (self.image_dim // 2, self.image_dim // 2)  # Center of the image
        length = (
            self.image_dim // 2 - self.image_dim * length_sub
        )  # Length of the pendulum rod

        # Create a transparent image for the pendulum
        pendulum_image = Image.new(
            "RGBA", (self.image_dim, self.image_dim), (0, 0, 0, 0)
        )
        pendulum_draw = ImageDraw.Draw(pendulum_image)

        # Apply opacity to the color
        fill_color = ImageColor.getrgb(color) + (opacity,)

        # Coordinates for the rounded rectangle (drawn pointing rightwards from origin)
        upper_left = (origin[0] - pendulum_width // 2, origin[1] - pendulum_width // 2)
        lower_right = (origin[0] + length, origin[1] + pendulum_width // 2)

        # Draw the rounded rectangle
        pendulum_draw.rounded_rectangle(
            [upper_left, lower_right], radius=pendulum_width // 2, fill=fill_color
        )

        # Draw the black circle at the base (origin)
        circle_radius = pendulum_width // 2
        circle_bbox = [
            origin[0] - circle_radius,
            origin[1] - circle_radius,
            origin[0] + circle_radius,
            origin[1] + circle_radius,
        ]
        pendulum_draw.ellipse(circle_bbox, fill=(0, 0, 0, opacity))

        # Rotate the pendulum image around the origin
        rotated_pendulum = pendulum_image.rotate(
            theta_degrees, center=origin, resample=Image.BICUBIC, expand=False
        )

        # Create the base image
        image = Image.new("RGBA", (self.image_dim, self.image_dim), (255, 255, 255, 0))

        # Composite the rotated pendulum onto the base image
        image = Image.alpha_composite(image, rotated_pendulum)

        return image

    def overlay_states_on_img(self, s1: np.ndarray, s2: np.ndarray) -> Image.Image:
        """Create an image overlaying two or three pendulum states with different colors, labels, and transparency.

        Args:
            s1 (np.ndarray): The first state.
            s2 (np.ndarray): The second state.
            goal_state (np.ndarray, optional): The goal state to overlay. Defaults to None.

        Returns:
            PIL.Image.Image: The combined image with overlaid pendulums and stacked labels.
        """
        # Colors, labels, and opacity settings
        color1 = "green"
        color2 = "blue"
        opacity = 245  # Adjust for desired transparency (0-255)

        # Create a base image
        combined_img = Image.new(
            "RGBA", (self.image_dim, self.image_dim), (255, 255, 255, 255)
        )

        # Render the states with specified colors and opacity
        img1 = self.render_state(
            s1,
            color=color1,
            opacity=opacity,
        )
        img2 = self.render_state(
            s2,
            color=color2,
            opacity=opacity,
        )

        # Overlay the two images onto the base image
        combined_img = Image.alpha_composite(combined_img, img1)
        combined_img = Image.alpha_composite(combined_img, img2)

        return combined_img

    def visualize_terminal_model(self, terminal_model, device, with_peak: bool = False):
        """Visualize the terminal model's output with respect to the angle around the pivot point.

        Args:
            terminal_model (TerminalModel): The trained terminal model.
            device (torch.device): The device to perform computations on.
            title (str): The title for the visualization. Defaults to "Terminal Model Visualization".

        Returns:
            PIL.Image.Image: The visualization image.
        """
        import numpy as np
        from PIL import Image, ImageDraw

        # Set font size
        font_size = 18
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            font = ImageFont.load_default()

        # Estimate title height (font size plus some padding)
        title_height = int(font_size * 1.5)
        title_padding = 6  # Space between title and circle
        total_padding = title_height + title_padding

        # Create a new image with a white background, adjusted for the title height
        img_height = self.image_dim + total_padding
        img = Image.new("RGB", (self.image_dim, img_height), "white")

        draw = ImageDraw.Draw(img)

        # Adjust center for the circle to be below the title
        center = (self.image_dim // 2, (self.image_dim // 2) + total_padding)
        radius = self.image_dim // 2 - 10  # Padding of 10 pixels

        # Generate angles from 0 to 2π
        num_points = 1500
        angles = torch.linspace(-np.pi, np.pi, steps=num_points, device=device)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        velocities = torch.zeros_like(angles)
        states = torch.stack([cos_angles, sin_angles, velocities], dim=1).to(device)

        # Get terminal model predictions
        with torch.no_grad():
            values = terminal_model.predict(states).squeeze().cpu().numpy()

        # Normalize values to [0, 1] for color mapping
        min_value = np.min(values)
        max_value = np.max(values)
        normalized_values = (values - min_value) / (max_value - min_value + 1e-8)

        # Get the rocket color palette from seaborn
        rocket_palette = sns.color_palette("rocket", as_cmap=True)
        angles = angles.cpu().numpy()
        angles = angles - np.pi / 2

        # Draw radial lines
        for i, angle in enumerate(angles):
            value = normalized_values[i]
            color = tuple(
                int(c * 255) for c in rocket_palette(value)[:3]
            )  # Convert to RGB
            end_point = (
                center[0] + radius * -np.cos(angle),
                center[1] + radius * np.sin(angle),
            )
            draw.line([center, end_point], fill=color, width=2)

        if with_peak:
            weights = normalized_values
            assert np.all(weights >= 0)
            peak_angle = (angles * weights).sum() / weights.sum()

            peak_end_point = (
                center[0] + radius * np.cos(peak_angle),
                center[1] + radius * np.sin(peak_angle),
            )
            draw.line([center, peak_end_point], fill="green", width=4)

        # flip horizontally
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Add title at the top center, ensuring it doesn't overlap with the circle
        title_position = (self.image_dim // 2, title_padding // 2)
        title = "Terminal value (reward) vs angle"
        draw.text(title_position, title, fill="black", font=font, anchor="mt")

        return img


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent.parent.parent / "images"

    # Create an instance of ImageHandler
    image_handler = ImageHandler(image_dim=1500)

    # Define pendulum states to render
    angles = [
        0,
        np.pi / 2 + 0.1,
        -np.pi / 4,
        np.pi - 0.1,
        np.pi,
    ]
    states = [np.array([np.cos(angle), np.sin(angle), 0.0]) for angle in angles]

    # Define the goal state
    goal_state = np.array([1.0, 0.0, 0.0])  # Angle 0 (rightward)

    # Specify the directory to save overlaid images
    overlaid_images_dir = out_dir / "overlaid_images_with_goal"
    overlaid_images_dir.mkdir(parents=True, exist_ok=True)

    from itertools import permutations

    # Iterate over all unique permutations of states
    for idx, (s1, s2) in enumerate(permutations(states, 2), start=1):
        if np.all(s1 == s2):
            continue

        # Create the overlaid image including the goal state
        overlaid_image = image_handler.overlay_states_on_img(s1, s2)

        # Define the filename for the overlaid image
        image_filename = f"overlaid_with_goal_{idx}.png"
        image_path = overlaid_images_dir / image_filename

        # Save the overlaid image
        overlaid_image.save(image_path)

    print(f"Overlaid images with goal have been saved to {overlaid_images_dir}")
