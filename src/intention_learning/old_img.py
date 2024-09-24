from PIL import Image, ImageDraw, ImageFont

# Function to resize image while maintaining aspect ratio
def resize_image(img, target_width):
    aspect_ratio = img.height / img.width
    target_height = int(target_width * aspect_ratio)
    return img.resize((target_width, target_height), Image.LANCZOS)

# Function to create a side-by-side image with labels
def create_labeled_pair(img1: Image.Image, img2: Image.Image, target_width=150) -> Image.Image:
    
    # Resize images
    img1 = resize_image(img1, target_width)
    img2 = resize_image(img2, target_width)
    
    # Define padding and margin
    top_padding = 40
    margin = 20

    # Calculate dimensions
    total_width = img1.width + img2.width + margin
    max_height = max(img1.height, img2.height)
    combined_height = max_height + top_padding

    # Create a new image with space for both images, labels, and margin
    combined_img = Image.new('RGB', (total_width, combined_height), color='black')
    
    # Paste the images
    combined_img.paste(img1, (0, top_padding))
    combined_img.paste(img2, (img1.width + margin, top_padding))
    
    # Add labels
    font = ImageFont.load_default(size=35)
    draw = ImageDraw.Draw(combined_img)
    draw.text((img1.width // 2, 5), "a", fill="red", font=font, anchor="mt")
    draw.text((img1.width + margin + img2.width // 2, 5), "b", fill="red", font=font, anchor="mt")
    
    # Add red line down the middle
    middle_x = img1.width + margin // 2
    draw.line([(middle_x - 0.2, 0), (middle_x - 0.2, combined_height)], fill="red", width=2)
    
    # Save the combined image
    return combined_img

import os
import itertools
from PIL import Image

def create_labeled_pairs_in_directory(directory_path, output_subdir='labeled_pairs'):
    """
    Gathers all PNG files in the specified directory, creates labeled pairs,
    and saves them in a new subdirectory.

    :param directory_path: Path to the directory containing PNG images.
    :param output_subdir: Name of the subdirectory to save labeled pairs.
    """
    # Gather all PNG files in the directory
    png_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.png')]

    if not png_files:
        print("No PNG files found in the specified directory.")
        return

    # Create output directory
    output_dir = os.path.join(directory_path, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate all unique pairs
    for img1_name, img2_name in itertools.combinations(png_files, 2):
        img1_path = os.path.join(directory_path, img1_name)
        img2_path = os.path.join(directory_path, img2_name)

        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
        except IOError as e:
            print(f"Error opening images {img1_name} or {img2_name}: {e}")
            continue

        # Create labeled pair
        try:
            labeled_pair = create_labeled_pair(img1, img2)
        except Exception as e:
            print(f"Error creating labeled pair for {img1_name} and {img2_name}: {e}")
            continue

        # Define the filename for the labeled pair
        pair_filename = f"{os.path.splitext(img1_name)[0]}_{os.path.splitext(img2_name)[0]}.png"
        pair_path = os.path.join(output_dir, pair_filename)

        # Save the labeled pair
        try:
            labeled_pair.save(pair_path)
            print(f"Saved labeled pair: {pair_path}")
        except IOError as e:
            print(f"Error saving labeled pair {pair_filename}: {e}")

    print(f"All labeled pairs have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create labeled image pairs from PNG files in a directory.")
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing PNG images."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="labeled_pairs",
        help="Name of the output subdirectory to save labeled pairs (default: 'labeled_pairs')."
    )

    args = parser.parse_args()
    create_labeled_pairs_in_directory(args.directory, args.output)