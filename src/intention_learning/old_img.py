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