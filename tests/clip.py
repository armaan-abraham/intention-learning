from PIL import Image
from pathlib import Path

from transformers import CLIPProcessor, CLIPModel

IMAGE_DIR = Path(__file__).parent.parent / "images"
image_paths = list(IMAGE_DIR.glob("*.png"))

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

for image_path in image_paths:
    # Load the image from the local path
    image = Image.open(IMAGE_DIR / image_path)

    texts = [
        "an elf in square (4,1)",
        "an elf in square (4,4)",
        "an elf in square (3,1)",
        "an elf in square (1,4)",
        "an elf with a green hat in the top left corner",
        "an elf with a green hat in the top right corner",
        "an elf with a green hat in the bottom right corner",
        "an elf with a green hat in the bottom left corner",
    ]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities

    print(f"{image_path}:")
    for text, logit in zip(texts, logits_per_image[0]):
        print(f"  {text}: {logit:.4f}")
    print()
