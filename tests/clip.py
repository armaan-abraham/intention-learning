from PIL import Image
from pathlib import Path

from transformers import CLIPProcessor, CLIPModel

IMAGE_DIR = Path(__file__).parent.parent / "images"
image_paths = ["3-0.png", "3-3.png", "2-0.png"]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

for image_path in image_paths:
    # Load the image from the local path
    image = Image.open(IMAGE_DIR / image_path)

    texts = ["an elf in square (3,0)", "an elf in square (3,3)", "an elf in square (2,0)", "an elf in square (0,3)", "elf is top left", "elf is top right", "elf is bottom left", "elf is bottom right"]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    print(f"{image_path}:")
    for text, logit in zip(texts, logits_per_image[0]):
        print(f"  {text}: {logit:.4f}")
    print()
