from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")


def run_mistral_7b():
    # Load the model and tokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", token=auth_token
    )

    # Function to generate text
    def generate_text(prompt, max_length=100):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_length=max_length, num_return_sequences=1
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Example usage
    prompt = "Explain the concept of machine learning in simple terms:"
    response = generate_text(prompt)
    print(f"Prompt: {prompt}\n")
    print(f"Response: {response}")


if __name__ == "__main__":
    run_mistral_7b()
