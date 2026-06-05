"""
Advanced Example: Using custom settings and negative prompts
"""

from text_to_image import TextToImageGenerator
import torch

# Initialize with specific settings
generator = TextToImageGenerator(
    model_name="runwayml/stable-diffusion-v1-5",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,
)

# Positive prompt - what you want to see
prompt = "A professional photograph of a golden retriever in a field of sunflowers"

# Negative prompt - what you want to avoid
negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

# Generate with fine-tuned parameters
image = generator.generate_and_save(
    prompt=prompt,
    output_path="advanced_output.png",
    negative_prompt=negative_prompt,
    num_inference_steps=100,  # More steps = better quality, slower generation
    guidance_scale=15.0,       # Higher guidance = stronger text adherence
    height=768,
    width=768,
    seed=42,                   # Fixed seed for reproducibility
)

print("Advanced image generation completed!")

# Clear GPU memory after generation
generator.clear_memory()
