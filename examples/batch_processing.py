"""
Batch Processing Example: Generate multiple images from different prompts
"""

from text_to_image import TextToImageGenerator

# Initialize the generator
generator = TextToImageGenerator()

# List of prompts to generate images for
prompts = [
    "A robot painting a masterpiece in oil colors",
    "A futuristic city with flying cars at night",
    "A cozy cabin in a snowy forest",
    "An underwater city with glowing buildings",
    "A serene garden with cherry blossom trees",
]

# Generate batch of images
images = generator.generate_batch(
    prompts=prompts,
    num_inference_steps=50,
    guidance_scale=7.5,
    output_dir="batch_output",
)

print(f"Generated {len(images)} images successfully!")
