"""
Basic Example: Generate a single image from a text prompt
"""

from text_to_image import TextToImageGenerator

# Initialize the generator
generator = TextToImageGenerator()

# Generate a single image
prompt = "A beautiful sunset over mountains with golden light reflecting on water"
image = generator.generate(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
)

# Save the image
image.save("output.png")
print("Image saved to output.png")
