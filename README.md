# Text to Image Generation

A Python-based project for generating images from text descriptions using state-of-the-art deep learning models.

## Features

- Generate high-quality images from text prompts
- Support for multiple models (Stable Diffusion, DALL-E alternatives)
- Batch processing capabilities
- Easy-to-use CLI and Python API
- Customizable parameters (guidance scale, num inference steps, etc.)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nabinbikramsah/nabin.git
cd nabin
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python -m text_to_image generate \
  --prompt "A beautiful sunset over mountains" \
  --output image.png \
  --model "stabilityai/stable-diffusion-2"
```

### Python API

```python
from text_to_image import TextToImageGenerator

# Initialize generator
generator = TextToImageGenerator(model_name="stabilityai/stable-diffusion-2")

# Generate image
image = generator.generate(
    prompt="A futuristic city with flying cars",
    num_inference_steps=50,
    guidance_scale=7.5
)

# Save image
image.save("output.png")
```

### Batch Processing

```python
from text_to_image import TextToImageGenerator

generator = TextToImageGenerator()
prompts = [
    "A cat wearing sunglasses",
    "A robot painting a masterpiece",
    "A forest covered in snow"
]

images = generator.generate_batch(prompts, output_dir="outputs/")
```

## Configuration

Create a `.env` file in the project root:

```env
MODEL_NAME=stabilityai/stable-diffusion-2
DEVICE=cuda
DTYPE=float16
MAX_BATCH_SIZE=4
OUTPUT_FORMAT=png
```

## API Reference

### TextToImageGenerator

#### `__init__(model_name, device=None, dtype=torch.float32)`
Initialize the generator with a specific model.

#### `generate(prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512, seed=None)`
Generate a single image from a text prompt.

**Parameters:**
- `prompt` (str): Text description of the image
- `num_inference_steps` (int): Number of denoising steps (higher = better quality, slower)
- `guidance_scale` (float): Scale of text guidance (higher = closer to prompt)
- `height` (int): Output image height (must be multiple of 8)
- `width` (int): Output image width (must be multiple of 8)
- `seed` (int): Random seed for reproducibility

**Returns:** PIL Image object

#### `generate_batch(prompts, num_inference_steps=50, guidance_scale=7.5, output_dir=None)`
Generate multiple images in batch.

**Parameters:**
- `prompts` (list): List of text prompts
- `num_inference_steps` (int): Number of denoising steps
- `guidance_scale` (float): Scale of text guidance
- `output_dir` (str): Directory to save images

**Returns:** List of PIL Image objects

## Supported Models

- `stabilityai/stable-diffusion-2` - Stable Diffusion 2.0
- `runwayml/stable-diffusion-v1-5` - Stable Diffusion 1.5
- `CompVis/stable-diffusion-v1-4` - Stable Diffusion 1.4

## Examples

See the `examples/` directory for more usage examples including:
- `basic_generation.py` - Simple image generation
- `batch_processing.py` - Processing multiple prompts
- `advanced_settings.py` - Custom configuration examples

## Performance Tips

1. **GPU Usage**: Enable CUDA for significantly faster generation
2. **Memory**: Use `float16` precision to reduce memory usage
3. **Quality vs Speed**: Balance `num_inference_steps` (50-100 recommended)
4. **Batch Size**: Adjust `MAX_BATCH_SIZE` based on your GPU memory

## Requirements

- torch >= 2.0
- transformers >= 4.30
- diffusers >= 0.21
- Pillow >= 9.0
- requests >= 2.28
- python-dotenv >= 0.21

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Uses [Diffusers library](https://github.com/huggingface/diffusers)
- Models from [Stability AI](https://stability.ai/)

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.
