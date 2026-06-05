"""Command Line Interface for Text to Image Generation"""

import click
from .generator import TextToImageGenerator
from .config import Config


@click.group()
def cli():
    """Text to Image Generation CLI"""
    pass


@cli.command()
@click.option(
    "--prompt",
    required=True,
    help="Text description of the image to generate",
)
@click.option(
    "--output",
    "-o",
    default="output.png",
    help="Output file path",
)
@click.option(
    "--model",
    default=Config.MODEL_NAME,
    help="Model name from Hugging Face",
)
@click.option(
    "--steps",
    default=50,
    type=int,
    help="Number of inference steps",
)
@click.option(
    "--guidance",
    default=7.5,
    type=float,
    help="Guidance scale",
)
@click.option(
    "--height",
    default=512,
    type=int,
    help="Image height (multiple of 8)",
)
@click.option(
    "--width",
    default=512,
    type=int,
    help="Image width (multiple of 8)",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed",
)
def generate(prompt, output, model, steps, guidance, height, width, seed):
    """Generate a single image from text prompt"""
    generator = TextToImageGenerator(model_name=model)

    generator.generate_and_save(
        prompt=prompt,
        output_path=output,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        seed=seed,
    )

    click.echo(f"✓ Image saved to {output}")


@cli.command()
@click.option(
    "--prompts",
    "-p",
    multiple=True,
    required=True,
    help="Text prompts (can be used multiple times)",
)
@click.option(
    "--output-dir",
    "-o",
    default="outputs",
    help="Output directory",
)
@click.option(
    "--model",
    default=Config.MODEL_NAME,
    help="Model name from Hugging Face",
)
@click.option(
    "--steps",
    default=50,
    type=int,
    help="Number of inference steps",
)
@click.option(
    "--guidance",
    default=7.5,
    type=float,
    help="Guidance scale",
)
def batch(prompts, output_dir, model, steps, guidance):
    """Generate multiple images from text prompts"""
    generator = TextToImageGenerator(model_name=model)

    generator.generate_batch(
        prompts=list(prompts),
        num_inference_steps=steps,
        guidance_scale=guidance,
        output_dir=output_dir,
    )

    click.echo(f"✓ Generated {len(prompts)} images in {output_dir}")


@cli.command()
def config():
    """Display current configuration"""
    Config.print_config()


if __name__ == "__main__":
    cli()
