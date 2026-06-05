"""Text to Image Generator using Hugging Face Diffusers"""

import os
from pathlib import Path
from typing import List, Optional, Union
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
from .config import Config


class TextToImageGenerator:
    """Text to Image Generator using Stable Diffusion"""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        dtype: torch.dtype = None,
        enable_attention_slicing: bool = None,
    ):
        """
        Initialize the Text to Image Generator.

        Args:
            model_name (str): Model name from Hugging Face
            device (str): Device to use ('cuda' or 'cpu')
            dtype (torch.dtype): Data type for model weights
            enable_attention_slicing (bool): Enable attention slicing for memory efficiency
        """
        self.model_name = model_name or Config.get_model_name()
        self.device = device or Config.get_device()
        self.dtype = dtype or Config.get_dtype()
        self.enable_attention_slicing = (
            enable_attention_slicing 
            if enable_attention_slicing is not None 
            else Config.ENABLE_ATTENTION_SLICING
        )

        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Data type: {self.dtype}")

        # Load the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            safety_checker=None,
        )

        # Move pipeline to device
        self.pipe = self.pipe.to(self.device)

        # Optimize for inference
        if self.enable_attention_slicing:
            self.pipe.enable_attention_slicing()

        # Use DPM-Solver scheduler for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        height: int = None,
        width: int = None,
        seed: int = None,
    ) -> Image.Image:
        """
        Generate a single image from a text prompt.

        Args:
            prompt (str): Text description of the image
            negative_prompt (str): Text describing what NOT to include
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Scale of text guidance
            height (int): Output image height (must be multiple of 8)
            width (int): Output image width (must be multiple of 8)
            seed (int): Random seed for reproducibility

        Returns:
            PIL.Image: Generated image
        """
        num_inference_steps = num_inference_steps or Config.DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = guidance_scale or Config.DEFAULT_GUIDANCE_SCALE
        height = height or Config.DEFAULT_HEIGHT
        width = width or Config.DEFAULT_WIDTH

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(
                torch.seed()
            )

        print(f"\nGenerating image for prompt: '{prompt}'")
        print(f"Parameters: steps={num_inference_steps}, guidance={guidance_scale}")

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

        return result.images[0]

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        height: int = None,
        width: int = None,
        output_dir: str = None,
    ) -> List[Image.Image]:
        """
        Generate multiple images in batch.

        Args:
            prompts (List[str]): List of text prompts
            negative_prompts (List[str]): List of negative prompts
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Scale of text guidance
            height (int): Output image height
            width (int): Output image width
            output_dir (str): Directory to save images

        Returns:
            List[PIL.Image]: List of generated images
        """
        output_dir = output_dir or Config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)

        images = []
        print(f"\nGenerating {len(prompts)} images in batch...")

        for idx, (prompt, negative_prompt) in enumerate(
            tqdm(zip(prompts, negative_prompts), total=len(prompts))
        ):
            image = self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )

            images.append(image)

            # Save image if output_dir is specified
            if output_dir:
                filename = f"image_{idx:04d}.{Config.OUTPUT_FORMAT}"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                print(f"Saved: {filepath}")

        return images

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        **kwargs,
    ) -> str:
        """
        Generate and save a single image.

        Args:
            prompt (str): Text description
            output_path (str): Path to save the image
            **kwargs: Additional arguments for generate()

        Returns:
            str: Path to saved image
        """
        image = self.generate(prompt, **kwargs)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        image.save(output_path)
        print(f"Image saved to: {output_path}")

        return output_path

    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
