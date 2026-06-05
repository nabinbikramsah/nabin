"""Configuration module for text to image generation"""

import os
from dotenv import load_dotenv
import torch

load_dotenv()


class Config:
    """Configuration class for text to image generation"""

    # Model configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-2")
    
    # Device configuration
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Data type configuration
    DTYPE = os.getenv("DTYPE", "float32")
    if DTYPE == "float16":
        TORCH_DTYPE = torch.float16
    else:
        TORCH_DTYPE = torch.float32
    
    # Generation parameters
    DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", 512))
    DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", 512))
    DEFAULT_NUM_INFERENCE_STEPS = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", 50))
    DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", 7.5))
    
    # Batch processing
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 4))
    
    # Output configuration
    OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "png").lower()
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
    
    # Performance
    ENABLE_ATTENTION_SLICING = os.getenv("ENABLE_ATTENTION_SLICING", "true").lower() == "true"
    ENABLE_MEMORY_EFFICIENT_ATTENTION = os.getenv("ENABLE_MEMORY_EFFICIENT_ATTENTION", "false").lower() == "true"
    
    @classmethod
    def get_device(cls):
        """Get device information"""
        return cls.DEVICE
    
    @classmethod
    def get_dtype(cls):
        """Get torch dtype"""
        return cls.TORCH_DTYPE
    
    @classmethod
    def get_model_name(cls):
        """Get model name"""
        return cls.MODEL_NAME
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("Configuration Settings:")
        print("=" * 50)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Data Type: {cls.DTYPE}")
        print(f"Default Height: {cls.DEFAULT_HEIGHT}")
        print(f"Default Width: {cls.DEFAULT_WIDTH}")
        print(f"Default Inference Steps: {cls.DEFAULT_NUM_INFERENCE_STEPS}")
        print(f"Default Guidance Scale: {cls.DEFAULT_GUIDANCE_SCALE}")
        print(f"Max Batch Size: {cls.MAX_BATCH_SIZE}")
        print(f"Output Format: {cls.OUTPUT_FORMAT}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print("=" * 50)
