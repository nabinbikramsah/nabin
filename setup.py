"""Setup configuration for text-to-image package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="text-to-image-generator",
    version="1.0.0",
    author="Nabin Bikram Sah",
    author_email="your.email@example.com",
    description="Generate high-quality images from text descriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nabinbikramsah/nabin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "text-to-image=text_to_image.cli:cli",
        ],
    },
)
