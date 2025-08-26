"""
Model loading utilities for SDXL pipeline components.

Handles loading SDXL models from safetensors files and managing device placement.
"""

import torch
from diffusers import StableDiffusionXLPipeline
from typing import Dict, Any
from src import config


def load_model(model_path: str = None, device: str = None) -> Dict[str, Any]:
    """
    Load SDXL pipeline components from a safetensors file.
    
    Args:
        model_path: Path to .safetensors model file. Uses config default if None.
        device: Target device ('cuda', 'cpu'). Auto-detects if None.
    
    Returns:
        Dictionary containing all pipeline components (unet, vae, text_encoders, etc.)
    """
    # Use config defaults if not provided
    if model_path is None:
        model_path = config.DEFAULT_MODEL_PATH
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SDXL model from: {model_path}")
    print(f"Target device: {device}")
    
    # Load the pipeline from single file
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    print(f"UNet config: {pipe.unet.config}")
    print(f"UNet input channels: {pipe.unet.in_channels}")
    print(f"UNet sample size: {pipe.unet.sample_size}")
        
    # Move to target device
    pipe = pipe.to(device)
    
    print("Model loaded successfully!")
    
    return {
        'pipeline': pipe,
        'unet': pipe.unet,
        'vae': pipe.vae,
        'text_encoder': pipe.text_encoder,
        'text_encoder_2': pipe.text_encoder_2,
        'tokenizer': pipe.tokenizer,
        'tokenizer_2': pipe.tokenizer_2,
        'scheduler': pipe.scheduler,
        'device': device
    }