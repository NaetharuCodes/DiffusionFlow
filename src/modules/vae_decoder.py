"""
VAE decoding utilities for SDXL pipeline.

Converts denoised latents back to pixel images using the Variational Autoencoder.
"""

import torch
from PIL import Image
import numpy as np


def decode_to_image(pipeline_components, latents):
    """
    Decode latent tensors to PIL Image using VAE decoder.
    
    Args:
        pipeline_components: Loaded model components from model_loader
        latents: Denoised latents from sampler
    
    Returns:
        PIL.Image: Final generated image
    """
    vae = pipeline_components['vae']
    device = pipeline_components['device']
    
    print("Decoding latents to image...")
    print(f"Pre-VAE latents shape: {latents.shape}")
    print(f"Pre-VAE latents range: [{latents.min():.4f}, {latents.max():.4f}]")
    
    with torch.no_grad():
        # Scale the latents for VAE (this is critical!)
        scaled_latents = latents / vae.config.scaling_factor
        print(f"VAE scaling factor: {vae.config.scaling_factor}")
        print(f"Scaled latents range: [{scaled_latents.min():.4f}, {scaled_latents.max():.4f}]")
        
        # Decode to pixel space
        image = vae.decode(scaled_latents, return_dict=False)[0]
    
    print(f"Decoded image tensor shape: {image.shape}")  # Should be [1, 3, 1024, 1024]
    print(f"Raw image tensor range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Convert from tensor to PIL Image
    # VAE outputs values roughly in [-1, 1] range, normalize to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # Convert to numpy and rearrange dimensions
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    
    # Convert to uint8 pixel values [0, 255]
    image = (image * 255).round().astype("uint8")
    
    # Create PIL Image from the first (and only) image in batch
    pil_image = Image.fromarray(image[0])
    
    print(f"Final PIL image size: {pil_image.size}")
    
    return pil_image