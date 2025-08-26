"""
Noise generation utilities for SDXL latent space.

Creates initial random noise tensors for the diffusion process.
"""

import torch
from src import config


def create_noise(pipeline_components, height=None, width=None, seed=None, **kwargs):
    """
    Generate initial latent noise for SDXL diffusion process.
    
    Args:
        pipeline_components: Loaded model components from model_loader
        height: Image height in pixels (default: config default)
        width: Image width in pixels (default: config default) 
        seed: Random seed for reproducible generation (default: random)
        **kwargs: Additional parameters (ignored for noise generation)
    
    Returns:
        torch.Tensor: Random noise tensor of shape [1, 4, latent_height, latent_width]
    """
    # Use config defaults if not provided
    if height is None:
        height = config.DEFAULT_HEIGHT
    if width is None:
        width = config.DEFAULT_WIDTH
    
    device = pipeline_components['device']
    
    # SDXL latents are 8x smaller than pixel dimensions
    latent_height = height // 8
    latent_width = width // 8
    
    print(f"Creating noise for {height}x{width} image ({latent_height}x{latent_width} latents)")
    
    # Set up generator for reproducible results
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        # Use a default seed so we get consistent results during development
        generator.manual_seed(42)
        print(f"Using default seed: 42")
    
    # Create random latent noise
    # Shape: [batch_size, channels, height, width]
    # SDXL latents have 4 channels
    latents = torch.randn(
        1, 4, latent_height, latent_width,
        generator=generator,
        device=device,
        dtype=torch.float16
    )
    
    print(f"Initial noise range: [{latents.min():.3f}, {latents.max():.3f}]")

    scheduler = pipeline_components['scheduler']
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(25) 
    print(f"Scaled noise range: [{latents.min():.3f}, {latents.max():.3f}]")

    print(f"Scheduler type: {type(scheduler)}")

    print(f"Scheduler init_noise_sigma: {scheduler.init_noise_sigma}")
    
    return latents