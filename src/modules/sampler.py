"""
Sampling utilities for SDXL denoising process.

Handles the iterative denoising loop that transforms noise into images.
"""

import torch
from src import config


def denoise(pipeline_components, latents, text_embeddings, num_steps=None, 
           guidance_scale=None, **kwargs):
    """
    Denoise latents through iterative sampling process.
    
    Args:
        pipeline_components: Loaded model components from model_loader
        latents: Initial noise tensor from noise_generator  
        text_embeddings: Encoded prompts from text_encoder
        num_steps: Number of denoising steps (default: config default)
        guidance_scale: Classifier-free guidance strength (default: config default)
        **kwargs: Additional parameters
    
    Returns:
        torch.Tensor: Denoised latents ready for VAE decoding
    """
    # Use config defaults if not provided
    if num_steps is None:
        num_steps = config.DEFAULT_STEPS
    if guidance_scale is None:
        guidance_scale = config.DEFAULT_GUIDANCE_SCALE
    
    device = pipeline_components['device']
    scheduler = pipeline_components['scheduler']
    unet = pipeline_components['unet']
    
    # Extract embeddings from text_encoder output
    positive_embeds = text_embeddings['positive_embeds']
    negative_embeds = text_embeddings['negative_embeds']
    pooled_positive = text_embeddings['pooled_positive']
    pooled_negative = text_embeddings['pooled_negative']
    time_ids = text_embeddings['time_ids']
    
    print(f"Starting denoising: {num_steps} steps, guidance scale {guidance_scale}")
    
    # Set up scheduler timesteps
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    
    # The denoising loop
    for i, t in enumerate(timesteps):
        # Prepare latent model input for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Combine embeddings for classifier-free guidance (negative first, then positive)
        text_embeddings_combined = torch.cat([negative_embeds, positive_embeds])
        add_text_embeds = torch.cat([pooled_negative, pooled_positive])
        add_time_ids = torch.cat([time_ids, time_ids])
        
        # Predict the noise with UNet
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings_combined,
                added_cond_kwargs={
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                },
                return_dict=False,
            )[0]
        
        # Classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Take a denoising step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Print progress every step to see the full denoising process
        print(f"Step {i+1}/{num_steps}: latent range [{latents.min():.3f}, {latents.max():.3f}]")
    
    print(f"Denoising complete! Final latents: [{latents.min():.3f}, {latents.max():.3f}]")
    
    return latents