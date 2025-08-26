"""
Stable Diffusion XL Pipeline Implementation

A modular implementation of SDXL for learning and experimentation.
"""

from src.modules import model_loader, text_encoder, noise_generator, sampler, vae_decoder
from src import config


def generate_image(prompt: str, negative_prompt: str = None, **kwargs) -> 'PIL.Image':
    """
    Generate an image from text prompts using SDXL pipeline.
    
    Args:
        prompt: Text description of desired image
        negative_prompt: Text description of what to avoid in image
        **kwargs: Override default generation parameters (steps, guidance_scale, etc.)
    
    Returns:
        PIL.Image: Generated image
    """
    # Load the SDXL pipeline components
    pipeline_components = model_loader.load_model()
    print("Model loading test successful!")
    
    # Convert prompts to embeddings
    text_embeddings = text_encoder.encode_prompts(
        pipeline_components, prompt, negative_prompt
    )
    print("Text encoding test successful!")
    
    # Generate initial noise
    latents = noise_generator.create_noise(pipeline_components, **kwargs)
    print("Noise generation test successful!")
    
    # Denoise through sampling loop
    denoised_latents = sampler.denoise(
        pipeline_components, latents, text_embeddings, **kwargs
    )
    print("Sampling test successful!")
    
    # Decode latents to final image
    image = vae_decoder.decode_to_image(pipeline_components, denoised_latents)
    print("VAE decoding test successful!")
    
    return image


if __name__ == "__main__":
    """Generate a full image!"""
    test_prompt = "a portrait of a blond woman"
    test_negative = "blurry, low quality"
    
    image = generate_image(test_prompt, test_negative)
    
    # Save the result
    image.save("my_modular_dragon.png")
    print("\n🎉 SUCCESS! Your modular SDXL dragon saved as 'my_modular_dragon.png'")