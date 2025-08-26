"""
Quick test to compare our implementation with the working built-in pipeline
"""

from diffusers import StableDiffusionXLPipeline
import torch

# Test the working pipeline first
print("=== Testing Built-in Pipeline ===")
pipe = StableDiffusionXLPipeline.from_single_file(
    "./model/zc.safetensors",
    torch_dtype=torch.float16
).to("cuda")

# Generate with built-in pipeline
built_in_image = pipe(
    "a portrait of a blond woman", 
    negative_prompt="blurry, low quality",
    num_inference_steps=25,
    guidance_scale=8.0,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

built_in_image.save("builtin_result.png")
print("Built-in pipeline saved as 'builtin_result.png'")

# Now test our modular implementation
print("\n=== Testing Our Implementation ===")
from src.modules import model_loader, text_encoder, noise_generator, sampler, vae_decoder

def generate_image(prompt: str, negative_prompt: str = None, **kwargs) -> 'PIL.Image':
    pipeline_components = model_loader.load_model()
    text_embeddings = text_encoder.encode_prompts(
        pipeline_components, prompt, negative_prompt
    )
    latents = noise_generator.create_noise(pipeline_components, **kwargs)
    denoised_latents = sampler.denoise(
        pipeline_components, latents, text_embeddings, **kwargs
    )
    image = vae_decoder.decode_to_image(pipeline_components, denoised_latents)
    return image

our_image = generate_image("a portrait of a blond woman", "blurry, low quality")
our_image.save("our_result.png")
print("Our implementation saved as 'our_result.png'")

print("\n🎯 Compare the two images to see the difference!")