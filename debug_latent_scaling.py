import torch
from diffusers import StableDiffusionXLPipeline

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Check what the built-in pipeline does for latent initialization
print("Built-in pipeline latent scaling:")
print(f"VAE scaling factor: {pipe.vae.config.scaling_factor}")
print(f"Scheduler init noise sigma: {pipe.scheduler.init_noise_sigma}")

# Test with a simple noise tensor
test_noise = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)
print(f"Raw noise range: [{test_noise.min():.3f}, {test_noise.max():.3f}]")

# How the built-in pipeline would scale it
scaled_noise = test_noise * pipe.scheduler.init_noise_sigma
print(f"Scheduler scaled: [{scaled_noise.min():.3f}, {scaled_noise.max():.3f}]")

# How VAE expects it for decoding
vae_input = test_noise / pipe.vae.config.scaling_factor
print(f"VAE input scaling: [{vae_input.min():.3f}, {vae_input.max():.3f}]")