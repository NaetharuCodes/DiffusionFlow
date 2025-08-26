import torch
from diffusers import StableDiffusionXLPipeline

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Capture what goes into the VAE from built-in pipeline
original_vae_decode = pipe.vae.decode

def capture_vae_input(latents, return_dict=True):
    print("=== BUILT-IN VAE INPUT ===")
    print(f"Shape: {latents.shape}")
    print(f"Range: [{latents.min():.4f}, {latents.max():.4f}]")
    print(f"Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
    
    # Save for comparison
    torch.save(latents.cpu(), "builtin_vae_input.pt")
    
    return original_vae_decode(latents, return_dict=return_dict)

pipe.vae.decode = capture_vae_input

# Generate with built-in to see what it feeds the VAE
print("Built-in pipeline VAE input:")
image = pipe(
    prompt="cinematic portrait of a man, professional lighting, high quality, detailed",
    generator=torch.Generator("cuda").manual_seed(42),
    num_inference_steps=25
).images[0]

print("\n" + "="*50)

# Now let's see what your working soft version would feed the VAE
print("Your working version would feed VAE:")
generator = torch.Generator(device="cuda").manual_seed(42)
latents = torch.randn(1, 4, 128, 128, generator=generator, device="cuda", dtype=torch.float16)

# Simulate your working denoising (without init_noise_sigma scaling)
print("Manual denoising simulation...")
# After denoising, your working version had around [-3.7, 2.5]
simulated_final_latents = latents * 0.8  # Approximate your final range

print("=== YOUR WORKING VERSION VAE INPUT ===")
vae_input = simulated_final_latents / pipe.vae.config.scaling_factor
print(f"Shape: {vae_input.shape}")
print(f"Range: [{vae_input.min():.4f}, {vae_input.max():.4f}]")
print(f"Mean: {vae_input.mean():.4f}, Std: {vae_input.std():.4f}")
torch.save(vae_input.cpu(), "manual_vae_input.pt")