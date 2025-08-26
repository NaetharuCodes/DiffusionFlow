import torch
from diffusers import StableDiffusionXLPipeline

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

print(f"Scheduler type: {type(pipe.scheduler).__name__}")
print(f"init_noise_sigma: {pipe.scheduler.init_noise_sigma}")

# Create the same noise both ways
generator1 = torch.Generator("cuda").manual_seed(42)
generator2 = torch.Generator("cuda").manual_seed(42)

# Your manual way
manual_latents = torch.randn(1, 4, 128, 128, generator=generator1, device="cuda", dtype=torch.float16)
print(f"\nManual latents BEFORE scaling: [{manual_latents.min():.3f}, {manual_latents.max():.3f}]")

# What happens when we scale
scaled = manual_latents * pipe.scheduler.init_noise_sigma
print(f"Manual latents AFTER scaling by {pipe.scheduler.init_noise_sigma}: [{scaled.min():.3f}, {scaled.max():.3f}]")

# What the built-in pipeline actually creates
pipe.scheduler.set_timesteps(25)
init_timestep = pipe.scheduler.timesteps[0]
builtin_latents = torch.randn(1, 4, 128, 128, generator=generator2, device="cuda", dtype=torch.float16)

# Check if the scheduler has a scale_model_input or other method
if hasattr(pipe.scheduler, 'scale_model_input'):
    test_scaled = pipe.scheduler.scale_model_input(builtin_latents, init_timestep)
    print(f"\nscale_model_input at t={init_timestep}: [{test_scaled.min():.3f}, {test_scaled.max():.3f}]")

# Let's also check the scheduler config
print(f"\nScheduler config:")
for key in ['prediction_type', 'sigma_min', 'sigma_max', 'beta_start', 'beta_end']:
    if hasattr(pipe.scheduler.config, key):
        print(f"  {key}: {getattr(pipe.scheduler.config, key)}")