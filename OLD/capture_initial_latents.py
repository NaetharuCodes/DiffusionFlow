import torch
from diffusers import StableDiffusionXLPipeline

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Store the original method
original_add_noise = pipe.scheduler.add_noise

# Create our hook
def capture_add_noise(original_samples, noise, timesteps):
    print("=== BUILT-IN PIPELINE INITIAL NOISE ===")
    print(f"Noise shape: {noise.shape}")
    print(f"Noise stats - mean: {noise.mean():.4f}, std: {noise.std():.4f}")
    print(f"Noise range: [{noise.min():.4f}, {noise.max():.4f}]")
    
    # Call the original method
    result = original_add_noise(original_samples, noise, timesteps)
    
    print("=== AFTER add_noise ===")
    print(f"Result stats - mean: {result.mean():.4f}, std: {result.std():.4f}")
    print(f"Result range: [{result.min():.4f}, {result.max():.4f}]")
    
    return result

# Replace with our hook
pipe.scheduler.add_noise = capture_add_noise

# Now run the pipeline
image = pipe(
    prompt="cinematic portrait of a man, professional lighting, high quality, detailed",
    negative_prompt="blurry, low quality, distorted, amateur",
    num_inference_steps=25,
    guidance_scale=8.5,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("builtin_debug.png")