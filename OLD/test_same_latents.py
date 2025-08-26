import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

model_path = "./model/zc.safetensors"

# Load the model
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# CREATE OUR OWN STARTING NOISE
print("Creating our own starting noise...")
generator = torch.Generator("cuda").manual_seed(42)
my_latents = torch.randn(1, 4, 128, 128, generator=generator, device="cuda", dtype=torch.float16)

print(f"Our noise stats - mean: {my_latents.mean():.4f}, std: {my_latents.std():.4f}")
print(f"Our noise range: [{my_latents.min():.4f}, {my_latents.max():.4f}]")

# Save these latents so we can use them in our manual pipeline too
torch.save(my_latents, "shared_latents.pt")
print("Saved latents to shared_latents.pt")

# Now use these EXACT latents in the built-in pipeline
print("\nRunning built-in pipeline with our latents...")
image = pipe(
    prompt="cinematic portrait of a man, professional lighting, high quality, detailed",
    negative_prompt="blurry, low quality, distorted, amateur",
    latents=my_latents.clone(),  # <-- THIS IS THE KEY PART! We're giving it our noise
    num_inference_steps=25,
    guidance_scale=8.5,
).images[0]

image.save("builtin_with_our_latents.png")
print("Saved: builtin_with_our_latents.png")