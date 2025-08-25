import torch
from diffusers import StableDiffusionXLPipeline

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

print("Testing Zavy Chroma with built-in pipeline...")

# Test with a prompt that should work great for Zavy Chroma
image = pipe(
    prompt="cinematic portrait of a man, professional lighting, high quality, detailed",
    negative_prompt="blurry, low quality, distorted, amateur",
    num_inference_steps=25,
    guidance_scale=8.0,  # Slightly higher for better prompt adherence
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("zavy_chroma_test.png")
print("Saved zavy_chroma_test.png")

print(f"Image size: {image.size}")