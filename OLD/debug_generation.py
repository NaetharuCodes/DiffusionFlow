import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

model_path = "./model/BLPN3.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Try different parameters
prompts = [
    "a red dragon",  # Simpler prompt
    "dragon flying over mountains at sunset",  # Original
    "majestic dragon, highly detailed, fantasy art"  # Different style
]

for i, prompt in enumerate(prompts):
    print(f"\nGenerating with prompt: '{prompt}'")
    
    # Use the built-in pipeline for comparison
    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality",
        num_inference_steps=30,  # More steps
        guidance_scale=12.0,     # Higher guidance
        generator=torch.Generator("cuda").manual_seed(42 + i)
    ).images[0]
    
    image.save(f"test_{i+1}_{prompt[:10].replace(' ', '_')}.png")
    print(f"Saved test_{i+1}_*.png")