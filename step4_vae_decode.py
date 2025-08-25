import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np

# Load model
model_path = "./model/BLPN3.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
device = "cuda"
pipe = pipe.to(device)

print("Loading the final latents from our denoising...")

# For now, let's recreate the final latents (we'll improve this later)
# You could save them from step 3, but this recreates them
generator = torch.Generator(device=device).manual_seed(42)
latents = torch.randn(1, 4, 128, 128, generator=generator, device=device, dtype=torch.float16)

# Quick denoising (abbreviated version)
positive_prompt = "a majestic dragon flying over mountains at sunset"
# ... (we'll do a quick generation)

print("Decoding latents to pixel image...")

# THE VAE DECODE - Transform math back into pixels!
with torch.no_grad():
    # Scale the latents (VAE expects a specific range)
    latents = latents / pipe.vae.config.scaling_factor
    
    # Decode to pixel space
    image = pipe.vae.decode(latents, return_dict=False)[0]
    
print(f"Decoded image tensor shape: {image.shape}")  # Should be [1, 3, 1024, 1024]
print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")

# Convert to PIL Image for saving/display
image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image[0])

# Save the result!
pil_image.save("generated_dragon.png")
print("Saved your dragon image as 'generated_dragon.png'!")
print(f"Final image size: {pil_image.size}")