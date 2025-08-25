import torch
from diffusers import StableDiffusionXLPipeline

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Monkey patch to capture latents before VAE decode
original_vae_decode = pipe.vae.decode

def capture_latents_decode(latents, return_dict=True):
    print(f"Built-in pre-VAE latents range: [{latents.min():.4f}, {latents.max():.4f}]")
    print(f"Built-in pre-VAE latents mean: {latents.mean():.4f}")
    print(f"Built-in pre-VAE latents std: {latents.std():.4f}")
    torch.save(latents.cpu(), "builtin_final_latents.pt")
    return original_vae_decode(latents, return_dict=return_dict)

pipe.vae.decode = capture_latents_decode

# Generate with built-in pipeline
image = pipe(
    prompt="cinematic portrait of a man, professional lighting, high quality, detailed",
    generator=torch.Generator("cuda").manual_seed(42),
    num_inference_steps=25
).images[0]

image.save("builtin_with_latent_capture.png")