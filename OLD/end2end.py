import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np

# Load model
model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
device = "cuda"
pipe = pipe.to(device)

print("=== COMPLETE SDXL PIPELINE ===")

# Step 1: Generate initial noise
print("\n1. Creating latent noise...")
generator = torch.Generator(device=device).manual_seed(42)
latents = torch.randn(1, 4, 128, 128, generator=generator, device=device, dtype=torch.float16)
print(f"Initial noise range: [{latents.min():.3f}, {latents.max():.3f}]")

print(f"Initial noise shape: {latents.shape}")
print(f"Initial noise stats - mean: {latents.mean():.4f}, std: {latents.std():.4f}")
print(f"Initial noise range: [{latents.min():.4f}, {latents.max():.4f}]")

latents = latents * pipe.scheduler.init_noise_sigma
print(f"Scaled initial noise by {pipe.scheduler.init_noise_sigma}")

# Step 2: Encode prompts
print("\n2. Encoding text prompts...")
positive_prompt = "cinematic portrait of a man, professional lighting, high quality, detailed"
negative_prompt = "blurry, low quality, distorted, amateur"

with torch.no_grad():
    # Positive prompt
    pos_tokens_1 = pipe.tokenizer(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    pos_tokens_2 = pipe.tokenizer_2(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    pos_embeds_1 = pipe.text_encoder(pos_tokens_1.input_ids.to(device))[0]
    pos_embeds_2 = pipe.text_encoder_2(pos_tokens_2.input_ids.to(device))
    
    # Negative prompt  
    neg_tokens_1 = pipe.tokenizer(negative_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    neg_tokens_2 = pipe.tokenizer_2(negative_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    neg_embeds_1 = pipe.text_encoder(neg_tokens_1.input_ids.to(device))[0]
    neg_embeds_2 = pipe.text_encoder_2(neg_tokens_2.input_ids.to(device))

# Combine embeddings
positive_embeds = torch.cat([pos_embeds_1, pos_embeds_2.last_hidden_state], dim=-1)
negative_embeds = torch.cat([neg_embeds_1, neg_embeds_2.last_hidden_state], dim=-1)
pooled_positive = pos_embeds_2.text_embeds
pooled_negative = neg_embeds_2.text_embeds

# Time IDs for SDXL
time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=torch.float16)

print(f"Text embeddings ready: {positive_embeds.shape}")

# Step 3: Denoising loop
print("\n3. Denoising loop...")
pipe.scheduler.set_timesteps(25)
timesteps = pipe.scheduler.timesteps
guidance_scale = 8.5

for i, t in enumerate(timesteps):
    # Prepare latent model input
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    
    # Combine embeddings for classifier-free guidance
    text_embeddings = torch.cat([negative_embeds, positive_embeds])
    add_text_embeds = torch.cat([pooled_negative, pooled_positive])
    add_time_ids = torch.cat([time_ids, time_ids])
    
    # Predict the noise with UNet
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs={
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids
            },
            return_dict=False,
        )[0]
    
    # Classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # Take a denoising step
    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    if (i + 1) % 5 == 0:  # Print every 5 steps
        print(f"Step {i+1}: latent range [{latents.min():.3f}, {latents.max():.3f}]")

print(f"Final denoised latents: [{latents.min():.3f}, {latents.max():.3f}]")

# Before VAE decoding, let's inspect our latents
print(f"Pre-VAE latents shape: {latents.shape}")
print(f"Pre-VAE latents range: [{latents.min():.4f}, {latents.max():.4f}]")
print(f"Pre-VAE latents mean: {latents.mean():.4f}")
print(f"Pre-VAE latents std: {latents.std():.4f}")

# Save the latents for inspection
torch.save(latents.cpu(), "manual_final_latents.pt")

# Step 4: VAE Decode
print("\n4. Decoding to image...")
with torch.no_grad():
    # Scale the latents for VAE (this is the correct scaling)
    latents = latents / pipe.vae.config.scaling_factor  # Should be /0.13025
    
    # Decode to pixel space
    image = pipe.vae.decode(latents, return_dict=False)[0]

print(f"Decoded image range: [{image.min():.3f}, {image.max():.3f}]")

# Convert to PIL Image
image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image[0])

# Save the result!
pil_image.save("my_first_sdxl_dragon.png")
print(f"\n🎉 SUCCESS! Your dragon image saved as 'my_first_sdxl_dragon.png'")
print(f"Image size: {pil_image.size}")