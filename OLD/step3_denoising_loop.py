import torch
from diffusers import StableDiffusionXLPipeline
import matplotlib.pyplot as plt
import numpy as np

# Load model and move to GPU
model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
device = "cuda"
pipe = pipe.to(device)

print("Setting up the denoising process...")
print(f"Scheduler: {type(pipe.scheduler).__name__}")

# Our prompts
positive_prompt = "a majestic dragon flying over mountains at sunset"
negative_prompt = "blurry, low quality, distorted"

# Encode the text (same as before)
print("Encoding text prompts...")
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

# SDXL requires time_ids for resolution and other conditioning
# Format: [height, width, crop_top, crop_left, target_height, target_width]
time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=torch.float16)

print(f"Text embeddings ready: {positive_embeds.shape}")
print(f"Time IDs: {time_ids}")

# Create initial noise
generator = torch.Generator(device=device).manual_seed(42)
latents = torch.randn(1, 4, 128, 128, generator=generator, device=device, dtype=torch.float16)

# Set up scheduler for 20 denoising steps
pipe.scheduler.set_timesteps(20)
timesteps = pipe.scheduler.timesteps

print(f"Starting denoising with {len(timesteps)} steps...")

# THE DENOISING LOOP
guidance_scale = 7.5

for i, t in enumerate(timesteps):
    # Prepare latent model input
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    
    # Combine embeddings for classifier-free guidance
    text_embeddings = torch.cat([negative_embeds, positive_embeds])
    add_text_embeds = torch.cat([pooled_negative, pooled_positive])
    add_time_ids = torch.cat([time_ids, time_ids])  # For both negative and positive
    
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
    
    print(f"Step {i+1}/{len(timesteps)}: t={t}, latent range: [{latents.min():.3f}, {latents.max():.3f}]")

print("Denoising complete! Final latents ready for VAE decoding.")
print(f"Final latent shape: {latents.shape}")