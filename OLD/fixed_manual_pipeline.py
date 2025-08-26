import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

print("=== FIXED MANUAL PIPELINE ===")

# Step 1: Create noise and apply proper scaling
generator = torch.Generator(device="cuda").manual_seed(42)
latents = torch.randn(1, 4, 128, 128, generator=generator, device="cuda", dtype=torch.float16)

# Apply the scaling that built-in pipeline uses
latents = latents * pipe.scheduler.init_noise_sigma
print(f"Initial scaled noise: [{latents.min():.3f}, {latents.max():.3f}]")

# Step 2: Text encoding (same as before)
positive_prompt = "cinematic portrait of a man, professional lighting, high quality, detailed"
negative_prompt = "blurry, low quality, distorted, amateur"

with torch.no_grad():
    pos_tokens_1 = pipe.tokenizer(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    pos_tokens_2 = pipe.tokenizer_2(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    pos_embeds_1 = pipe.text_encoder(pos_tokens_1.input_ids.to("cuda"))[0]
    pos_embeds_2 = pipe.text_encoder_2(pos_tokens_2.input_ids.to("cuda"))
    
    neg_tokens_1 = pipe.tokenizer(negative_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    neg_tokens_2 = pipe.tokenizer_2(negative_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    neg_embeds_1 = pipe.text_encoder(neg_tokens_1.input_ids.to("cuda"))[0]
    neg_embeds_2 = pipe.text_encoder_2(neg_tokens_2.input_ids.to("cuda"))

positive_embeds = torch.cat([pos_embeds_1, pos_embeds_2.last_hidden_state], dim=-1)
negative_embeds = torch.cat([neg_embeds_1, neg_embeds_2.last_hidden_state], dim=-1)
pooled_positive = pos_embeds_2.text_embeds
pooled_negative = neg_embeds_2.text_embeds

time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda", dtype=torch.float16)

# Step 3: Denoising with proper scaling
pipe.scheduler.set_timesteps(25)
timesteps = pipe.scheduler.timesteps

for i, t in enumerate(timesteps):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    
    text_embeddings = torch.cat([negative_embeds, positive_embeds])
    add_text_embeds = torch.cat([pooled_negative, pooled_positive])
    add_time_ids = torch.cat([time_ids, time_ids])
    
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
    
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 8.0 * (noise_pred_text - noise_pred_uncond)
    
    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    if (i + 1) % 5 == 0:
        print(f"Step {i+1}: [{latents.min():.3f}, {latents.max():.3f}]")

# Step 4: VAE decode (no additional scaling - scheduler handles it)
print("Final latents before VAE:", f"[{latents.min():.3f}, {latents.max():.3f}]")

with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image[0])

pil_image.save("fixed_manual_result.png")
print("Saved fixed_manual_result.png")