import torch
from diffusers import StableDiffusionXLPipeline

# Load our model (should be cached now)
model_path = "./model/BLPN3.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)

device = "cuda"
pipe = pipe.to(device)

# Let's encode some text!
positive_prompt = "a majestic dragon flying over mountains at sunset"
negative_prompt = "blurry, low quality, distorted"

print(f"Encoding prompts...")
print(f"Positive: {positive_prompt}")
print(f"Negative: {negative_prompt}")

# SDXL uses TWO text encoders
text_encoder_1 = pipe.text_encoder
text_encoder_2 = pipe.text_encoder_2
tokenizer_1 = pipe.tokenizer
tokenizer_2 = pipe.tokenizer_2

print(f"\nText Encoder 1: {type(text_encoder_1).__name__}")
print(f"Text Encoder 2: {type(text_encoder_2).__name__}")

# Step 1: Tokenize (turn words into numbers)
print("\n=== TOKENIZATION ===")
pos_tokens_1 = tokenizer_1(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
pos_tokens_2 = tokenizer_2(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)

print(f"Tokenizer 1 - Token count: {pos_tokens_1.input_ids.shape}")
print(f"First 10 tokens: {pos_tokens_1.input_ids[0][:10].tolist()}")
print(f"Tokenizer 2 - Token count: {pos_tokens_2.input_ids.shape}")
print(f"First 10 tokens: {pos_tokens_2.input_ids[0][:10].tolist()}")

# Step 2: Encode (turn tokens into embeddings)
print("\n=== TEXT ENCODING ===")
with torch.no_grad():
    # Encoder 1 gives us per-token embeddings
    pos_embeds_1 = text_encoder_1(pos_tokens_1.input_ids.to(device))[0]
    
    # Encoder 2 gives us both per-token AND pooled embeddings
    encoder_2_output = text_encoder_2(pos_tokens_2.input_ids.to(device))
    pos_embeds_2 = encoder_2_output.last_hidden_state  # Per-token embeddings
    pooled_embeds_2 = encoder_2_output.text_embeds     # Pooled (single vector)

print(f"Encoder 1 output shape: {pos_embeds_1.shape}")      # [1, 77, 768]
print(f"Encoder 2 per-token shape: {pos_embeds_2.shape}")   # [1, 77, 1280] 
print(f"Encoder 2 pooled shape: {pooled_embeds_2.shape}")   # [1, 1280]

# SDXL concatenates the per-token embeddings
final_positive_embeds = torch.cat([pos_embeds_1, pos_embeds_2], dim=-1)
print(f"Combined embedding shape: {final_positive_embeds.shape}")  # [1, 77, 2048]

print(f"\nThis is how '{positive_prompt}' becomes math that the UNet understands!")
print(f"77 tokens × 2048 dimensions = {77 * 2048:,} numbers representing your prompt")
print(f"Plus a pooled vector of {pooled_embeds_2.shape[-1]} dimensions for global context")