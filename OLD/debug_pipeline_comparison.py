import torch
from diffusers import StableDiffusionXLPipeline
import numpy as np

model_path = "./model/zc.safetensors"

# We'll capture data at these points:
# 1. Initial noise (before and after scaling)
# 2. Text embeddings
# 3. Latents after each denoising step
# 4. Final latents before VAE
# 5. VAE input (after scaling)

debug_data = {
    "builtin": {},
    "manual": {}
}