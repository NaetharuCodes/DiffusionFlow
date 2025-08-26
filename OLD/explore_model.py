import torch
from diffusers import StableDiffusionXLPipeline
import os

model_path = "./model/zc.safetensors"

print(f"Loading local model: {model_path}")

pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)

print("Successfully loaded with from_single_file!")
print("Model loaded! Let's explore its components...")
print(f"Pipeline components: {list(pipe.components.keys())}")

# Let's examine each component more carefully
for name, component in pipe.components.items():
    print(f"\n{name}: {type(component)}")
    
    # Handle different config types
    if hasattr(component, 'config'):
        config = component.config
        if hasattr(config, 'keys'):
            print(f"  Config keys: {list(config.keys())}")
        else:
            # For transformers configs, show the dict representation
            print(f"  Config type: {type(config)}")
            if hasattr(config, '__dict__'):
                print(f"  Config attributes: {list(config.__dict__.keys())}")