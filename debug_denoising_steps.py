import torch
from diffusers import StableDiffusionXLPipeline

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Let's hook into the scheduler to see what values it's producing
original_step = pipe.scheduler.step

def debug_step(model_output, timestep, sample, **kwargs):
    print(f"Timestep {timestep}: model_output range [{model_output.min():.3f}, {model_output.max():.3f}]")
    print(f"Timestep {timestep}: sample_in range [{sample.min():.3f}, {sample.max():.3f}]")
    result = original_step(model_output, timestep, sample, **kwargs)
    sample_out = result.prev_sample if hasattr(result, 'prev_sample') else result[0]
    print(f"Timestep {timestep}: sample_out range [{sample_out.min():.3f}, {sample_out.max():.3f}]\n")
    return result

pipe.scheduler.step = debug_step

# Run a quick generation to see the step-by-step values
print("Built-in pipeline denoising steps:")
image = pipe(
    prompt="cinematic portrait of a man, professional lighting, high quality, detailed",
    generator=torch.Generator("cuda").manual_seed(42),
    num_inference_steps=5,  # Just a few steps to see the pattern
    guidance_scale=8.0
).images[0]