import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DDIMScheduler

model_path = "./model/zc.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

prompt = "cinematic portrait of a man, professional lighting, high quality, detailed"
seed = 42

# Test 1: Default scheduler (whatever it came with)
print(f"Default scheduler: {type(pipe.scheduler).__name__}")
image1 = pipe(prompt=prompt, generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=25).images[0]
image1.save("sampler_default.png")

# Test 2: DPMSolverMultistep
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
print(f"DPM++ scheduler: {type(pipe.scheduler).__name__}")
image2 = pipe(prompt=prompt, generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=25).images[0]
image2.save("sampler_dpm.png")

# Test 3: DDIM
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
print(f"DDIM scheduler: {type(pipe.scheduler).__name__}")
image3 = pipe(prompt=prompt, generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=25).images[0]
image3.save("sampler_ddim.png")