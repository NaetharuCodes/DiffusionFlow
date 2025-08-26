import torch
import matplotlib.pyplot as plt
import numpy as np

# Set up our device and generator for reproducible results
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device).manual_seed(42)

print(f"Working on: {device}")

# SDXL works in latent space - images are compressed 8x in each dimension
# So 1024x1024 pixel image = 128x128 latent
height, width = 1024, 1024
latent_height, latent_width = height // 8, width // 8

print(f"Pixel size: {height}x{width}")
print(f"Latent size: {latent_height}x{latent_width}")

# Create random latent noise
# Shape: [batch_size, channels, height, width]
# SDXL latents have 4 channels
latent_noise = torch.randn(
    1, 4, latent_height, latent_width,
    generator=generator,
    device=device,
    dtype=torch.float16
)

print(f"Latent noise shape: {latent_noise.shape}")
print(f"Latent noise range: {latent_noise.min().item():.3f} to {latent_noise.max().item():.3f}")

# Let's visualize what this noise looks like
# Convert to numpy for matplotlib
noise_np = latent_noise.cpu().numpy()

# Plot each of the 4 channels
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Raw Latent Noise (4 Channels)', fontsize=16)

for i in range(4):
    row, col = i // 2, i % 2
    axes[row, col].imshow(noise_np[0, i], cmap='RdBu', vmin=-3, vmax=3)
    axes[row, col].set_title(f'Channel {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('latent_noise_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved visualization as 'latent_noise_visualization.png'")
print(f"This noise will be our starting point for generation!")