"""
Configuration settings for the SDXL pipeline.
"""

# Model settings
DEFAULT_MODEL_PATH = "./model/zc.safetensors"  # You can change this to your preferred default

# Device settings
DEFAULT_DEVICE = "cuda"  # Will auto-detect if not available

# Generation defaults
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024

# Operation modes
class SafetyMode:
    FREE = "free"       # No content filtering (default)
    SFW = "sfw"         # Add NSFW-blocking negatives

DEFAULT_SAFETY_MODE = SafetyMode.FREE

# Safety negatives (only used when SFW mode is explicitly chosen)
NSFW_NEGATIVES = ["nsfw", "nude", "explicit", "sexual", "naked"]