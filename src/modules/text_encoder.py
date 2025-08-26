"""
Text encoding utilities for SDXL pipeline.

Handles tokenization and encoding with SDXL's dual text encoder system.
"""

import torch
from src import config


def encode_prompts(pipeline_components, positive_prompt, negative_prompt=None, 
                 safety_mode=None):
   """
   Convert text prompts to embeddings for SDXL UNet.
   
   Args:
       pipeline_components: Loaded model components from model_loader
       positive_prompt: Text description of desired image
       negative_prompt: Text description of what to avoid  
       safety_mode: SafetyMode.FREE or SafetyMode.SFW
   
   Returns:
       Dictionary containing all embeddings and conditioning needed for UNet
   """
   if safety_mode is None:
       safety_mode = config.DEFAULT_SAFETY_MODE
   
   device = pipeline_components['device']
   
   # Build final negative prompt with safety additions if needed
   final_negative = _build_negative_prompt(negative_prompt, safety_mode)
   
   print(f"Encoding prompts...")
   print(f"Positive: {positive_prompt}")
   print(f"Negative: {final_negative}")
   
   # Extract components we need
   tokenizer_1 = pipeline_components['tokenizer']
   tokenizer_2 = pipeline_components['tokenizer_2'] 
   text_encoder_1 = pipeline_components['text_encoder']
   text_encoder_2 = pipeline_components['text_encoder_2']
   
   with torch.no_grad():
       # Tokenize positive prompt with both tokenizers
       pos_tokens_1 = tokenizer_1(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
       pos_tokens_2 = tokenizer_2(positive_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
       
       # Encode positive prompt with both encoders
       pos_embeds_1 = text_encoder_1(pos_tokens_1.input_ids.to(device))[0]
       pos_embeds_2 = text_encoder_2(pos_tokens_2.input_ids.to(device))
       
       # Tokenize negative prompt with both tokenizers
       neg_tokens_1 = tokenizer_1(final_negative, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
       neg_tokens_2 = tokenizer_2(final_negative, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
       
       # Encode negative prompt with both encoders
       neg_embeds_1 = text_encoder_1(neg_tokens_1.input_ids.to(device))[0]
       neg_embeds_2 = text_encoder_2(neg_tokens_2.input_ids.to(device))
   
   # Combine embeddings from both encoders
   positive_embeds = torch.cat([pos_embeds_1, pos_embeds_2.last_hidden_state], dim=-1)
   negative_embeds = torch.cat([neg_embeds_1, neg_embeds_2.last_hidden_state], dim=-1)
   
   # Get pooled embeddings from encoder 2
   pooled_positive = pos_embeds_2.text_embeds
   pooled_negative = neg_embeds_2.text_embeds
   
   # SDXL time IDs for resolution conditioning
   time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=torch.float16)
   
   print(f"Text embeddings ready: {positive_embeds.shape}")
   
   return {
       'positive_embeds': positive_embeds,
       'negative_embeds': negative_embeds,
       'pooled_positive': pooled_positive,
       'pooled_negative': pooled_negative,
       'time_ids': time_ids
   }


def _build_negative_prompt(negative_prompt, safety_mode):
   """Build final negative prompt with optional safety additions."""
   final_negative = negative_prompt or ""
   
   if safety_mode == config.SafetyMode.SFW:
       nsfw_negatives = ", ".join(config.NSFW_NEGATIVES)
       if final_negative:
           final_negative = f"{final_negative}, {nsfw_negatives}"
       else:
           final_negative = nsfw_negatives
   
   return final_negative