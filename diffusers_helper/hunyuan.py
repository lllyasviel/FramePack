import torch
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import (
    DEFAULT_PROMPT_TEMPLATE,
)
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)

from diffusers_helper.utils import crop_or_pad_yield_mask
import torch.nn.functional as F
import math

@torch.no_grad()
def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)

    prompt = [prompt]

    # LLAMA

    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)
    llama_attention_length = int(llama_attention_mask.sum())

    llama_outputs = text_encoder(
        input_ids=llama_input_ids,
        attention_mask=llama_attention_mask,
        output_hidden_states=True,
    )

    llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
    # llama_vec_remaining = llama_outputs.hidden_states[-3][:, llama_attention_length:]
    llama_attention_mask = llama_attention_mask[:, crop_start:llama_attention_length]

    assert torch.all(llama_attention_mask.bool())

    # CLIP

    clip_l_input_ids = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    clip_l_pooler = text_encoder_2(clip_l_input_ids.to(text_encoder_2.device), output_hidden_states=False).pooler_output

    return llama_vec, clip_l_pooler


@torch.no_grad()
def vae_decode_fake(latents):
    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],
        [0.0696, 0.0795, 0.0518],
        [0.0135, -0.0945, -0.0282],
        [0.0108, -0.0250, -0.0765],
        [-0.0209, 0.0032, 0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991, 0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [0.1166, 0.1627, 0.0962],
        [0.1165, 0.0432, 0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270, 0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [0.0249, -0.0469, -0.1703]
    ]  # From comfyui

    latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.clamp(0.0, 1.0)

    return images


@torch.no_grad()
def vae_decode(latents, vae, image_mode=False):
    latents = latents / vae.config.scaling_factor

    if not image_mode:
        image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    else:
        latents = latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
        image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
        image = torch.cat(image, dim=2)

    return image

def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
            y / blend_extent
        )
    return b

@torch.no_grad()
def vae_decode_parallel(latents, vae, image_mode=False):
    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()
    latents = latents / vae.config.scaling_factor

    if not image_mode:
        if world_size > 1:
            # Distributed processing: split height dimension across GPUs
            blend_height = 4
            stride_height = math.ceil(latents.shape[3] / world_size)
            tile_height = blend_height + stride_height
            compression_rate = 8
            img_height_per_gpu = stride_height * compression_rate
            
            start_idx = rank * stride_height
            end_idx = start_idx + tile_height
            
            local_latents = latents[:, :, :, start_idx:end_idx, :].to(
                device=vae.device, dtype=vae.dtype
            )
            local_image = vae.decode(local_latents).sample
            
            if local_image.shape[3] != tile_height * compression_rate:
                local_image = F.pad(
                    local_image, 
                    (0, 0, 0, tile_height * compression_rate - local_image.shape[3])
                )

            # Gather results from all GPUs
            gathered_images = [torch.zeros_like(local_image) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_images, local_image)
            
            # Reconstruct full image with blending at boundaries
            image_list = []
            for i, local_image in enumerate(gathered_images):
                if i > 0:
                    # Apply blending between adjacent GPU boundaries
                    local_image = blend_v(gathered_images[i-1], local_image, blend_height * 8)

                image_list.append(local_image[:, :, :, :img_height_per_gpu, :])
            # Concatenate all processed portions
            image = torch.cat(image_list, dim=3)
        else:
            # Single GPU processing
            image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    else:
        if world_size > 1:
            # Distributed processing for image mode
            blend_height = 4
            stride_height = math.ceil(latents.shape[3] / world_size)
            tile_height = blend_height + stride_height
            compression_rate = 8
            img_height_per_gpu = stride_height * compression_rate  # Should be 192

            start_idx = rank * stride_height
            end_idx = start_idx + tile_height
            
            local_latents = latents[:, :, :, start_idx:end_idx, :].to(
                device=vae.device, dtype=vae.dtype
            )
            local_latents = local_latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
            local_image = [vae.decode(l.unsqueeze(2)).sample for l in local_latents]
            local_image = torch.cat(local_image, dim=2)
            
            if local_image.shape[3] != tile_height * compression_rate:
                local_image = F.pad(
                    local_image, 
                    (0, 0, 0, tile_height * compression_rate - local_image.shape[3])
                )

            # Gather results from all GPUs
            gathered_images = [torch.zeros_like(local_image) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_images, local_image)
            
            # Reconstruct full image with blending at boundaries
            image_list = []
            for i, local_image in enumerate(gathered_images):
                if i > 0:
                    # Apply blending between adjacent GPU boundaries
                    local_image = blend_v(gathered_images[i-1], local_image, blend_height * 8)
                
                image_list.append(local_image[:, :, :, :img_height_per_gpu, :])
            
            # Concatenate all processed portions
            image = torch.cat(image_list, dim=3)
        else:
            # Single GPU processing for image mode
            image = [vae.decode(latent.unsqueeze(2)).sample for latent in latents]
            image = torch.cat(image, dim=2)
   
    return image

@torch.no_grad()
def vae_encode(image, vae):
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents
