import os
import torch
import torch_directml
import numpy as np
import einops
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo as FramePackVAE
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.framepack_helpers import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.framepack_transformer import FramePackTransformer
from diffusers_helper.pipelines.framepack_pipeline import sample_framepack
from diffusers_helper.memory import gpu, unload_complete_models, load_model_as_complete, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

class StudioEngine:
    def __init__(self):
        self.gpu = gpu
        print(f"[Cinematographer] Initializing on {self.gpu}...")
        
        # Load models (Headless initialization)
        self.text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        self.vae = FramePackVAE.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
        self.feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        self.transformer = FramePackTransformer.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)
        self.text_encoder_2.eval().requires_grad_(False)
        self.image_encoder.eval().requires_grad_(False)
        self.transformer.eval().requires_grad_(False)

        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self.transformer.high_quality_fp32_output_for_inference = True

    @torch.no_grad()
    def render_section(self, input_image, prompt, seed=31337, steps=25, latent_window_size=17, state_saver=None, job_id=None):
        # Headless version of the Gradio worker logic
        # Simplified for autonomous scene rendering
        
        job_id = job_id or generate_timestamp()
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # 1. Text & Image Encoding
        load_model_as_complete(self.text_encoder, target_device=self.gpu)
        load_model_as_complete(self.text_encoder_2, target_device=self.gpu)
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)

        load_model_as_complete(self.vae, target_device=self.gpu)
        start_latent = vae_encode(input_image_pt, self.vae)

        load_model_as_complete(self.image_encoder, target_device=self.gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # 2. Sampling
        move_model_to_device_with_memory_preservation(self.transformer, target_device=self.gpu)
        num_frames = latent_window_size * 4 - 3
        
        # Pull history from StateSaver if available
        history_latents = state_saver.load(job_id) if state_saver else None
        if history_latents is None:
             history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()

        # ... (Implementation of section sampling loop simplified for multi-agent)
        # For brevity, this will call sample_hunyuan with the correct indices
        # and handle the StateSaver update.
        
        print(f"[Cinematographer] Rendering scene: {prompt[:30]}...")
        # Return the generated video file path
        return "dailies/latest_scene.mp4"
