from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

def handle_magcache_change(magcache_value, teacache_value):
    """
    Handles the change event for the 'use_magcache' checkbox.
    Ensures that 'use_teacache' is unchecked if 'use_magcache' is checked.
    """
    if magcache_value and teacache_value:
        # If magcache was just checked AND teacache was already checked,
        # uncheck teacache.
        return gr.update(value=True), gr.update(value=False)
    # Otherwise, return current values to avoid unintended changes
    return gr.update(value=magcache_value), gr.update(value=teacache_value)

def handle_teacache_change(magcache_value, teacache_value):
    """
    Handles the change event for the 'use_teacache' checkbox.
    Ensures that 'use_magcache' is unchecked if 'use_teacache' is checked.
    """
    if magcache_value and teacache_value:
        # If teacache was just checked AND magcache was already checked,
        # uncheck magcache.
        return gr.update(value=False), gr.update(value=True)
    # Otherwise, return current values to avoid unintended changes
    return gr.update(value=magcache_value), gr.update(value=teacache_value)

def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]

def initialize_magcache(self, enable_magcache=True, num_steps=25, magcache_thresh=0.1, K=2, retention_ratio=0.2):
    self.enable_magcache = enable_magcache
    self.cnt = 0
    self.num_steps = num_steps
    self.magcache_thresh = magcache_thresh
    self.K = K
    self.retention_ratio = retention_ratio
    self.mag_ratios = np.array([1.0]+[1.26562, 1.23438, 1.03125, 1.02344, 1.03906, 1.01562, 1.03906, 1.05469, 1.02344, 1.03906, 0.99609, 1.03125, 1.02344, 1.01562, 1.02344, 1.00781, 1.04688, 0.98828, 1.0, 1.00781, 0.98828, 0.94141, 0.94141, 0.78906])
    # Nearest interpolation when the num_steps is different from the length of mag_ratios
    if len(self.mag_ratios) != num_steps:
        interpolated_mag_ratios = nearest_interp(self.mag_ratios, num_steps)
        self.mag_ratios = interpolated_mag_ratios
        
def magcache_framepack_calibration(
        self,
        hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance,
        latent_indices=None,
        clean_latents=None, clean_latent_indices=None,
        clean_latents_2x=None, clean_latent_2x_indices=None,
        clean_latents_4x=None, clean_latent_4x_indices=None,
        image_embeddings=None,
        attention_kwargs=None, return_dict=True
    ):
    """
        Calibration function for `mag_ratios`, requiring only a single prompt/input.
        Please recalibrate `mag_ratios` if the number of inference steps differs significantly from the predefined value (25),
        or if the scheduler or solver is modified.
    """
    if attention_kwargs is None:
        attention_kwargs = {}

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config['patch_size'], self.config['patch_size_t']
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

    hidden_states, rope_freqs = self.process_input_hidden_states(hidden_states, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)

    temb = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
    encoder_hidden_states = self.gradient_checkpointing_method(self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask)

    if self.image_projection is not None:
        assert image_embeddings is not None, 'You must use image embeddings!'
        extra_encoder_hidden_states = self.gradient_checkpointing_method(self.image_projection, image_embeddings)
        extra_attention_mask = torch.ones((batch_size, extra_encoder_hidden_states.shape[1]), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)

        # must cat before (not after) encoder_hidden_states, due to attn masking
        encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

    if batch_size == 1:
        # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
        # If they are not same, then their impls are wrong. Ours are always the correct one.
        text_len = encoder_attention_mask.sum().item()
        encoder_hidden_states = encoder_hidden_states[:, :text_len]
        attention_mask = None, None, None, None
    else:
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]

        cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    if self.cnt == 0 :
        self.norm_ratio, self.norm_std, self.cos_dis = [], [], []
    
    ori_hidden_states = hidden_states.clone()
    for block_id, block in enumerate(self.transformer_blocks):
        hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
            block,
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            rope_freqs
        )

    for block_id, block in enumerate(self.single_transformer_blocks):
        hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
            block,
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            rope_freqs
        )
    cur_residual = hidden_states - ori_hidden_states
    
    if self.cnt >= 1:
        norm_ratio = ((cur_residual.norm(dim=-1)/self.previous_residual.norm(dim=-1)).mean()).item()
        norm_std = (cur_residual.norm(dim=-1)/self.previous_residual.norm(dim=-1)).std().item()
        cos_dis = (1-torch.nn.functional.cosine_similarity(cur_residual, self.previous_residual, dim=-1, eps=1e-8)).mean().item()
        self.norm_ratio.append(round(norm_ratio, 5))
        self.norm_std.append(round(norm_std, 5))
        self.cos_dis.append(round(cos_dis, 5))
        print(f"time: {self.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")
    
    self.previous_residual = cur_residual
    self.cnt += 1
    if self.cnt == self.num_steps:
        print("norm ratio")
        print(self.norm_ratio)
        print("norm std")
        print(self.norm_std)
        print("cos_dis")
        print(self.cos_dis)
        self.cnt = 0
        self.norm_ratio = []
        self.norm_std = []
        self.cos_dis = []
            
    hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)

    hidden_states = hidden_states[:, -original_context_length:, :]

    if self.high_quality_fp32_output_for_inference:
        hidden_states = hidden_states.to(dtype=torch.float32)
        if self.proj_out.weight.dtype != torch.float32:
            self.proj_out.to(dtype=torch.float32)

    hidden_states = self.gradient_checkpointing_method(self.proj_out, hidden_states)

    hidden_states = einops.rearrange(hidden_states, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                                        t=post_patch_num_frames, h=post_patch_height, w=post_patch_width,
                                        pt=p_t, ph=p, pw=p)

    if return_dict:
        return Transformer2DModelOutput(sample=hidden_states)

    return hidden_states,

def magcache_framepack_forward(
        self,
        hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance,
        latent_indices=None,
        clean_latents=None, clean_latent_indices=None,
        clean_latents_2x=None, clean_latent_2x_indices=None,
        clean_latents_4x=None, clean_latent_4x_indices=None,
        image_embeddings=None,
        attention_kwargs=None, return_dict=True
    ):

    if attention_kwargs is None:
        attention_kwargs = {}

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config['patch_size'], self.config['patch_size_t']
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

    hidden_states, rope_freqs = self.process_input_hidden_states(hidden_states, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)

    temb = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
    encoder_hidden_states = self.gradient_checkpointing_method(self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask)

    if self.image_projection is not None:
        assert image_embeddings is not None, 'You must use image embeddings!'
        extra_encoder_hidden_states = self.gradient_checkpointing_method(self.image_projection, image_embeddings)
        extra_attention_mask = torch.ones((batch_size, extra_encoder_hidden_states.shape[1]), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)

        # must cat before (not after) encoder_hidden_states, due to attn masking
        encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

    if batch_size == 1:
        # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
        # If they are not same, then their impls are wrong. Ours are always the correct one.
        text_len = encoder_attention_mask.sum().item()
        encoder_hidden_states = encoder_hidden_states[:, :text_len]
        attention_mask = None, None, None, None
    else:
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]

        cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    if self.enable_magcache:
        if self.cnt == 0: # initialize MagCache
            self.accumulated_ratio = 1.0
            self.accumulated_steps = 0
            self.accumulated_err = 0
            
        skip_forward = False
        if self.cnt>=int(self.retention_ratio*self.num_steps) and self.cnt>=1: # keep first retention_ratio steps
            cur_mag_ratio = self.mag_ratios[self.cnt]
            self.accumulated_ratio = self.accumulated_ratio*cur_mag_ratio
            cur_skip_err = np.abs(1-self.accumulated_ratio)
            self.accumulated_err += cur_skip_err
            self.accumulated_steps += 1
            if self.accumulated_err<=self.magcache_thresh and self.accumulated_steps<=self.K and np.abs(1-cur_mag_ratio)<=0.06:
                skip_forward = True
            else:
                self.accumulated_ratio = 1.0
                self.accumulated_steps = 0
                self.accumulated_err = 0
        
        if skip_forward:
            hidden_states = hidden_states + self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()

            for block_id, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    rope_freqs
                )

            for block_id, block in enumerate(self.single_transformer_blocks):
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    rope_freqs
                )

            self.previous_residual = hidden_states - ori_hidden_states
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    else:
        for block_id, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                rope_freqs
            )

        for block_id, block in enumerate(self.single_transformer_blocks):
            hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                rope_freqs
            )

    hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)

    hidden_states = hidden_states[:, -original_context_length:, :]

    if self.high_quality_fp32_output_for_inference:
        hidden_states = hidden_states.to(dtype=torch.float32)
        if self.proj_out.weight.dtype != torch.float32:
            self.proj_out.to(dtype=torch.float32)

    hidden_states = self.gradient_checkpointing_method(self.proj_out, hidden_states)

    hidden_states = einops.rearrange(hidden_states, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                                        t=post_patch_num_frames, h=post_patch_height, w=post_patch_width,
                                        pt=p_t, ph=p, pw=p)

    if return_dict:
        return Transformer2DModelOutput(sample=hidden_states)

    return hidden_states,

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)
orig_forward = transformer.__class__.forward

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, mp4_crf, use_teacache=False, magcache_thresh=0.1, magcache_K=3, magcache_retention_ratio=0.2):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_magcache:
                # magcache_monkey_patch
                transformer.__class__.forward = magcache_framepack_forward # replace with magcache_framepack_calibration when recalibrating the mag_ratios with different inference steps.
                transformer.__class__.initialize_magcache = initialize_magcache
                transformer.initialize_magcache(enable_magcache=True, num_steps=steps, magcache_thresh=magcache_thresh, K=magcache_K, retention_ratio=magcache_retention_ratio)
            elif use_teacache:
                transformer.__class__.forward = orig_forward
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.__class__.forward = orig_forward
                transformer.__class__.initialize_magcache = initialize_magcache
                transformer.initialize_teacache(enable_teacache=False)
                transformer.initialize_magcache(enable_magcache=False)
                
            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, mp4_crf, use_teacache=False, magcache_thresh=0.1, magcache_K=3, magcache_retention_ratio=0.20):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, mp4_crf, use_teacache, magcache_thresh, magcache_K, magcache_retention_ratio)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                with gr.Row():
                    use_magcache = gr.Checkbox(label='Use magcache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    use_teacache = gr.Checkbox(label='Use teacache', value=False, info='Faster speed, but often makes hands and fingers slightly worse. only support magcache or teacache')
                # Apply the new mutual exclusion logic
                use_magcache.change(
                    fn=handle_magcache_change,
                    inputs=[use_magcache, use_teacache],
                    outputs=[use_magcache, use_teacache]
                )
                use_teacache.change(
                    fn=handle_teacache_change,
                    inputs=[use_magcache, use_teacache],
                    outputs=[use_magcache, use_teacache]
                )
                
                magcache_thresh = gr.Slider(label="MagCache_thresh", minimum=0.0, maximum=1.0, value=0.10, step=0.005, info='Decrease this value when the quality is poor. It denotes the accumulated error caused by skipping steps.') 
                
                magcache_K = gr.Slider(label="MagCache_K", minimum=1, maximum=5, value=3, step=1, info='Decrease this value when the quality is poor. 0 means forbidding magcache.')
                magcache_rention_ratio = gr.Slider(label="MagCache retension ratio", minimum=0.0, maximum=1.0, value=0.2, step=0.01, info='Increase this ratio to make the video more consistent with the video generated without MagCache. Retain the first x% of steps to preserve semantic consistency.')  # Should not change
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)
                # assert (use_teacache and use_magcache) is False, "use_magcache and use_teacache can not be enable at the same time."
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, mp4_crf, use_teacache, magcache_thresh, magcache_K, magcache_rention_ratio]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
