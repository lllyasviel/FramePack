import os
import argparse
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import logging
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInitializer:
    def __init__(self, model_path="hunyuanvideo-community/HunyuanVideo"):
        self.model_path = model_path
        self.free_mem_gb = get_cuda_free_memory_gb(gpu)
        self.high_vram = self.free_mem_gb > 60

        logger.info(f'Free VRAM {self.free_mem_gb} GB')
        logger.info(f'High-VRAM Mode: {self.high_vram}')

        self.text_encoder = LlamaModel.from_pretrained(self.model_path, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(self.model_path, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model_path, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.model_path, subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(self.model_path, subfolder='vae', torch_dtype=torch.float16).cpu()

        self.feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        self.transformer.high_quality_fp32_output_for_inference = True
        logger.info('transformer.high_quality_fp32_output_for_inference = True')

        self.transformer.to(dtype=torch.bfloat16)
        self.vae.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.float16)
        self.text_encoder_2.to(dtype=torch.float16)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        if not self.high_vram:
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
        else:
            self.text_encoder.to(gpu)
            self.text_encoder_2.to(gpu)
            self.image_encoder.to(gpu)
            self.vae.to(gpu)
            self.transformer.to(gpu)

        self.stream = AsyncStream()

        self.outputs_folder = './outputs/'
        os.makedirs(self.outputs_folder, exist_ok=True)


@torch.no_grad()
def generate_video(initializer, input_image):
    try:
        total_latent_sections = (args.total_second_length * 30) / (args.latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        job_id = generate_timestamp()

        initializer.stream.output_queue.push(('progress', (None, '', 'Starting ...')))

        if not initializer.high_vram:
            unload_complete_models(
                initializer.text_encoder, initializer.text_encoder_2, initializer.image_encoder, initializer.vae, initializer.transformer
            )

        llama_vec, clip_l_pooler = encode_prompt_conds(args.prompt, initializer.text_encoder, initializer.text_encoder_2, initializer.tokenizer, initializer.tokenizer_2)

        if args.cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(args.n_prompt, initializer.text_encoder, initializer.text_encoder_2, initializer.tokenizer, initializer.tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(initializer.outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        if not initializer.high_vram:
            load_model_as_complete(initializer.vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, initializer.vae)

        if not initializer.high_vram:
            load_model_as_complete(initializer.image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, initializer.feature_extractor, initializer.image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        llama_vec = llama_vec.to(initializer.transformer.dtype)
        llama_vec_n = llama_vec_n.to(initializer.transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(initializer.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(initializer.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(initializer.transformer.dtype)

        rnd = torch.Generator("cpu").manual_seed(args.seed)
        num_frames = args.latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * args.latent_window_size

            logger.info(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, args.latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, args.latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not initializer.high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(initializer.transformer, target_device=gpu, preserved_memory_gb=args.gpu_memory_preservation)

            if args.use_teacache:
                initializer.transformer.initialize_teacache(enable_teacache=True, num_steps=args.steps)
            else:
                initializer.transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                logger.info(f'Sampling {d["i"] + 1}/{args.steps}')

            generated_latents = sample_hunyuan(
                transformer=initializer.transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=args.cfg,
                distilled_guidance_scale=args.gs,
                guidance_rescale=args.rs,
                num_inference_steps=args.steps,
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

            if not initializer.high_vram:
                offload_model_from_device_for_memory_preservation(initializer.transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(initializer.vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, initializer.vae).cpu()
            else:
                section_latent_frames = (args.latent_window_size * 2 + 1) if is_last_section else (args.latent_window_size * 2)
                overlapped_frames = args.latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], initializer.vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not initializer.high_vram:
                unload_complete_models()

            output_filename = os.path.join(initializer.outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)

            logger.info(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

        return output_filename
    except Exception as e:
        logger.error("An error occurred during video generation", exc_info=True)
        if not initializer.high_vram:
            unload_complete_models(
                initializer.text_encoder, initializer.text_encoder_2, initializer.image_encoder, initializer.vae, initializer.transformer
            )
        raise e


if __name__ == "__main__":
    # Define the command-line argument parser
    parser = argparse.ArgumentParser(description="Generate a video. ")

    # Add arguments
    parser.add_argument("--model_path", type=str, default="hunyuanvideo-community/HunyuanVideo", help="Path to the pre-trained model.")
    parser.add_argument("--input_image_path", type=str, default="input.jpg", help="Path to the input image.")
    parser.add_argument("--prompt", type=str, default="A character doing some simple body movements.", help="Prompt for the video generation.")
    parser.add_argument("--n_prompt", type=str, default="", help="Negative prompt for the video generation.")
    parser.add_argument("--seed", type=int, default=31337, help="Random seed for the video generation.")
    parser.add_argument("--total_second_length", type=int, default=5, help="Total length of the video in seconds.")
    parser.add_argument("--latent_window_size", type=int, default=9, help="Latent window size for the video generation.")
    parser.add_argument("--steps", type=int, default=25, help="Number of sampling steps.")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale for the video generation.")
    parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG scale for the video generation.")
    parser.add_argument("--rs", type=float, default=0.0, help="CFG rescale for the video generation.")
    parser.add_argument("--gpu_memory_preservation", type=int, default=6, help="GPU memory preservation in GB.")
    parser.add_argument("--use_teacache", type=bool, default=True, help="Whether to use TeaCache.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Initialize the model
    initializer = ModelInitializer(model_path=args.model_path)

    # Prepare the input image
    input_image = np.array(Image.open(args.input_image_path))

    # Generate the video
    output_video_path = generate_video(
        initializer,
        input_image,
    )

    logger.info(f"The generated video has been saved to: {output_video_path}")