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
# 20250506 pftq: Added for video input loading
import decord
# 20250506 pftq: Added for progress bars in video_encode
from tqdm import tqdm
# 20250506 pftq: Normalize file paths for Windows compatibility
import pathlib
# 20250506 pftq: for easier to read timestamp
from datetime import datetime
# 20250508 pftq: for saving prompt to mp4 comments metadata
import imageio_ffmpeg
import tempfile
import shutil
import subprocess

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

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

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

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()

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

# 20250506 pftq: Added function to encode input video frames into latents
@torch.no_grad()
def video_encode(video_path, resolution, no_resize, vae, vae_batch_size=16, device="cuda", width=None, height=None):
    """
    Encode a video into latent representations using the VAE.
    
    Args:
        video_path: Path to the input video file.
        vae: AutoencoderKLHunyuanVideo model.
        height, width: Target resolution for resizing frames.
        vae_batch_size: Number of frames to process per batch.
        device: Device for computation (e.g., "cuda").
    
    Returns:
        start_latent: Latent of the first frame (for compatibility with original code).
        input_image_np: First frame as numpy array (for CLIP vision encoding).
        history_latents: Latents of all frames (shape: [1, channels, frames, height//8, width//8]).
        fps: Frames per second of the input video.
    """
    # 20250506 pftq: Normalize video path for Windows compatibility
    video_path = str(pathlib.Path(video_path).resolve())
    print(f"Processing video: {video_path}")

    # 20250506 pftq: Check CUDA availability and fallback to CPU if needed
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = "cpu"

    try:
        # 20250506 pftq: Load video and get FPS
        print("Initializing VideoReader...")
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()  # Get input video FPS
        num_real_frames = len(vr)
        print(f"Video loaded: {num_real_frames} frames, FPS: {fps}")

        # Truncate to nearest latent size (multiple of 4)
        latent_size_factor = 4
        num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
        if num_frames != num_real_frames:
            print(f"Truncating video from {num_real_frames} to {num_frames} frames for latent size compatibility")
        num_real_frames = num_frames

        # 20250506 pftq: Read frames
        print("Reading video frames...")
        frames = vr.get_batch(range(num_real_frames)).asnumpy()  # Shape: (num_real_frames, height, width, channels)
        print(f"Frames read: {frames.shape}")

        # 20250506 pftq: Get native video resolution
        native_height, native_width = frames.shape[1], frames.shape[2]
        print(f"Native video resolution: {native_width}x{native_height}")
    
        # 20250506 pftq: Use native resolution if height/width not specified, otherwise use provided values
        target_height = native_height if height is None else height
        target_width = native_width if width is None else width
    
        # 20250506 pftq: Adjust to nearest bucket for model compatibility
        if not no_resize:
            target_height, target_width = find_nearest_bucket(target_height, target_width, resolution=resolution)
            print(f"Adjusted resolution: {target_width}x{target_height}")
        else:
            print(f"Using native resolution without resizing: {target_width}x{target_height}")

        # 20250506 pftq: Preprocess frames to match original image processing
        processed_frames = []
        for i, frame in enumerate(frames):
            #print(f"Preprocessing frame {i+1}/{num_frames}")
            frame_np = resize_and_center_crop(frame, target_width=target_width, target_height=target_height)
            processed_frames.append(frame_np)
        processed_frames = np.stack(processed_frames)  # Shape: (num_real_frames, height, width, channels)
        print(f"Frames preprocessed: {processed_frames.shape}")

        # 20250506 pftq: Save first frame for CLIP vision encoding
        input_image_np = processed_frames[0]

        # 20250506 pftq: Convert to tensor and normalize to [-1, 1]
        print("Converting frames to tensor...")
        frames_pt = torch.from_numpy(processed_frames).float() / 127.5 - 1
        frames_pt = frames_pt.permute(0, 3, 1, 2)  # Shape: (num_real_frames, channels, height, width)
        frames_pt = frames_pt.unsqueeze(0)  # Shape: (1, num_real_frames, channels, height, width)
        frames_pt = frames_pt.permute(0, 2, 1, 3, 4)  # Shape: (1, channels, num_real_frames, height, width)
        print(f"Tensor shape: {frames_pt.shape}")
        
        # 20250507 pftq: Save pixel frames for use in worker
        input_video_pixels = frames_pt.cpu()

        # 20250506 pftq: Move to device
        print(f"Moving tensor to device: {device}")
        frames_pt = frames_pt.to(device)
        print("Tensor moved to device")

        # 20250506 pftq: Move VAE to device
        print(f"Moving VAE to device: {device}")
        vae.to(device)
        print("VAE moved to device")

        # 20250506 pftq: Encode frames in batches
        print(f"Encoding input video frames in VAE batch size {vae_batch_size} (reduce if memory issues here or if forcing video resolution)")
        latents = []
        vae.eval()
        with torch.no_grad():
            for i in tqdm(range(0, frames_pt.shape[2], vae_batch_size), desc="Encoding video frames", mininterval=0.1):
                #print(f"Encoding batch {i//vae_batch_size + 1}: frames {i} to {min(i + vae_batch_size, frames_pt.shape[2])}")
                batch = frames_pt[:, :, i:i + vae_batch_size]  # Shape: (1, channels, batch_size, height, width)
                try:
                    # 20250506 pftq: Log GPU memory before encoding
                    if device == "cuda":
                        free_mem = torch.cuda.memory_allocated() / 1024**3
                        #print(f"GPU memory before encoding: {free_mem:.2f} GB")
                    batch_latent = vae_encode(batch, vae)
                    # 20250506 pftq: Synchronize CUDA to catch issues
                    if device == "cuda":
                        torch.cuda.synchronize()
                        #print(f"GPU memory after encoding: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    latents.append(batch_latent)
                    #print(f"Batch encoded, latent shape: {batch_latent.shape}")
                except RuntimeError as e:
                    print(f"Error during VAE encoding: {str(e)}")
                    if device == "cuda" and "out of memory" in str(e).lower():
                        print("CUDA out of memory, try reducing vae_batch_size or using CPU")
                    raise
        
        # 20250506 pftq: Concatenate latents
        print("Concatenating latents...")
        history_latents = torch.cat(latents, dim=2)  # Shape: (1, channels, frames, height//8, width//8)
        print(f"History latents shape: {history_latents.shape}")

        # 20250506 pftq: Get first frame's latent
        start_latent = history_latents[:, :, :1]  # Shape: (1, channels, 1, height//8, width//8)
        print(f"Start latent shape: {start_latent.shape}")

        # 20250506 pftq: Move VAE back to CPU to free GPU memory
        if device == "cuda":
            vae.to(cpu)
            torch.cuda.empty_cache()
            print("VAE moved back to CPU, CUDA cache cleared")

        return start_latent, input_image_np, history_latents, fps, target_height, target_width, input_video_pixels

    except Exception as e:
        print(f"Error in video_encode: {str(e)}")
        raise

# 20250508 pftq: for saving prompt to mp4 metadata comments
def set_mp4_comments_imageio_ffmpeg(input_file, comments):
    try:
        # Get the path to the bundled FFmpeg binary from imageio-ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist")
            return False
            
        # Create a temporary file path
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # FFmpeg command using the bundled binary
        command = [
            ffmpeg_path,                   # Use imageio-ffmpeg's FFmpeg
            '-i', input_file,              # input file
            '-metadata', f'comment={comments}',  # set comment metadata
            '-c:v', 'copy',                # copy video stream without re-encoding
            '-c:a', 'copy',                # copy audio stream without re-encoding
            '-y',                          # overwrite output file if it exists
            temp_file                      # temporary output file
        ]
        
        # Run the FFmpeg command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            # Replace the original file with the modified one
            shutil.move(temp_file, input_file)
            print(f"Successfully added comments to {input_file}")
            return True
        else:
            # Clean up temp file if FFmpeg fails
            if os.path.exists(temp_file):
                os.remove(temp_file)
            print(f"Error: FFmpeg failed with message:\n{result.stderr}")
            return False
            
    except Exception as e:
        # Clean up temp file in case of other errors
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"Error saving prompt to video metadata, ffmpeg may be required: "+str(e))
        return False

# 20250506 pftq: Modified worker to accept video input and clean frame count
@torch.no_grad()
def worker(input_video, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch):
    
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

        # 20250506 pftq: Processing input video instead of image
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Video processing ...'))))

        # 20250506 pftq: Encode video
        #H, W = 640, 640  # Default resolution, will be adjusted
        #height, width = find_nearest_bucket(H, W, resolution=640)
        #start_latent, input_image_np, history_latents, fps = video_encode(input_video, vae, height, width, vae_batch_size=16, device=gpu)
        start_latent, input_image_np, video_latents, fps, height, width, input_video_pixels  = video_encode(input_video, resolution, no_resize, vae, vae_batch_size=vae_batch, device=gpu)

        #Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png')) 

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

        total_latent_sections = (total_second_length * fps) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        for idx in range(batch):
            if idx>0:
                seed = seed + 1
            
            if batch > 1:
                print(f"Beginning video {idx+1} of {batch} with seed {seed} ")
            
            #job_id = generate_timestamp()
            job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+f"_framepackf1-videoinput_{width}-{total_second_length}sec_seed-{seed}_steps-{steps}_distilled-{gs}_cfg-{cfg}" # 20250506 pftq: easier to read timestamp and filename
            
            # Sampling
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
    
            rnd = torch.Generator("cpu").manual_seed(seed)
    
            # 20250506 pftq: Initialize history_latents with video latents
            history_latents = video_latents.cpu()
            total_generated_latent_frames = history_latents.shape[2]
            # 20250506 pftq: Initialize history_pixels to fix UnboundLocalError
            history_pixels = None
            previous_video = None
            
            # 20250507 pftq: hot fix for initial video being corrupted by vae encoding, issue with ghosting because of slight differences
            #history_pixels = input_video_pixels 
            #save_bcthw_as_mp4(vae_decode(video_latents, vae).cpu(), os.path.join(outputs_folder, f'{job_id}_input_video.mp4'), fps=fps, crf=mp4_crf) # 20250507 pftq: test fast movement corrupted by vae encoding if vae batch size too low
            
            for section_index in range(total_latent_sections):
                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    return
    
                print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')
    
                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
    
                if use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)
    
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
                    desc = f'Total frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / fps) :.2f} seconds (FPS-{fps}), Seed: {seed}, Video {idx+1} of {batch}. The video is generating part {section_index+1} of {total_latent_sections}...'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                    return
    
                # 20250506 pftq: Use user-specified number of context frames, matching original allocation for num_clean_frames=2
                available_frames = history_latents.shape[2]  # Number of latent frames
                max_pixel_frames = min(latent_window_size * 4 - 3, available_frames * 4)  # Cap at available pixel frames
                adjusted_latent_frames = max(1, (max_pixel_frames + 3) // 4)  # Convert back to latent frames
                # Adjust num_clean_frames to match original behavior: num_clean_frames=2 means 1 frame for clean_latents_1x
                effective_clean_frames = max(0, num_clean_frames - 1) if num_clean_frames > 1 else 0
                effective_clean_frames = min(effective_clean_frames, available_frames - 2) if available_frames > 2 else 0 # 20250507 pftq: changed 1 to 2 for edge case for <=1 sec videos
                num_2x_frames = min(2, max(1, available_frames - effective_clean_frames - 1)) if available_frames > effective_clean_frames + 1 else 0 # 20250507 pftq: subtracted 1 for edge case for <=1 sec videos
                num_4x_frames = min(16, max(1, available_frames - effective_clean_frames - num_2x_frames)) if available_frames > effective_clean_frames + num_2x_frames else 0 # 20250507 pftq: Edge case for <=1 sec
                
                total_context_frames = num_4x_frames + num_2x_frames + effective_clean_frames
                total_context_frames = min(total_context_frames, available_frames)  # 20250507 pftq: Edge case for <=1 sec videos

                indices = torch.arange(0, sum([1, num_4x_frames, num_2x_frames, effective_clean_frames, adjusted_latent_frames])).unsqueeze(0) # 20250507 pftq: latent_window_size to adjusted_latent_frames for edge case for <=1 sec videos
                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split(
                    [1, num_4x_frames, num_2x_frames, effective_clean_frames, adjusted_latent_frames], dim=1 # 20250507 pftq: latent_window_size to adjusted_latent_frames for edge case for <=1 sec videos
                )
                clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
    
                # 20250506 pftq: Split history_latents dynamically based on available frames
                fallback_frame_count = 2 # 20250507 pftq: Changed 0 to 2 Edge case for <=1 sec videos
                context_frames = history_latents[:, :, -total_context_frames:, :, :] if total_context_frames > 0 else history_latents[:, :, :fallback_frame_count, :, :]  
                if total_context_frames > 0:
                    split_sizes = [num_4x_frames, num_2x_frames, effective_clean_frames] 
                    split_sizes = [s for s in split_sizes if s > 0]  # Remove zero sizes
                    if split_sizes:
                        splits = context_frames.split(split_sizes, dim=2)
                        split_idx = 0
                        clean_latents_4x = splits[split_idx] if num_4x_frames > 0 else history_latents[:, :, :fallback_frame_count, :, :]
                        if clean_latents_4x.shape[2] < 2:  # 20250507 pftq: edge case for <=1 sec videos
                            clean_latents_4x = torch.cat([clean_latents_4x, clean_latents_4x[:, :, -1:, :, :]], dim=2)[:, :, :2, :, :]
                        split_idx += 1 if num_4x_frames > 0 else 0
                        clean_latents_2x = splits[split_idx] if num_2x_frames > 0 and split_idx < len(splits) else history_latents[:, :, :fallback_frame_count, :, :]
                        if clean_latents_2x.shape[2] < 2:  # 20250507 pftq: edge case for <=1 sec videos
                            clean_latents_2x = torch.cat([clean_latents_2x, clean_latents_2x[:, :, -1:, :, :]], dim=2)[:, :, :2, :, :]
                        split_idx += 1 if num_2x_frames > 0 else 0
                        clean_latents_1x = splits[split_idx] if effective_clean_frames > 0 and split_idx < len(splits) else history_latents[:, :, :fallback_frame_count, :, :] 
                    else:
                        clean_latents_4x = clean_latents_2x = clean_latents_1x = history_latents[:, :, :fallback_frame_count, :, :] 
                else:
                    clean_latents_4x = clean_latents_2x = clean_latents_1x = history_latents[:, :, :fallback_frame_count, :, :] 
    
                clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

                # 20250507 pftq: Fix for <=1 sec videos.
                max_frames = min(latent_window_size * 4 - 3, history_latents.shape[2] * 4)

                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=max_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
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
    
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
    
                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(vae, target_device=gpu)
    
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
    
                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                  section_latent_frames = latent_window_size * 2
                  overlapped_frames = min(latent_window_size * 4 - 3, history_pixels.shape[2])
                  
                  #if section_index == 0: 
                    #extra_latents = 1  # Add up to 2 extra latent frames for smoother overlap to initial video
                    #extra_pixel_frames = extra_latents * 4  # Approx. 4 pixel frames per latent
                    #overlapped_frames = min(overlapped_frames + extra_pixel_frames, history_pixels.shape[2], section_latent_frames * 4)

                  current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                  history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
    
                if not high_vram:
                    unload_complete_models()
    
                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
    
                # 20250506 pftq: Use input video FPS for output
                save_bcthw_as_mp4(history_pixels, output_filename, fps=fps, crf=mp4_crf)
                print(f"Latest video saved: {output_filename}")
                # 20250508 pftq: Save prompt to mp4 metadata comments
                set_mp4_comments_imageio_ffmpeg(output_filename, f"Prompt: {prompt} | Negative Prompt: {n_prompt}");
                print(f"Prompt saved to mp4 metadata comments: {output_filename}")
    
                # 20250506 pftq: Clean up previous partial files
                if previous_video is not None and os.path.exists(previous_video):
                    try:
                        os.remove(previous_video)
                        print(f"Previous partial video deleted: {previous_video}")
                    except Exception as e:
                        print(f"Error deleting previous partial video {previous_video}: {e}")
                previous_video = output_filename
    
                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
    
                stream.output_queue.push(('file', output_filename))
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return

# 20250506 pftq: Modified process to pass clean frame count, etc from video_encode
def process(input_video, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch):
    global stream, high_vram
    # 20250506 pftq: Updated assertion for video input
    assert input_video is not None, 'No input video!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    # 20250507 pftq: Even the H100 needs offloading if the video dimensions are 720p or higher
    if high_vram and (no_resize or resolution>640):
        print("Disabling high vram mode due to no resize and/or potentially higher resolution...")
        high_vram = False
        vae.enable_slicing()
        vae.enable_tiling()
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)

    # 20250508 pftq: automatically set distilled cfg to 1 if cfg is used
    if cfg > 1:
        gs = 1
    
    stream = AsyncStream()

    # 20250506 pftq: Pass num_clean_frames, vae_batch, etc
    async_run(worker, input_video, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            #yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
            yield output_filename, gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True) # 20250506 pftq: Keep refreshing the video in case it got hidden when the tab was in the background

        if flag == 'end':
            yield output_filename, gr.update(visible=False), desc+' Video complete.', '', gr.update(interactive=True), gr.update(interactive=False)
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
    # 20250506 pftq: Updated title to reflect video input functionality
    gr.Markdown('# Framepack F1 with Video Input (Video Extension)')
    with gr.Row():
        with gr.Column():
            # 20250506 pftq: Changed to Video input from Image
            input_video = gr.Video(sources='upload', label="Input Video", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            #example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            #example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                with gr.Row():
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=False, info='Faster speed, but often makes hands and fingers slightly worse.')
                    no_resize = gr.Checkbox(label='Force Original Video Resolution (No Resizing)', value=False, info='Might run out of VRAM (720p requires > 24GB VRAM).')

                seed = gr.Number(label="Seed", value=31337, precision=0)

                batch = gr.Slider(label="Batch Size (Number of Videos)", minimum=1, maximum=1000, value=1, step=1, info='Generate multiple videos each with a different seed.')

                resolution = gr.Number(label="Resolution (max width or height)", value=640, precision=0, visible=False)

                total_second_length = gr.Slider(label="Additional Video Length to Generate (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                
                # 20250506 pftq: Reduced default distilled guidance scale to improve adherence to input video
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=3.0, step=0.01, info='Prompt adherence at the cost of less details from the input video, but to a lesser extent than Context Frames.')
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=True, info='Use this instead of Distilled for more detail/control + Negative Prompt (make sure Distilled set to 1). Doubles render time.')  # Should not change
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True, info='Requires using normal CFG (undistilled) instead of Distilled (set Distilled=1 and CFG > 1).') 
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Increase for more quality, especially if using high non-distilled CFG.')

                # 20250506 pftq: Renamed slider to Number of Context Frames and updated description
                num_clean_frames = gr.Slider(label="Number of Context Frames", minimum=2, maximum=10, value=5, step=1, info="Retain more video details but increase memory use. Reduce to 2 if memory issues.")
                
                default_vae = 32
                if high_vram:
                    default_vae = 128
                elif free_mem_gb>=20:
                    default_vae = 64
                    
                vae_batch = gr.Slider(label="VAE Batch Size for Input Video", minimum=4, maximum=256, value=default_vae, step=4, info="Reduce if running out of memory. Increase for better quality frames during fast motion.")

                latent_window_size = gr.Slider(label="Latent Window Size", minimum=9, maximum=33, value=9, step=1, visible=True, info='Generate more frames at a time (larger chunks). Less degradation and better blending but higher VRAM cost.') 

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML("""
        <div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>
    """)

    # 20250506 pftq: Updated inputs to include num_clean_frames
    ips = [input_video, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
