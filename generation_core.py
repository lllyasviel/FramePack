import torch
import traceback
import einops
import numpy as np
import os
import threading
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    generate_timestamp
)
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    unload_complete_models,
    load_model_as_complete,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    gpu
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.gradio.progress_bar import make_progress_bar_html


@torch.no_grad()
def worker(
    # --- Task I/O & Identity ---
    task_id,
    input_image,
    output_folder,
    output_queue_ref,
    
    # --- Creative Parameters (The "Recipe") ---
    prompt,
    n_prompt,
    seed,
    total_second_length,
    steps,
    cfg,
    gs,
    gs_final,
    gs_schedule_active,
    rs,
    preview_frequency,
    segments_to_decode_csv,

    # --- Environment & Debug Parameters ---
    latent_window_size,
    gpu_memory_preservation,
    use_teacache,
    use_fp32_transformer_output,
    mp4_crf,
    
    # --- Model & System Objects (Passed from main app) ---
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    vae,
    feature_extractor,
    image_encoder,
    transformer,
    high_vram_flag,
    
    # --- Control Flow ---
    abort_event: threading.Event = None
):
    outputs_folder = os.path.expanduser(output_folder) if output_folder else './outputs/'
    os.makedirs(outputs_folder, exist_ok=True)

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = f"{generate_timestamp()}_task{task_id}"
    output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Starting ...'))))

    parsed_segments_to_decode_set = set()
    if segments_to_decode_csv:
        try:
            parsed_segments_to_decode_set = {int(s.strip()) for s in segments_to_decode_csv.split(',') if s.strip()}
        except ValueError:
            print(f"Task {task_id}: Warning - Could not parse 'Segments to Decode CSV': \"{segments_to_decode_csv}\".")
    final_output_filename = None; success = False
    initial_gs_from_ui = gs
    gs_final_value_for_schedule = gs_final if gs_final is not None else initial_gs_from_ui
    original_fp32_setting = transformer.high_quality_fp32_output_for_inference
    transformer.high_quality_fp32_output_for_inference = use_fp32_transformer_output
    print(f"Task {task_id}: transformer.high_quality_fp32_output_for_inference set to {use_fp32_transformer_output}")

    try:
        if not isinstance(input_image, np.ndarray): raise ValueError(f"Task {task_id}: input_image is not a NumPy array.")

        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Image processing ...'))))
        if input_image.shape[-1] == 4:
            pil_img = Image.fromarray(input_image)
            input_image = np.array(pil_img.convert("RGB"))
        H, W, C = input_image.shape
        if C != 3: raise ValueError(f"Task {task_id}: Input image must be RGB, found {C} channels.")
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        metadata_obj = PngInfo()
        params_to_save_in_metadata = {
            "prompt": prompt,
            "n_prompt": n_prompt,
            "seed": seed,
            "total_second_length": total_second_length,
            "steps": steps,
            "cfg": cfg,
            "gs": gs,
            "gs_final": gs_final,
            "gs_schedule_active": gs_schedule_active,
            "rs": rs,
            "preview_frequency": preview_frequency,
            "segments_to_decode_csv": segments_to_decode_csv
        }
        metadata_obj.add_text("parameters", json.dumps(params_to_save_in_metadata))
        initial_image_with_params_path = os.path.join(outputs_folder, f'{job_id}_initial_image_with_params.png')
        try: Image.fromarray(input_image_np).save(initial_image_with_params_path, pnginfo=metadata_obj)
        except Exception as e_png: print(f"Task {task_id}: WARNING - Failed to save initial image with parameters: {e_png}")

        if not high_vram_flag: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Text encoding ...'))))
        if not high_vram_flag: fake_diffusers_current_device(text_encoder, gpu); load_model_as_complete(text_encoder_2, target_device=gpu)
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1: llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else: llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512); llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        input_image_pt = torch.from_numpy(input_image_np).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1.0; input_image_pt = input_image_pt[:,:,None,:,:]
        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram_flag: load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)
        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram_flag: load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state = [t.to(transformer.dtype) for t in [llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state]]

        output_queue_ref.push(('progress', (task_id, None, f'Total Segments: {total_latent_sections}', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator(device="cpu").manual_seed(int(seed))
        num_frames = latent_window_size * 4 - 3
        #overlapped_frames = num_frames

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device="cpu"); history_pixels = None
        total_generated_latent_frames = 0
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4: latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding_iteration, latent_padding in enumerate(latent_paddings):
            if abort_event and abort_event.is_set(): raise KeyboardInterrupt("Abort signal received.")
            is_last_section = (latent_padding == 0)
            latent_padding_size = latent_padding * latent_window_size
            print(f'Task {task_id}: Seg {latent_padding_iteration + 1}/{total_latent_sections} (lp_val={latent_padding}), last_loop_seg={is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16]), device="cpu").unsqueeze(0)
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latents_pre = start_latent.to(history_latents.device, dtype=history_latents.dtype)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            #current_history_depth_for_clean_split = history_latents.shape[2]; needed_depth_for_clean_split = 1 + 2 + 16
            #history_latents_for_clean_split = history_latents
            #if current_history_depth_for_clean_split < needed_depth_for_clean_split:
            #     padding_needed = needed_depth_for_clean_split - current_history_depth_for_clean_split
            #     pad_tensor = torch.zeros(history_latents.shape[0], history_latents.shape[1], padding_needed, history_latents.shape[3], history_latents.shape[4], dtype=history_latents.dtype, device=history_latents.device)
            #     history_latents_for_clean_split = torch.cat((history_latents, pad_tensor), dim=2)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram_flag: unload_complete_models(); move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)

            def callback_diffusion_step(d):
                if abort_event and abort_event.is_set(): raise KeyboardInterrupt("Abort signal received during sampling.")
                current_diffusion_step = d['i'] + 1
                is_first_step = current_diffusion_step == 1
                is_last_step = current_diffusion_step == steps
                is_preview_step = preview_frequency > 0 and (current_diffusion_step % preview_frequency == 0)
                if not (is_first_step or is_last_step or is_preview_step):
                    return
                preview_latent = d['denoised']; preview_img_np = vae_decode_fake(preview_latent)
                preview_img_np = (preview_img_np * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview_img_np = einops.rearrange(preview_img_np, 'b c t h w -> (b h) (t w) c')
                percentage = int(100.0 * current_diffusion_step / steps)
                hint = f'Segment {latent_padding_iteration + 1}, Sampling {current_diffusion_step}/{steps}'
                current_video_frames_count = history_pixels.shape[2] if history_pixels is not None else 0
                desc = f'Task {task_id}: Vid Frames: {current_video_frames_count}, Len: {current_video_frames_count / 30 :.2f}s. Seg {latent_padding_iteration + 1}/{total_latent_sections}. Extending...'
                output_queue_ref.push(('progress', (task_id, preview_img_np, desc, make_progress_bar_html(percentage, hint))))

            current_segment_gs_to_use = initial_gs_from_ui
            if gs_schedule_active and total_latent_sections > 1:
                progress_for_gs = latent_padding_iteration / (total_latent_sections - 1) if total_latent_sections > 1 else 0
                current_segment_gs_to_use = initial_gs_from_ui + (gs_final_value_for_schedule - initial_gs_from_ui) * progress_for_gs

            generated_latents = sample_hunyuan(
                 transformer=transformer, sampler='unipc', width=width, height=height, frames=num_frames,
                 real_guidance_scale=cfg, distilled_guidance_scale=current_segment_gs_to_use, guidance_rescale=rs,
                 num_inference_steps=steps, generator=rnd, prompt_embeds=llama_vec.to(transformer.device),
                 prompt_embeds_mask=llama_attention_mask.to(transformer.device), prompt_poolers=clip_l_pooler.to(transformer.device),
                 negative_prompt_embeds=llama_vec_n.to(transformer.device), negative_prompt_embeds_mask=llama_attention_mask_n.to(transformer.device),
                 negative_prompt_poolers=clip_l_pooler_n.to(transformer.device), device=transformer.device, dtype=transformer.dtype,
                 image_embeddings=image_encoder_last_hidden_state.to(transformer.device), latent_indices=latent_indices.to(transformer.device),
                 clean_latents=clean_latents.to(transformer.device, dtype=transformer.dtype), clean_latent_indices=clean_latent_indices.to(transformer.device),
                 clean_latents_2x=clean_latents_2x.to(transformer.device, dtype=transformer.dtype), clean_latent_2x_indices=clean_latent_2x_indices.to(transformer.device),
                 clean_latents_4x=clean_latents_4x.to(transformer.device, dtype=transformer.dtype), clean_latent_4x_indices=clean_latent_4x_indices.to(transformer.device),
                 callback=callback_diffusion_step
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram_flag:
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

            if not high_vram_flag: unload_complete_models()

            current_video_frame_count = history_pixels.shape[2]

            # Skip writing preview mp4 for this segment logic
            should_save_mp4_this_iteration = False
            current_segment_1_indexed = latent_padding_iteration # + 1
            if (latent_padding_iteration == 0) or is_last_section or (parsed_segments_to_decode_set and current_segment_1_indexed in parsed_segments_to_decode_set):
                should_save_mp4_this_iteration = True
            if should_save_mp4_this_iteration:
                segment_mp4_filename = os.path.join(outputs_folder, f'{job_id}_segment_{latent_padding_iteration}_frames_{current_video_frame_count}.mp4')
                save_bcthw_as_mp4(history_pixels, segment_mp4_filename, fps=30, crf=mp4_crf)
                final_output_filename = segment_mp4_filename
                print(f"Task {task_id}: SAVED MP4 for segment {latent_padding_iteration} to {segment_mp4_filename}. Total video frames: {current_video_frame_count}")
                output_queue_ref.push(('file', (task_id, segment_mp4_filename, f"Segment {latent_padding_iteration} MP4 saved ({current_video_frame_count} frames)")))
            else:
                print(f"Task {task_id}: SKIPPED MP4 save for intermediate segment {current_segment_1_indexed}.")

            if is_last_section: success = True; break

    except KeyboardInterrupt:
        print(f"Worker task {task_id} caught KeyboardInterrupt (likely abort signal).")
        output_queue_ref.push(('aborted', task_id)); success = False
    except Exception as e:
        print(f"Error in worker task {task_id}: {e}"); traceback.print_exc(); output_queue_ref.push(('error', (task_id, str(e)))); success = False
    finally:
        transformer.high_quality_fp32_output_for_inference = original_fp32_setting
        print(f"Task {task_id}: Restored transformer.high_quality_fp32_output_for_inference to {original_fp32_setting}")
        if not high_vram_flag: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        if final_output_filename and not os.path.dirname(final_output_filename) == os.path.abspath(outputs_folder):
             final_output_filename = os.path.join(outputs_folder, os.path.basename(final_output_filename))
        output_queue_ref.push(('end', (task_id, success, final_output_filename)))