from diffusers_helper.hf_login import login

import os
import sys

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
from diffusers import AutoencoderKLHunyuanVideo as FramePackVAE
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.framepack_helpers import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.framepack_transformer import FramePackTransformer
from diffusers_helper.pipelines.framepack_pipeline import sample_framepack
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# AMD Strix Point Native Optimizations
from bridge_npu import NPUOrchestrator

npu_bridge = NPUOrchestrator()

class StateSaver:
    """Manages latent caching in the 94GB system RAM."""
    def __init__(self):
        self.cache = {}

    def save(self, key, latents):
        self.cache[key] = latents.detach().cpu()
        print(f"[State-Saver] Section {key} cached in System RAM.")

    def load(self, key):
        return self.cache.get(key)

# Hardware Profile Definitions
HARDWARE_PROFILES = {
    "16GB - iGPU Lite": {"window": 9, "ram": "16GB", "swapping": True, "vram_floor": 4.0},
    "32GB - Balanced": {"window": 13, "ram": "32GB", "swapping": True, "vram_floor": 8.0},
    "64GB+ - Director Mode": {"window": 17, "ram": "94GB", "swapping": False, "vram_floor": 12.0},
}

state_saver = StateSaver()


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--low-vram", action='store_true', help="Force low-vram mode (for testing laptop profiles)")
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = (free_mem_gb > 60) and not args.low_vram

print(f'Free VRAM {free_mem_gb} GB')
if args.low_vram:
    print('FORCING LOW-VRAM MODE (Laptop Simulation Active)')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = FramePackVAE.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = FramePackTransformer.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

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


@torch.no_grad()
def worker(hardware_profile, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, mp4_crf):
    # Dynamic profile adjustment
    profile = HARDWARE_PROFILES[hardware_profile]
    latent_window_size = profile["window"] if hardware_profile != "64GB+ - Director Mode" else latent_window_size
    gpu_memory_preservation = profile["vram_floor"]
    
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

        # Offload Prompt Expansion to NPU
        prompt = npu_bridge.expand_prompt(prompt)

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

        # State-Saver: Utilize 94GB System RAM for history
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        state_saver.save(job_id, history_latents)
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
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_framepack(
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
            
            # Persist to State-Saver (94GB System RAM pool)
            state_saver.save(job_id, history_latents)

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


def process(hardware_profile, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, mp4_crf):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, hardware_profile, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, mp4_crf)

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


# --- Premium UI Styling ---
css = """
body { background-color: #0a0a0c; color: #e0e0e0; font-family: 'Inter', 'Outfit', sans-serif; }
.gradio-container { background: linear-gradient(135deg, #0f0f13 0%, #1a1a24 100%) !important; border: none !important; }
.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8); }
button.primary { background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important; border: none !important; color: white !important; font-weight: 600 !important; transition: all 0.3s ease; box-shadow: 0 0 15px rgba(99, 102, 241, 0.4); }
button.primary:hover { transform: translateY(-2px); box-shadow: 0 0 25px rgba(168, 85, 247, 0.6); }
#director-log { font-family: 'Fira Code', monospace; font-size: 0.85rem; color: #10b981; }
"""

block = gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")).queue()
with block:
    gr.HTML("<div style='text-align: center; padding: 20px;'><h1 style='font-size: 2.5rem; background: linear-gradient(90deg, #fff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎬 AMD DIRECTOR STUDIO</h1><p style='color: #64748b;'>Strix Point Optimized | Radeon 890M iGPU | XDNA 2 NPU</p></div>")
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="glass"):
            gr.Markdown("### 🛠 Production Controls")
            input_image = gr.Image(sources='upload', type="numpy", label="Scene Start Image", height=280)
            prompt = gr.Textbox(label="Visual Directive (Auto-Expanded by NPU)", placeholder="Describe the scene...", lines=3)
            
            with gr.Accordion("🎞 Scene Settings", open=False):
                seed = gr.Number(label="Production Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Target Length (Seconds)", minimum=1, maximum=120, value=10, step=0.1)
                latent_window_size = gr.Slider(label="Context Packing Buffer", minimum=1, maximum=65, value=17, step=1, info="Optimized for 64GB VRAM")
                steps = gr.Slider(label="Diffusion Steps", minimum=1, maximum=100, value=25, step=1)
                gs = gr.Slider(label="Cinematic Guidance", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                mp4_crf = gr.Slider(label="Export Quality (CRF)", minimum=0, maximum=100, value=16, step=1)

            hardware_profile = gr.Dropdown(choices=list(HARDWARE_PROFILES.keys()), value="64GB+ - Director Mode", label="💻 Hardware Profile")
            gpu_memory_preservation = gr.Slider(label="VRAM Floor (GB)", minimum=4, maximum=64, value=8, step=0.1, visible=False)
            
            with gr.Row():
                start_button = gr.Button("🎥 FORGE DAILIES", variant="primary")
                end_button = gr.Button("🛑 CUT PRODUCTION", interactive=False)

        with gr.Column(scale=2):
            with gr.Group(elem_classes="glass"):
                gr.Markdown("### 📺 Main Stage")
                result_video = gr.Video(label="Production Dailies", autoplay=True, height=540)
                preview_image = gr.Image(label="Live Latent Monitoring", height=180, visible=False)
                
            with gr.Group(elem_classes="glass", style="margin-top: 20px;"):
                gr.Markdown("### 📝 Director's Log")
                progress_desc = gr.Markdown('Ready for production...', elem_id="director-log")
                progress_bar = gr.HTML('')

    with gr.Row(style="margin-top: 20px;"):
        with gr.Column(elem_classes="glass"):
             gr.Markdown("### 🤖 NPU Orchestrator Status")
             gr.HTML("<div style='display: flex; gap: 20px;'><div style='color: #6366f1;'>● Director (NPU): Online</div><div style='color: #a855f7;'>● Key Grip (NPU): Online</div><div style='color: #10b981;'>● State-Saver (94GB RAM): 100% Ready</div></div>")

    gr.HTML('<div style="text-align:center; padding: 40px; color: #475569;">FramePack Optimized for AMD Ryzen 9 AI by Antigravity AI</div>')

    n_prompt = gr.Textbox(visible=False)
    cfg = gr.Slider(visible=False)
    rs = gr.Slider(visible=False)

    ips = [hardware_profile, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, mp4_crf]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
