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
import time
import json
import base64
import io
import zipfile
import tempfile
import atexit
import shutil
from pathlib import Path
import threading

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
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

queue_lock = threading.Lock()
AUTOSAVE_FILENAME = "framepack_queue.zip"
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)
queue_cache_dir = os.path.join(outputs_folder, "_queue_cache")
os.makedirs(queue_cache_dir, exist_ok=True)
SETTINGS_FILENAME = "settings.json"

param_names = [
    'input_image', 'prompt', 'n_prompt', 'seed', 'total_second_length',
    'latent_window_size', 'steps', 'cfg', 'gs', 'rs',
    'gpu_memory_preservation', 'use_teacache', 'mp4_crf', 'output_folder'
]

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

def patched_video_is_playable(video_filepath):
     return True

gr.processing_utils.video_is_playable = patched_video_is_playable

def save_defaults(*args):
    settings_to_save = dict(zip(param_names[1:], args))
    try:
        with open(SETTINGS_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(settings_to_save, f, indent=4)
        gr.Info(f"Defaults saved to {SETTINGS_FILENAME}")
    except Exception as e:
        gr.Warning(f"Error saving defaults: {e}")
        print(f"Error saving defaults: {e}")
        traceback.print_exc()
    return

def load_defaults():
    default_values = {
        'prompt': '', 'n_prompt': '', 'seed': 31337, 'total_second_length': 5,
        'latent_window_size': 9, 'steps': 25, 'cfg': 1.0, 'gs': 10.0, 'rs': 0.0,
        'gpu_memory_preservation': 6, 'use_teacache': True, 'mp4_crf': 16, 'output_folder': './outputs/'
    }
    loaded_settings = {}
    if os.path.exists(SETTINGS_FILENAME):
        try:
            with open(SETTINGS_FILENAME, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
            print(f"Loaded defaults from {SETTINGS_FILENAME}")
        except Exception as e:
            print(f"Error loading defaults from {SETTINGS_FILENAME}: {e}")
            loaded_settings = {}

    updates = []
    for name in param_names[1:]:
        value = loaded_settings.get(name, default_values.get(name))
        updates.append(gr.update(value=value))

    return updates

def quit_application():
    print("Save and Quit requested...")
    autosave_queue_on_exit(global_state_for_autosave)
    import signal
    os.kill(os.getpid(), signal.SIGINT)

def np_to_base64_uri(np_array_or_tuple, format="png"):
    if np_array_or_tuple is None:
        return None
    try:
        if isinstance(np_array_or_tuple, tuple) and len(np_array_or_tuple) > 0 and isinstance(np_array_or_tuple[0], np.ndarray):
            np_array = np_array_or_tuple[0]
        elif isinstance(np_array_or_tuple, np.ndarray):
             np_array = np_array_or_tuple
        else:
             print(f"Warning: Unexpected type in np_to_base64_uri: {type(np_array_or_tuple)}")
             return None

        pil_image = Image.fromarray(np_array.astype(np.uint8))
        output_format = format.lower()
        if output_format == "jpeg" and pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format=output_format)
        img_bytes = buffer.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{output_format};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting NumPy array/tuple to base64: {e}")
        return None

def get_queue_state(state_dict):
    if state_dict is None: state_dict = {}
    if "queue_state" not in state_dict:
        state_dict["queue_state"] = {"queue": [], "next_id": 1, "processing": False, "abort_current": False}
    return state_dict["queue_state"]

def update_queue_df(queue_state):
    queue = queue_state.get("queue", [])
    data = []
    processing = queue_state.get("processing", False)
    editing_task_id = queue_state.get("editing_task_id", None)

    for i, task in enumerate(queue):
        params = task['params']
        task_id = task['id']
        prompt_display = (params['prompt'][:77] + '...') if len(params['prompt']) > 80 else params['prompt']
        prompt_title = params['prompt'].replace('"', '"')
        prompt_cell = f'<span title="{prompt_title}">{prompt_display}</span>'

        img_uri = np_to_base64_uri(params['input_image'], format="png")
        thumbnail_size = "50px"
        img_md = ""
        if img_uri:
            img_md = f'<img src="{img_uri}" alt="Input" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" />'

        is_processing = processing and i == 0
        is_editing = editing_task_id == task_id

        up_btn = "↑"
        down_btn = "↓"
        remove_btn = "✖"
        edit_btn = "✎"

        task_status = task.get("status")
        status = "<?>"
        if is_processing:
            status = "⏳ Processing"
        elif is_editing:
            status = "✏️ Editing"
        elif task_status == "done":
            status = "✅ Done"
        elif task_status == "error":
            status = "❌ Error"
        elif task_status == "aborted":
            status = "⏹️ Aborted"
        elif task_status == "pending":
            status = "⏸️ Pending"

        data.append([
            task_id,
            status,
            prompt_cell,
            f"{params['total_second_length']:.1f}s",
            params['steps'],
            img_md,
            up_btn,
            down_btn,
            remove_btn,
            edit_btn
        ])
    return gr.DataFrame(value=data, visible=len(data) > 0)

def add_task_to_queue(state, *args):
    inputs = list(args)
    input_image_np = inputs[0]
    prompt = inputs[1]

    if input_image_np is None:
        gr.Warning("Input image is required!")
        return state, update_queue_df(get_queue_state(state))

    queue_state = get_queue_state(state)
    queue = queue_state["queue"]
    next_id = queue_state["next_id"]

    params_dict = dict(zip(param_names, inputs))

    task = {
        "id": next_id,
        "params": params_dict,
        "status": "pending"
    }

    with queue_lock:
        queue.append(task)
        queue_state["next_id"] += 1

    gr.Info(f"Task {next_id} added to queue.")
    return state, update_queue_df(queue_state)

def move_task(state, direction, selected_indices):
    if not selected_indices:
        return state, update_queue_df(get_queue_state(state))

    idx = selected_indices[0]
    if isinstance(idx, list): idx = idx[0]
    idx = int(idx)

    queue_state = get_queue_state(state)
    queue = queue_state["queue"]

    with queue_lock:
        if direction == 'up' and idx > 0:
            queue[idx], queue[idx-1] = queue[idx-1], queue[idx]
        elif direction == 'down' and idx < len(queue) - 1:
            queue[idx], queue[idx+1] = queue[idx+1], queue[idx]

    return state, update_queue_df(queue_state)

def remove_task(state, selected_indices):
    removed_task_id = None
    if not selected_indices:
         return state, update_queue_df(get_queue_state(state)), removed_task_id

    idx = selected_indices[0]
    if isinstance(idx, list): idx = idx[0]
    idx = int(idx)

    queue_state = get_queue_state(state)
    queue = queue_state["queue"]

    with queue_lock:
        if 0 <= idx < len(queue):
            removed_task = queue.pop(idx)
            removed_task_id = removed_task['id']
            gr.Info(f"Removed task {removed_task_id} (Prompt: {removed_task['params']['prompt'][:30]}...).")
        else:
            gr.Warning("Invalid index selected for removal.")

    return state, update_queue_df(queue_state), removed_task_id

def handle_queue_action(state, evt: gr.SelectData, *ips):
    (input_image_ui, prompt_ui, n_prompt_ui, seed_ui, total_second_length_ui,
     latent_window_size_ui, steps_ui, cfg_ui, gs_ui, rs_ui,
     gpu_memory_preservation_ui, use_teacache_ui, mp4_crf_ui, output_folder_ui) = ips

    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
         return [state, update_queue_df(get_queue_state(state))] + [gr.update()] * (len(ips) + 3)

    row_index, col_index = evt.index
    button_clicked = evt.value
    queue_state = get_queue_state(state)
    queue = queue_state["queue"]
    processing = queue_state.get("processing", False)

    outputs = [state, update_queue_df(queue_state)] + [gr.update()] * (len(ips) + 3)

    if button_clicked == "↑":
        if processing and row_index == 0:
             gr.Warning("Cannot move the currently processing task.")
             return outputs
        new_state, new_df = move_task(state, 'up', [[row_index, col_index]])
        outputs[0] = new_state
        outputs[1] = new_df
        return outputs
    elif button_clicked == "↓":
        if processing and row_index == 0:
             gr.Warning("Cannot move the currently processing task.")
             return outputs
        if processing and row_index == 1:
             gr.Warning("Cannot move a task below the currently processing task.")
             return outputs
        new_state, new_df = move_task(state, 'down', [[row_index, col_index]])
        outputs[0] = new_state
        outputs[1] = new_df
        return outputs
    elif button_clicked == "✖":
        if processing and row_index == 0:
             gr.Warning("Cannot remove the currently processing task.")
             return outputs
        new_state, new_df, removed_task_id = remove_task(state, [[row_index, col_index]])
        outputs[0] = new_state
        outputs[1] = new_df
        if removed_task_id is not None and queue_state.get("editing_task_id", None) == removed_task_id:
             gr.Info(f"Edit mode cancelled as Task {removed_task_id} was removed.")
             queue_state["editing_task_id"] = None
             outputs[2 + len(ips)] = gr.update(value="Add Task to Queue")
             outputs[2 + len(ips) + 1] = gr.update(visible=False)
        return outputs
    elif button_clicked == "✎":
        if processing and row_index == 0:
            gr.Warning("Cannot edit the currently processing task.")
            return outputs

        if 0 <= row_index < len(queue):
             task_to_edit = queue[row_index]
             task_id_to_edit = task_to_edit['id']
             params_to_edit = task_to_edit['params']

             queue_state["editing_task_id"] = task_id_to_edit
             gr.Info(f"Editing Task {task_id_to_edit}. Make changes and click 'Update Task'.")

             ui_updates = []
             for i, name in enumerate(param_names):
                 value_to_set = params_to_edit.get(name)
                 if i == 0:
                      if isinstance(value_to_set, np.ndarray):
                           gallery_value = [(value_to_set, None)]
                           ui_updates.append(gr.update(value=gallery_value))
                      else:
                           ui_updates.append(gr.update(value=None))
                 else:
                      ui_updates.append(gr.update(value=value_to_set))

             return ([state, update_queue_df(queue_state)] +
                     ui_updates +
                     [gr.update(value="Update Task"), gr.update(visible=True), gr.update()])
        else:
             gr.Warning("Invalid index for edit.")
             return outputs

    return outputs

def add_or_update_task(state, *args):
    queue_state = get_queue_state(state)
    editing_task_id = queue_state.get("editing_task_id", None)

    inputs = list(args)
    input_images_gallery_output = inputs[0]
    prompt = inputs[1]
    output_folder = inputs[-1]

    if not input_images_gallery_output:
        gr.Warning("Input image(s) are required!")
        return state, update_queue_df(queue_state), gr.update(value="Add Task to Queue" if editing_task_id is None else "Update Task"), gr.update(visible=editing_task_id is not None)

    if not isinstance(input_images_gallery_output, list):
         input_images_gallery_output = [input_images_gallery_output]

    tasks_added_count = 0
    first_new_task_id = -1

    base_params_dict = dict(zip(param_names[1:], inputs[1:]))

    if editing_task_id is not None:
        if len(input_images_gallery_output) > 1:
            gr.Warning("Cannot update a task with multiple images from Gallery. Please cancel edit and add as new tasks.")
            return state, update_queue_df(queue_state), gr.update(value="Update Task"), gr.update(visible=True)

        img_tuple = input_images_gallery_output[0]
        if not isinstance(img_tuple, tuple) or not isinstance(img_tuple[0], np.ndarray):
             gr.Warning("Invalid image format received during update.")
             return state, update_queue_df(queue_state), gr.update(value="Update Task"), gr.update(visible=True)
        img_np_for_update = img_tuple[0]

        with queue_lock:
            task_found = False
            for task in queue_state["queue"]:
                if task["id"] == editing_task_id:
                    task["params"] = {'input_image': img_np_for_update, **base_params_dict}
                    task["status"] = "pending"
                    task_found = True
                    break
            if not task_found:
                gr.Warning(f"Task {editing_task_id} not found for update. Edit cancelled.")
            else:
                gr.Info(f"Task {editing_task_id} updated.")
            queue_state["editing_task_id"] = None
            return state, update_queue_df(queue_state), gr.update(value="Add Task to Queue"), gr.update(visible=False)
    else:
        with queue_lock:
            for img_tuple in input_images_gallery_output:
                if not isinstance(img_tuple, tuple) or not isinstance(img_tuple[0], np.ndarray):
                    gr.Warning("One of the provided image inputs was invalid. Skipping.")
                    continue
                img_np = img_tuple[0]

                next_id = queue_state["next_id"]
                if first_new_task_id == -1: first_new_task_id = next_id

                task = {
                    "id": next_id,
                    "params": {'input_image': img_np, **base_params_dict},
                    "status": "pending"
                }
                queue_state["queue"].append(task)
                queue_state["next_id"] += 1
                tasks_added_count += 1

        if tasks_added_count > 0:
             gr.Info(f"Added {tasks_added_count} task(s) to queue (starting ID: {first_new_task_id}).")
        else:
             gr.Warning("No valid tasks were added.")
        return state, update_queue_df(queue_state), gr.update(value="Add Task to Queue"), gr.update(visible=False)

def cancel_edit_mode(state):
    queue_state = get_queue_state(state)
    if queue_state.get("editing_task_id") is not None:
        gr.Info("Edit cancelled.")
        queue_state["editing_task_id"] = None
    return state, update_queue_df(queue_state), gr.update(value="Add Task to Queue"), gr.update(visible=False)

def clear_queue(state):
    queue_state = get_queue_state(state)
    queue = queue_state["queue"]
    processing = queue_state["processing"]
    cleared_count = 0
    with queue_lock:
        if processing:
             if len(queue) > 1:
                 cleared_count = len(queue) - 1
                 queue_state["queue"] = [queue[0]]
                 gr.Info(f"Cleared {cleared_count} pending tasks. Current task continues.")
             else:
                 gr.Info("Queue only contains the currently processing task. Nothing cleared.")
        elif queue:
             cleared_count = len(queue)
             queue.clear()
             gr.Info(f"Cleared {cleared_count} tasks from the queue.")
        else:
             gr.Info("Queue is already empty.")
    if not processing and cleared_count > 0:
         try:
             if os.path.isfile(AUTOSAVE_FILENAME):
                 os.remove(AUTOSAVE_FILENAME)
                 print(f"Clear Queue: Deleted autosave file '{AUTOSAVE_FILENAME}'.")
         except OSError as e:
             print(f"Clear Queue: Error deleting autosave file '{AUTOSAVE_FILENAME}': {e}")

    return state, update_queue_df(queue_state)

def save_queue(state):
    queue_state = get_queue_state(state)
    queue = queue_state.get("queue", [])

    if not queue:
        gr.Info("Queue is empty. Nothing to save.")
        return state, ""

    zip_buffer = io.BytesIO()
    saved_files_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_manifest = []
        image_paths_in_zip = {}

        for task_index, task in enumerate(queue):
            params_copy = task['params'].copy()
            task_id_s = task['id']
            input_image_np = params_copy.pop('input_image', None)

            manifest_entry = {
                "id": task['id'],
                "params": params_copy
            }

            if input_image_np is not None:
                img_hash = hash(input_image_np.tobytes())
                if img_hash in image_paths_in_zip:
                     manifest_entry['image_ref'] = image_paths_in_zip[img_hash]
                else:
                     img_filename_in_zip = f"task_{task_id_s}_input.png"
                     img_save_path = os.path.join(tmpdir, img_filename_in_zip)
                     try:
                         Image.fromarray(input_image_np).save(img_save_path, "PNG")
                         manifest_entry['image_ref'] = img_filename_in_zip
                         image_paths_in_zip[img_hash] = img_filename_in_zip
                         saved_files_count += 1
                     except Exception as e:
                         print(f"Error saving image for task {task_id_s}: {e}")
                         manifest_entry['image_ref'] = None

            queue_manifest.append(manifest_entry)

        manifest_path = os.path.join(tmpdir, "queue_manifest.json")
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(queue_manifest, f, indent=4)
        except Exception as e:
            print(f"Error writing queue manifest: {e}")
            gr.Warning("Failed to create queue manifest.")
            return state, ""

        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="queue_manifest.json")

                for img_filename_rel in image_paths_in_zip.values():
                    img_abs_path = os.path.join(tmpdir, img_filename_rel)
                    if os.path.exists(img_abs_path):
                        zf.write(img_abs_path, arcname=img_filename_rel)
                    else:
                        print(f"Warning: Image file {img_filename_rel} not found during zipping.")

            zip_buffer.seek(0)
            zip_binary_content = zip_buffer.getvalue()
            zip_base64 = base64.b64encode(zip_binary_content).decode('utf-8')
            gr.Info(f"Queue with {len(queue)} tasks ({saved_files_count} images) saved.")
            return state, zip_base64

        except Exception as e:
            print(f"Error creating zip file: {e}")
            gr.Warning("Failed to create zip data for download.")
            return state, ""
        finally:
            zip_buffer.close()

def load_queue(state, filepath):
    if not filepath or not hasattr(filepath, 'name') or not Path(filepath.name).is_file():
        gr.Warning("No valid file selected or file not found.")
        return state, update_queue_df(get_queue_state(state))

    queue_state = get_queue_state(state)
    newly_loaded_queue = []
    max_id_in_file = 0
    loaded_files_count = 0
    error_message = ""

    try:
        os.makedirs(queue_cache_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(filepath.name, 'r') as zf:
                if "queue_manifest.json" not in zf.namelist():
                     raise ValueError("queue_manifest.json not found in zip file")
                zf.extractall(tmpdir)

            manifest_path = os.path.join(tmpdir, "queue_manifest.json")
            with open(manifest_path, 'r', encoding='utf-8') as f:
                loaded_manifest = json.load(f)

            for task_data in loaded_manifest:
                params = task_data.get('params', {})
                task_id_loaded = task_data.get('id', 0)
                max_id_in_file = max(max_id_in_file, task_id_loaded)
                image_ref = task_data.get('image_ref')
                input_image_np = None

                if image_ref:
                    img_load_path = os.path.join(tmpdir, image_ref)
                    if os.path.exists(img_load_path):
                        try:
                            persistent_img_path = os.path.join(queue_cache_dir, image_ref)
                            shutil.copy2(img_load_path, persistent_img_path)

                            with Image.open(persistent_img_path) as img:
                                input_image_np = np.array(img)
                            loaded_files_count += 1
                        except Exception as img_e:
                            print(f"Error loading/copying image {image_ref}: {img_e}")
                            error_message += f"Failed to load image for task {task_id_loaded}. "
                    else:
                        print(f"Image file not found in extracted data: {img_load_path}")
                        error_message += f"Missing image file for task {task_id_loaded}. "

                runtime_task = {
                    "id": task_id_loaded,
                    "params": {**params, 'input_image': input_image_np},
                    "status": "pending"
                }
                newly_loaded_queue.append(runtime_task)

        with queue_lock:
            queue_state["queue"] = newly_loaded_queue
            queue_state["next_id"] = max(max_id_in_file + 1, queue_state.get("next_id", 1))

        gr.Info(f"Successfully loaded {len(newly_loaded_queue)} tasks ({loaded_files_count} images) from queue.")
        if error_message:
            gr.Warning(error_message)

        return state, update_queue_df(queue_state)

    except (ValueError, zipfile.BadZipFile, FileNotFoundError, Exception) as e:
        error_message = f"Error during queue load: {e}"
        print(f"[load_queue] Error: {error_message}")
        traceback.print_exc()
        gr.Warning(f"Failed to load queue: {error_message[:200]}")
        return state, update_queue_df(queue_state)
    finally:
         if filepath and hasattr(filepath, 'name') and filepath.name and os.path.exists(filepath.name):
             if tempfile.gettempdir() in os.path.abspath(filepath.name):
                 try:
                     os.remove(filepath.name)
                 except OSError as e:
                     print(f"[load_queue] Info: Could not remove temp file {filepath.name}: {e}")

def autosave_queue_on_exit(state):
    print("Attempting to autosave queue on exit...")
    queue_state = get_queue_state(state.copy())
    queue = queue_state.get("queue", [])

    if not queue:
        print("Autosave: Queue is empty, nothing to save.")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_manifest = []
            image_paths_in_zip = {}

            for task in queue:
                 params_copy = task['params'].copy()
                 task_id_s = task['id']
                 input_image_np = params_copy.pop('input_image', None)
                 manifest_entry = { "id": task['id'], "params": params_copy}

                 if input_image_np is not None:
                     img_hash = hash(input_image_np.tobytes())
                     if img_hash in image_paths_in_zip:
                         manifest_entry['image_ref'] = image_paths_in_zip[img_hash]
                     else:
                         img_filename_in_zip = f"task_{task_id_s}_input.png"
                         img_save_path = os.path.join(tmpdir, img_filename_in_zip)
                         try:
                             Image.fromarray(input_image_np).save(img_save_path, "PNG")
                             manifest_entry['image_ref'] = img_filename_in_zip
                             image_paths_in_zip[img_hash] = img_filename_in_zip
                         except Exception as e:
                             print(f"Autosave error saving image for task {task_id_s}: {e}")
                             manifest_entry['image_ref'] = None
                 queue_manifest.append(manifest_entry)

            manifest_path = os.path.join(tmpdir, "queue_manifest.json")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(queue_manifest, f, indent=4)

            with zipfile.ZipFile(AUTOSAVE_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="queue_manifest.json")
                for img_filename_rel in image_paths_in_zip.values():
                    img_abs_path = os.path.join(tmpdir, img_filename_rel)
                    if os.path.exists(img_abs_path):
                        zf.write(img_abs_path, arcname=img_filename_rel)

            print(f"Autosave successful: Saved {len(queue)} tasks to {AUTOSAVE_FILENAME}")

    except Exception as e:
        print(f"Error during autosave: {e}")
        traceback.print_exc()

def autoload_queue_on_start(state):
    queue_state = get_queue_state(state)
    df_update = update_queue_df(queue_state)

    if not queue_state["queue"] and Path(AUTOSAVE_FILENAME).is_file():
        print(f"Autoloading queue from {AUTOSAVE_FILENAME}...")
        class MockFile:
            def __init__(self, name): self.name = name
        mock_filepath = MockFile(AUTOSAVE_FILENAME)

        temp_state_copy = {"queue_state": queue_state.copy()}
        loaded_state, df_update = load_queue(temp_state_copy, mock_filepath)

        if loaded_state["queue_state"]["queue"]:
            queue_state.update(loaded_state["queue_state"])
            print(f"Autoload successful. Loaded {len(queue_state['queue'])} tasks.")
            try:
                os.remove(AUTOSAVE_FILENAME)
                print(f"Removed autosave file: {AUTOSAVE_FILENAME}")
            except OSError as e:
                print(f"Error removing autosave file '{AUTOSAVE_FILENAME}': {e}")
        else:
            print("Autoload attempted but queue remains empty (file might be invalid or empty).")
            queue_state["queue"] = []
            queue_state["next_id"] = 1
            df_update = update_queue_df(queue_state)
    return state, df_update

@torch.no_grad()
def worker(task_id, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, output_folder, output_queue_ref):

    current_output_folder = output_folder if output_folder else './outputs/'
    os.makedirs(current_output_folder, exist_ok=True)

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = f"{generate_timestamp()}_task{task_id}"
    output_queue_ref.push(('progress', (task_id, None, '', make_progress_bar_html(0, 'Starting ...'))))

    final_output_filename = None
    success = False

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        output_queue_ref.push(('progress', (task_id, None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        output_queue_ref.push(('progress', (task_id, None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        output_queue_ref.push(('progress', (task_id, None, '', make_progress_bar_html(0, 'Image processing ...'))))

        if input_image.shape[-1] == 4:
            print(f"Task {task_id}: Converting input image from RGBA to RGB.")
            input_image = Image.fromarray(input_image).convert("RGB")
            input_image = np.array(input_image)

        H, W, C = input_image.shape
        if C != 3:
             raise ValueError(f"Task {task_id}: Input image must have 3 channels (RGB), but found {C} after potential conversion.")

        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        output_queue_ref.push(('progress', (task_id, None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        output_queue_ref.push(('progress', (task_id, None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

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
        output_queue_ref.push(('progress', (task_id, None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            # Check for external abort signal (from queue_state)
            # This requires worker to have access to queue_state or a signal mechanism
            # Simplification: Rely on KeyboardInterrupt check inside callback for now
            # if queue_state["abort_current"]: # Needs access to queue_state
            #     output_queue_ref.push(('aborted', task_id))
            #     return

            print(f'Task {task_id}: latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

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

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Task {task_id}: Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). Extending...'
                output_queue_ref.push(('progress', (task_id, preview, desc, make_progress_bar_html(percentage, hint))))
                return

            try:
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
            except KeyboardInterrupt:
                 print(f"Task {task_id} received interrupt signal.")
                 output_queue_ref.push(('aborted', task_id))
                 return

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

            output_filename = os.path.join(current_output_folder, f'{job_id}_progress_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            final_output_filename = output_filename

            print(f'Task {task_id}: Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            # output_queue_ref.push(('file', (task_id, output_filename)))

            if is_last_section:
                success = True
                break

    except Exception as e:
        print(f"Error in worker for task {task_id}:")
        traceback.print_exc()
        output_queue_ref.push(('error', (task_id, str(e)))) # Send error message
        success = False

    finally:
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        if final_output_filename and not os.path.dirname(final_output_filename) == os.path.abspath(current_output_folder):
             final_output_filename = os.path.join(current_output_folder, os.path.basename(final_output_filename))
        output_queue_ref.push(('end', (task_id, success, final_output_filename)))


def process_queue(state, progress=gr.Progress(track_tqdm=True)):
    queue_state = get_queue_state(state)

    if queue_state["processing"]:
        yield state, update_queue_df(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=True) # Keep add active, abort maybe active
        return

    if not queue_state["queue"]:
        yield state, update_queue_df(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False) # Add active, abort inactive
        return

    queue_state["processing"] = True
    queue_state["abort_current"] = False

    yield (
        state,
        update_queue_df(queue_state),
        gr.update(),
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        gr.update(interactive=True),
        gr.update(interactive=True)
    )

    stream = AsyncStream()
    global_abort_signal = False

    while queue_state["queue"] and not global_abort_signal:
        current_task = None
        task_params = None
        task_id = -1

        with queue_lock:
            if queue_state["queue"]:
                current_task = queue_state["queue"][0]
                task_params = current_task["params"]
                task_id = current_task["id"]
            else:
                break

        print(f"Starting task {task_id}...")
        current_task["status"] = "processing"

        yield (
             state, update_queue_df(queue_state), gr.update(), gr.update(), gr.update(), gr.update(),
             gr.update(interactive=True),
             gr.update(interactive=True)
        )

        async_run(worker, task_id, **task_params, output_queue_ref=stream.output_queue)

        output_filename = None
        task_successful = False
        task_aborted_by_user = False

        while True:
            flag, data = stream.output_queue.next()

            if flag == 'progress':
                msg_task_id, preview, desc, html = data
                if msg_task_id == task_id:
                     yield (
                         state, gr.update(), gr.update(value=output_filename), gr.update(visible=preview is not None, value=preview),
                         desc, html,
                         gr.update(interactive=True),
                         gr.update(interactive=True)
                     )
            elif flag == 'aborted':
                 msg_task_id = data
                 if msg_task_id == task_id:
                     print(f"Task {task_id} was aborted by worker signal.")
                     task_aborted_by_user = True
                     current_task["status"] = "aborted"
                     break

            elif flag == 'error':
                 msg_task_id, error_msg = data
                 if msg_task_id == task_id:
                     print(f"Task {task_id} failed: {error_msg}")
                     gr.Warning(f"Task {task_id} failed: {error_msg}")
                     current_task["status"] = "error"
                     break

            elif flag == 'end':
                 msg_task_id, success, final_output_filename = data
                 if msg_task_id == task_id:
                     task_successful = success
                     output_filename = final_output_filename
                     current_task["status"] = "done" if success else "error"
                     print(f"Task {task_id} finished. Success: {success}. Output: {output_filename}")
                     break

            if queue_state["abort_current"]:
                 print("Global abort signal received. Stopping worker if running...")
                 global_abort_signal = True
                 task_aborted_by_user = True
                 current_task["status"] = "aborted"
        with queue_lock:
            if queue_state["queue"] and queue_state["queue"][0]["id"] == task_id:
                queue_state["queue"].pop(0)
            else:
                print(f"Warning: Task {task_id} not found at head of queue after finishing.")

        yield (
            state, update_queue_df(queue_state), gr.update(value=output_filename), gr.update(visible=False),
            gr.update(value=f"Task {task_id} finished."), gr.update(value=""),
            gr.update(interactive=True),
            gr.update(interactive=True)
        )

        if global_abort_signal:
            gr.Info(f"Processing halted after task {task_id} due to abort signal.")
            break

    queue_state["processing"] = False
    queue_state["abort_current"] = False
    print("Queue processing finished.")

    yield (
        state,
        update_queue_df(queue_state),
        gr.update(),
        gr.update(visible=False),
        gr.update(value="Queue processing complete."),
        gr.update(value=""),
        gr.update(interactive=True),
        gr.update(interactive=False)
    )


def abort_processing(state):
    queue_state = get_queue_state(state)
    aborted = False
    if queue_state["processing"]:
        gr.Info("Abort signal sent. Current task will attempt to stop. Processing will halt after the current task finishes or aborts.")
        queue_state["abort_current"] = True
        aborted = True
    else:
        gr.Info("Nothing is currently processing.")
    return state, gr.update(interactive=not aborted)


css = make_progress_bar_css() + """
#queue_df th:nth-child(1), #queue_df td:nth-child(1) { width: 5%; }
#queue_df th:nth-child(2), #queue_df td:nth-child(2) { width: 10%; }
#queue_df th:nth-child(3), #queue_df td:nth-child(3) { width: 40%; }
#queue_df th:nth-child(4), #queue_df td:nth-child(4) { width: 8%; }
#queue_df th:nth-child(5), #queue_df td:nth-child(5) { width: 8%; }
#queue_df th:nth-child(6), #queue_df td:nth-child(6) { width: 10%; }
#queue_df th:nth-child(7), #queue_df td:nth-child(7) { width: 4%; cursor: pointer; }
#queue_df th:nth-child(8), #queue_df td:nth-child(8) { width: 4%; cursor: pointer; }
#queue_df th:nth-child(9), #queue_df td:nth-child(9) { width: 4%; cursor: pointer; }
#queue_df th:nth-child(10), #queue_df td:nth-child(10) { width: 4%; cursor: pointer; }
#queue_df td:nth-child(7):hover,
#queue_df td:nth-child(8):hover,
#queue_df td:nth-child(9):hover,
#queue_df td:nth-child(10):hover { background-color: #e0e0e0; }
"""
block = gr.Blocks(css=css).queue()

global_state_for_autosave = {}
atexit.register(autosave_queue_on_exit, global_state_for_autosave)

with block:
    state = gr.State({})

    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Gallery(type="numpy", label="Image(s)", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=[[x] for x in [
                'The girl dances gracefully, with clear movements, full of charm.',
                'A character doing some simple body movements.',
            ]], label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False) # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False) # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False) # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False) # Should not change
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                output_folder_ui = gr.Textbox(label="Output Folder", value="./outputs/", info="Folder where videos will be saved.")
                save_defaults_btn = gr.Button(value="Save Defaults", variant="secondary")

            with gr.Row():
                add_button = gr.Button(value="Add Task to Queue")
                cancel_edit_button = gr.Button(value="Cancel Edit", visible=False, variant="secondary")
                abort_button = gr.Button(value="Abort Current Task", interactive=False, variant="stop")

        with gr.Column(scale=2):
            gr.Markdown("## Queue")
            queue_df = gr.DataFrame(
                 headers=["ID", "Status", "Prompt", "Length", "Steps", "Input", "", "", "", ""],
                 datatype=["number", "str", "markdown", "str", "number", "markdown", "str", "str", "str", "str"],
                 col_count=(10, "fixed"),
                 value=[], interactive=False,
                 visible=False,
                 elem_id="queue_df"
            )

            with gr.Row():
                save_queue_zip_output = gr.Text(visible=False)
                save_queue_btn = gr.DownloadButton("Save Queue", size="sm")
                load_queue_btn = gr.UploadButton("Load Queue", file_types=[".zip"], size="sm")
                clear_queue_btn = gr.Button("Clear Queue", size="sm", variant="stop")
                quit_button = gr.Button("Save and Quit", size="sm", variant="secondary")

            gr.Markdown("## Current Task Progress")
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

            gr.Markdown("## Last Finished Video")
            result_video = gr.Video(label="Finished Video", autoplay=True, show_share_button=False, height=400, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')


    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, output_folder_ui]
    default_settings_components = ips[1:]

    add_button.click(
        fn=add_or_update_task,
        inputs=[state] + ips,
        outputs=[state, queue_df, add_button, cancel_edit_button]
    ).then(
        fn=process_queue,
        inputs=[state],
        outputs=[state, queue_df, result_video, preview_image, progress_desc, progress_bar, add_button, abort_button]
    )

    cancel_edit_button.click(
        fn=cancel_edit_mode,
        inputs=[state],
        outputs=[state, queue_df, add_button, cancel_edit_button]
    )

    quit_button.click(
         fn=quit_application,
         inputs=[],
         outputs=[]
    )

    abort_button.click(fn=abort_processing, inputs=[state], outputs=[state, abort_button])

    clear_queue_btn.click(fn=clear_queue, inputs=[state], outputs=[state, queue_df])

    save_queue_btn.click(
         fn=save_queue,
         inputs=[state],
         outputs=[state, save_queue_zip_output]
     ).then(
         fn=None,
         inputs=[save_queue_zip_output],
         outputs=None,
         js="""
            (base64String) => {
              if (!base64String) { return; }
              try {
                const byteCharacters = atob(base64String);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) { byteNumbers[i] = byteCharacters.charCodeAt(i); }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], { type: 'application/zip' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none'; a.href = url; a.download = 'framepack_queue.zip';
                document.body.appendChild(a); a.click();
                window.URL.revokeObjectURL(url); document.body.removeChild(a);
              } catch (e) { console.error("Error triggering download:", e); }
            }
            """
     )

    load_queue_btn.upload(fn=load_queue, inputs=[state, load_queue_btn], outputs=[state, queue_df])

    save_defaults_btn.click(
        fn=save_defaults,
        inputs=default_settings_components,
        outputs=[]
    )

    queue_df.select(
        fn=handle_queue_action,
        inputs=[state] + ips,
        outputs=[state, queue_df] + ips + [add_button, cancel_edit_button, result_video]
    )

    block.load(
        fn=load_defaults,
        inputs=[],
        outputs=default_settings_components
    ).then(
        fn=autoload_queue_on_start,
        inputs=[state],
        outputs=[state, queue_df]
    ).then(
        lambda s: global_state_for_autosave.update(s),
        inputs=[state], outputs=[]
    ).then(
        fn=process_queue,
        inputs=[state],
        outputs=[state, queue_df, result_video, preview_image, progress_desc, progress_bar, add_button, abort_button]
    )

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)