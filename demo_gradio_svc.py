# Stub temporary callout for many thanks to @Tophness PR-150 for great work on the queueing system integrated into this code.

from diffusers_helper.hf_login import login

import os
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
from gradio_modal import Modal
import torch
import traceback
import numpy as np
import argparse
import time
import json
import base64
import io
import zipfile
import tempfile
import atexit
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css

from generation_core import worker

# --- Globals and Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true', default=False, help="Enable Gradio sharing link.")
parser.add_argument("--server", type=str, default='127.0.0.1', help="Server name to bind to.")
parser.add_argument("--port", type=int, required=False, help="Port to run the server on.")
parser.add_argument("--inbrowser", action='store_true', default=False, help="Launch in browser automatically.")
args = parser.parse_args()

abort_event = threading.Event()
queue_lock = threading.Lock()
outputs_folder = './outputs_svc/'
os.makedirs(outputs_folder, exist_ok=True)

SETTINGS_FILENAME = "framepack_svc_settings.json" # The file for default workspace settings.
AUTOSAVE_FILENAME = "framepack_svc_queue.zip"

# Creative "Recipe" Parameters (for portable PNG metadata and task editing)
CREATIVE_PARAM_KEYS = [
    'prompt', 'n_prompt', 'total_second_length', 'seed', 'preview_frequency_ui',
    'segments_to_decode_csv', 'gs_ui', 'gs_schedule_shape_ui', 'gs_final_ui', 'steps', 'cfg', 'rs'
]

# Environment/Debug Parameters (for the full workspace, machine/session-specific)
ENVIRONMENT_PARAM_KEYS = [
    'use_teacache', 'use_fp32_transformer_output_ui', 'gpu_memory_preservation',
    'mp4_crf', 'output_folder_ui', 'latent_window_size'
]

# A comprehensive list of all UI components that define a task.
ALL_TASK_UI_KEYS = CREATIVE_PARAM_KEYS + ENVIRONMENT_PARAM_KEYS

# This maps UI key names to the names expected by the 'worker' function.
# It acts as a bridge between the Gradio UI and the backend processing logic.
UI_TO_WORKER_PARAM_MAP = {
    'prompt': 'prompt', 'n_prompt': 'n_prompt', 'total_second_length': 'total_second_length',
    'seed': 'seed', 'use_teacache': 'use_teacache', 'preview_frequency_ui': 'preview_frequency',
    'segments_to_decode_csv': 'segments_to_decode_csv', 'gs_ui': 'gs',
    'gs_schedule_shape_ui': 'gs_schedule_active', 'gs_final_ui': 'gs_final', 'steps': 'steps', 'cfg': 'cfg',
    'latent_window_size': 'latent_window_size', 'gpu_memory_preservation': 'gpu_memory_preservation',
    'use_fp32_transformer_output_ui': 'use_fp32_transformer_output', 'rs': 'rs',
    'mp4_crf': 'mp4_crf', 'output_folder_ui': 'output_folder'
}
print(f"FramePack SVC launching with args: {args}")

# --- Model Loading ---
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')
print("Initializing models...")
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
print("Models loaded to CPU. Configuring...")
vae.eval(); text_encoder.eval(); text_encoder_2.eval(); image_encoder.eval(); transformer.eval()
if not high_vram: vae.enable_slicing(); vae.enable_tiling()
transformer.high_quality_fp32_output_for_inference = False
transformer.to(dtype=torch.bfloat16); vae.to(dtype=torch.float16); image_encoder.to(dtype=torch.float16); text_encoder.to(dtype=torch.float16); text_encoder_2.to(dtype=torch.float16)
vae.requires_grad_(False); text_encoder.requires_grad_(False); text_encoder_2.requires_grad_(False); image_encoder.requires_grad_(False); transformer.requires_grad_(False)
if not high_vram:
    print("Low VRAM mode: Installing DynamicSwap for transformer and text_encoder.")
    DynamicSwapInstaller.install_model(transformer, device=gpu); DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    print("High VRAM mode: Moving all models to GPU.")
    text_encoder.to(gpu); text_encoder_2.to(gpu); image_encoder.to(gpu); vae.to(gpu); transformer.to(gpu)
print("Model configuration and placement complete.")

# --- Helper Functions ---
def patched_video_is_playable(video_filepath): return True
gr.processing_utils.video_is_playable = patched_video_is_playable

def save_settings_to_file(filepath, *ui_values_tuple):
    settings_to_save = dict(zip(ALL_TASK_UI_KEYS, ui_values_tuple))
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(settings_to_save, f, indent=4)
        gr.Info(f"Workspace saved to {filepath}")
        print(f"Workspace saved to {filepath}")
    except Exception as e:
        gr.Warning(f"Error saving workspace: {e}")
        traceback.print_exc()

def save_workspace(*ui_values_tuple):
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save Full Workspace As",
        defaultextension=".json",
        initialfile="framepack_workspace.json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root.destroy()
    if file_path:
        save_settings_to_file(file_path, *ui_values_tuple)
    else:
        gr.Warning("Save cancelled by user.")

def save_as_default_workspace(*ui_values_tuple):
    gr.Info(f"Saving current settings as default to {SETTINGS_FILENAME}")
    save_settings_to_file(SETTINGS_FILENAME, *ui_values_tuple)

def get_default_values_map():
    return {
        'prompt': '', 'n_prompt': '', 'total_second_length': 5.0, 'seed': -1,
        'use_teacache': True, 'preview_frequency_ui': 5, 'segments_to_decode_csv': '',
        'gs_ui': 10.0, 'gs_schedule_shape_ui': 'Off', 'gs_final_ui': 10.0, 'steps': 25,
        'cfg': 1.0, 'latent_window_size': 9, 'gpu_memory_preservation': 6.0,
        'use_fp32_transformer_output_ui': False, 'rs': 0.0, 'mp4_crf': 18,
        'output_folder_ui': outputs_folder,
    }

def load_settings_from_file(filepath, return_updates=True):
    default_values_map = get_default_values_map()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_settings = json.load(f)
        gr.Info(f"Loaded workspace from {filepath}")
    except Exception as e:
        gr.Warning(f"Could not load workspace from {filepath}: {e}")
        loaded_settings = {}

    final_settings = default_values_map.copy()
    final_settings.update(loaded_settings)
    output_values = []
    for key in ALL_TASK_UI_KEYS:
        raw_value = final_settings.get(key)
        new_val = raw_value
        try:
            if key in ['seed', 'latent_window_size', 'steps', 'mp4_crf', 'preview_frequency_ui']:
                new_val = int(raw_value)
            elif key in ['total_second_length', 'cfg', 'gs_ui', 'rs', 'gpu_memory_preservation', 'gs_final_ui']:
                new_val = float(raw_value)
            elif key in ['use_teacache', 'use_fp32_transformer_output_ui']:
                if isinstance(raw_value, str):
                    new_val = raw_value.lower() == 'true'
                elif not isinstance(raw_value, bool):
                    new_val = default_values_map.get(key)
        except (ValueError, TypeError):
            print(f"Settings Warning: Could not convert '{raw_value}' for '{key}'. Using default.")
            new_val = default_values_map.get(key)
        output_values.append(new_val)
    if return_updates:
        return [gr.update(value=val) for val in output_values]
    else:
        return output_values

def load_workspace():
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Workspace JSON File",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root.destroy()
    if file_path:
        return load_settings_from_file(file_path, return_updates=True)
    return [gr.update()] * len(ALL_TASK_UI_KEYS)

def load_default_workspace_on_start():
    if os.path.exists(SETTINGS_FILENAME):
        print(f"Found and loading default workspace from {SETTINGS_FILENAME}")
        return load_settings_from_file(SETTINGS_FILENAME)
    print("No default workspace file found. Using default values.")
    default_vals = get_default_values_map()
    return [default_vals[key] for key in ALL_TASK_UI_KEYS]

def np_to_base64_uri(np_array_or_tuple, format="png"):
    if np_array_or_tuple is None: return None
    try:
        np_array = np_array_or_tuple[0] if isinstance(np_array_or_tuple, tuple) and len(np_array_or_tuple) > 0 and isinstance(np_array_or_tuple[0], np.ndarray) else np_array_or_tuple if isinstance(np_array_or_tuple, np.ndarray) else None
        if np_array is None: return None
        pil_image = Image.fromarray(np_array.astype(np.uint8))
        if format.lower() == "jpeg" and pil_image.mode == "RGBA": pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO(); pil_image.save(buffer, format=format.upper()); img_bytes = buffer.getvalue()
        return f"data:image/{format.lower()};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e: print(f"Error converting NumPy to base64: {e}"); return None

def get_queue_state(state_dict_gr_state):
    if "queue_state" not in state_dict_gr_state: state_dict_gr_state["queue_state"] = {"queue": [], "next_id": 1, "processing": False, "editing_task_id": None}
    return state_dict_gr_state["queue_state"]

def update_queue_df_display(queue_state):
    queue = queue_state.get("queue", []); data = []; processing = queue_state.get("processing", False); editing_task_id = queue_state.get("editing_task_id", None)
    for i, task in enumerate(queue):
        params = task['params']; task_id = task['id']; prompt_display = (params['prompt'][:77] + '...') if len(params['prompt']) > 80 else params['prompt']; prompt_title = params['prompt'].replace('"', '&quot;'); prompt_cell = f'<span title="{prompt_title}">{prompt_display}</span>'; img_uri = np_to_base64_uri(params.get('input_image'), format="png"); thumbnail_size = "50px"; img_md = f'<img src="{img_uri}" alt="Input" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display:block; margin:auto; object-fit:contain;" />' if img_uri else ""; is_processing_current_task = processing and i == 0; is_editing_current_task = editing_task_id == task_id; task_status_val = task.get("status", "pending");
        if is_processing_current_task: status_display = "⏳ Processing"
        elif is_editing_current_task: status_display = "✏️ Editing"
        elif task_status_val == "done": status_display = "✅ Done"
        elif task_status_val == "error": status_display = f"❌ Error: {task.get('error_message', 'Unknown')}"
        elif task_status_val == "aborted": status_display = "⏹️ Aborted"
        elif task_status_val == "pending": status_display = "⏸️ Pending"
        data.append([task_id, status_display, prompt_cell, f"{params.get('total_second_length', 0):.1f}s", params.get('steps', 0), img_md, "↑", "↓", "✖", "✎"])
    return gr.DataFrame(value=data, visible=len(data) > 0)

def add_or_update_task_in_queue(state_dict_gr_state, *args_from_ui_controls_tuple):
    queue_state = get_queue_state(state_dict_gr_state); editing_task_id = queue_state.get("editing_task_id", None)

    # The first argument is the image gallery, the rest are the parameter controls.
    input_images_pil_list = args_from_ui_controls_tuple[0]
    all_ui_values_tuple = args_from_ui_controls_tuple[1:]
    if not input_images_pil_list:
        gr.Warning("Input image(s) are required!")
        return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task to Queue" if editing_task_id is None else "Update Task"), gr.update(visible=editing_task_id is not None)

    # Create a dictionary of UI parameters from the new ALL_TASK_UI_KEYS list.
    temp_params_from_ui = dict(zip(ALL_TASK_UI_KEYS, all_ui_values_tuple))

    # Build the dictionary of parameters that the backend worker function expects.
    base_params_for_worker_dict = {}
    for ui_key, worker_key in UI_TO_WORKER_PARAM_MAP.items():
        if ui_key == 'gs_schedule_shape_ui':
             # Special handling to convert the UI radio choice to a boolean for the worker.
            base_params_for_worker_dict[worker_key] = temp_params_from_ui.get(ui_key) != 'Off'
        else:
            base_params_for_worker_dict[worker_key] = temp_params_from_ui.get(ui_key)

    if editing_task_id is not None:
        if len(input_images_pil_list) > 1: gr.Warning("Cannot update task with multiple images. Cancel edit."); return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Update Task"), gr.update(visible=True)
        pil_img_for_update = input_images_pil_list[0][0] if isinstance(input_images_pil_list[0], tuple) else input_images_pil_list[0]
        if not isinstance(pil_img_for_update, Image.Image): gr.Warning("Invalid image format for update."); return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Update Task"), gr.update(visible=True)
        img_np_for_update = np.array(pil_img_for_update)
        with queue_lock:
            task_found = False
            for task in queue_state["queue"]:
                if task["id"] == editing_task_id:
                    task["params"] = {**base_params_for_worker_dict, 'input_image': img_np_for_update}
                    task["status"] = "pending"
                    task_found = True
                    break
            if not task_found: gr.Warning(f"Task {editing_task_id} not found for update.")
            else: gr.Info(f"Task {editing_task_id} updated.")
            queue_state["editing_task_id"] = None
    else:
        tasks_added_count = 0; first_new_task_id = -1
        with queue_lock:
            for img_obj in input_images_pil_list:
                pil_image = img_obj[0] if isinstance(img_obj, tuple) else img_obj
                if not isinstance(pil_image, Image.Image): gr.Warning("Skipping invalid image input."); continue
                img_np_data = np.array(pil_image)
                next_id = queue_state["next_id"]
                if first_new_task_id == -1: first_new_task_id = next_id
                task = {"id": next_id, "params": {**base_params_for_worker_dict, 'input_image': img_np_data}, "status": "pending"}
                queue_state["queue"].append(task); queue_state["next_id"] += 1; tasks_added_count += 1
        if tasks_added_count > 0: gr.Info(f"Added {tasks_added_count} task(s) (start ID: {first_new_task_id}).")
        else: gr.Warning("No valid tasks added.")

    return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task(s) to Queue", variant="secondary"), gr.update(visible=False)


def cancel_edit_mode_action(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state)
    if queue_state.get("editing_task_id") is not None: gr.Info("Edit cancelled."); queue_state["editing_task_id"] = None
    return state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value="Add Task(s) to Queue", variant="secondary"), gr.update(visible=False)

def move_task_in_queue(state_dict_gr_state, direction: str, selected_indices_list: list):
    if not selected_indices_list or not selected_indices_list[0]: return state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state))
    idx = int(selected_indices_list[0][0]); queue_state = get_queue_state(state_dict_gr_state); queue = queue_state["queue"]
    with queue_lock:
        if direction == 'up' and idx > 0: queue[idx], queue[idx-1] = queue[idx-1], queue[idx]
        elif direction == 'down' and idx < len(queue) - 1: queue[idx], queue[idx+1] = queue[idx+1], queue[idx]
    return state_dict_gr_state, update_queue_df_display(queue_state)

def remove_task_from_queue(state_dict_gr_state, selected_indices_list: list):
    removed_task_id = None
    if not selected_indices_list or not selected_indices_list[0]: return state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state)), removed_task_id
    idx = int(selected_indices_list[0][0]); queue_state = get_queue_state(state_dict_gr_state); queue = queue_state["queue"]
    with queue_lock:
        if 0 <= idx < len(queue): removed_task = queue.pop(idx); removed_task_id = removed_task['id']; gr.Info(f"Removed task {removed_task_id}.")
        else: gr.Warning("Invalid index for removal.")
    return state_dict_gr_state, update_queue_df_display(queue_state), removed_task_id

def handle_queue_action_on_select(evt: gr.SelectData, state_dict_gr_state, *ui_param_controls_tuple):
    if evt.index is None or evt.value not in ["↑", "↓", "✖", "✎"]:
        return [state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state))] + [gr.update()] * (len(ALL_TASK_UI_KEYS) + 4)

    row_index, col_index = evt.index; button_clicked = evt.value; queue_state = get_queue_state(state_dict_gr_state); queue = queue_state["queue"]; processing_flag = queue_state.get("processing", False)

    outputs_list = [state_dict_gr_state, update_queue_df_display(queue_state)] + [gr.update()] * (len(ALL_TASK_UI_KEYS) + 4)

    if button_clicked == "↑":
        if processing_flag and row_index == 0: gr.Warning("Cannot move processing task."); return outputs_list
        new_state, new_df = move_task_in_queue(state_dict_gr_state, 'up', [[row_index, col_index]]); outputs_list[0], outputs_list[1] = new_state, new_df
    elif button_clicked == "↓":
        if processing_flag and row_index == 0: gr.Warning("Cannot move processing task."); return outputs_list
        if processing_flag and row_index == 1: gr.Warning("Cannot move below processing task."); return outputs_list
        new_state, new_df = move_task_in_queue(state_dict_gr_state, 'down', [[row_index, col_index]]); outputs_list[0], outputs_list[1] = new_state, new_df
    elif button_clicked == "✖":
        if processing_flag and row_index == 0: gr.Warning("Cannot remove processing task."); return outputs_list
        new_state, new_df, removed_id = remove_task_from_queue(state_dict_gr_state, [[row_index, col_index]]); outputs_list[0], outputs_list[1] = new_state, new_df
        # If the removed task was being edited, cancel the edit mode.
        if removed_id is not None and queue_state.get("editing_task_id", None) == removed_id:
            queue_state["editing_task_id"] = None
            outputs_list[2 + 1 + len(ALL_TASK_UI_KEYS)] = gr.update(value="Add Task(s) to Queue", variant="secondary") # add_task_button
            outputs_list[2 + 1 + len(ALL_TASK_UI_KEYS) + 1] = gr.update(visible=False) # cancel_edit_task_button
    elif button_clicked == "✎":
        if processing_flag and row_index == 0: gr.Warning("Cannot edit processing task."); return outputs_list
        if 0 <= row_index < len(queue):
            task_to_edit = queue[row_index]; task_id_to_edit = task_to_edit['id']; params_to_load_to_ui = task_to_edit['params']
            queue_state["editing_task_id"] = task_id_to_edit; gr.Info(f"Editing Task {task_id_to_edit}.")

            img_np_from_task = params_to_load_to_ui.get('input_image')
            if isinstance(img_np_from_task, np.ndarray):
                pil_image_for_gallery = Image.fromarray(img_np_from_task)
                # The gallery expects a list of tuples or list of images
                outputs_list[2] = gr.update(value=[(pil_image_for_gallery, "loaded_image")])
            else:
                outputs_list[2] = gr.update(value=None)

            for i, ui_key in enumerate(ALL_TASK_UI_KEYS):
                worker_key = UI_TO_WORKER_PARAM_MAP.get(ui_key)
                if worker_key in params_to_load_to_ui:
                    value_from_task = params_to_load_to_ui[worker_key]
                    if ui_key == 'gs_schedule_shape_ui':
                        outputs_list[3 + i] = gr.update(value="Linear" if value_from_task else "Off")
                    else:
                        outputs_list[3 + i] = gr.update(value=value_from_task)

            outputs_list[2 + 1 + len(ALL_TASK_UI_KEYS)] = gr.update(value="Update Task", variant="secondary")
            outputs_list[2 + 1 + len(ALL_TASK_UI_KEYS) + 1] = gr.update(visible=True)
        else: gr.Warning("Invalid index for edit.")
    return outputs_list


def clear_task_queue_action(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state); queue = queue_state["queue"]; processing = queue_state["processing"]; cleared_count = 0
    with queue_lock:
        if processing:
             if len(queue) > 1: cleared_count = len(queue) - 1; queue_state["queue"] = [queue[0]]; gr.Info(f"Cleared {cleared_count} pending tasks.")
             else: gr.Info("Only processing task in queue.")
        elif queue: cleared_count = len(queue); queue.clear(); gr.Info(f"Cleared {cleared_count} tasks.")
        else: gr.Info("Queue empty.")
    if not processing and cleared_count > 0 and os.path.isfile(AUTOSAVE_FILENAME):
         try: os.remove(AUTOSAVE_FILENAME); print(f"Cleared autosave: {AUTOSAVE_FILENAME}.")
         except OSError as e: print(f"Error deleting autosave: {e}")
    return state_dict_gr_state, update_queue_df_display(queue_state)

def save_queue_to_zip(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state); queue = queue_state.get("queue", [])
    if not queue: gr.Info("Queue is empty. Nothing to save."); return state_dict_gr_state, ""
    zip_buffer = io.BytesIO(); saved_files_count = 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_manifest = []; image_paths_in_zip = {}
            for task in queue:
                params_copy = task['params'].copy(); task_id_s = task['id']; input_image_np_data = params_copy.pop('input_image', None)
                manifest_entry = {"id": task_id_s, "params": params_copy, "status": task.get("status", "pending")}
                if input_image_np_data is not None:
                    img_hash = hash(input_image_np_data.tobytes()); img_filename_in_zip = f"task_{task_id_s}_input.png"; manifest_entry['image_ref'] = img_filename_in_zip
                    if img_hash not in image_paths_in_zip:
                        img_save_path = os.path.join(tmpdir, img_filename_in_zip)
                        try: Image.fromarray(input_image_np_data).save(img_save_path, "PNG"); image_paths_in_zip[img_hash] = img_filename_in_zip; saved_files_count +=1
                        except Exception as e: print(f"Error saving image for task {task_id_s} in zip: {e}")
                queue_manifest.append(manifest_entry)
            manifest_path = os.path.join(tmpdir, "queue_manifest.json");
            with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(queue_manifest, f, indent=4)
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="queue_manifest.json")
                for img_hash, img_filename_rel in image_paths_in_zip.items(): zf.write(os.path.join(tmpdir, img_filename_rel), arcname=img_filename_rel)
            zip_buffer.seek(0); zip_base64 = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
            gr.Info(f"Queue with {len(queue)} tasks ({saved_files_count} images) prepared for download.")
            return state_dict_gr_state, zip_base64
    except Exception as e: print(f"Error creating zip for queue: {e}"); traceback.print_exc(); gr.Warning("Failed to create zip data."); return state_dict_gr_state, ""
    finally: zip_buffer.close()

def load_queue_from_zip(state_dict_gr_state, uploaded_zip_file_obj):
    if not uploaded_zip_file_obj or not hasattr(uploaded_zip_file_obj, 'name') or not Path(uploaded_zip_file_obj.name).is_file(): gr.Warning("No valid file selected."); return state_dict_gr_state, update_queue_df_display(get_queue_state(state_dict_gr_state))
    queue_state = get_queue_state(state_dict_gr_state); newly_loaded_queue = []; max_id_in_file = 0; loaded_image_count = 0; error_messages = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir_extract:
            with zipfile.ZipFile(uploaded_zip_file_obj.name, 'r') as zf:
                if "queue_manifest.json" not in zf.namelist(): raise ValueError("queue_manifest.json not found in zip")
                zf.extractall(tmpdir_extract)
            manifest_path = os.path.join(tmpdir_extract, "queue_manifest.json")
            with open(manifest_path, 'r', encoding='utf-8') as f: loaded_manifest = json.load(f)

            for task_data in loaded_manifest:
                params_from_manifest = task_data.get('params', {}); task_id_loaded = task_data.get('id', 0); max_id_in_file = max(max_id_in_file, task_id_loaded)
                image_ref_from_manifest = task_data.get('image_ref'); input_image_np_data = None
                if image_ref_from_manifest:
                    img_path_in_extract = os.path.join(tmpdir_extract, image_ref_from_manifest)
                    if os.path.exists(img_path_in_extract):
                        try:
                            with Image.open(img_path_in_extract) as img_pil:
                                input_image_np_data = np.array(img_pil)
                            loaded_image_count +=1
                        except Exception as img_e: error_messages.append(f"Err loading img for task {task_id_loaded}: {img_e}")
                    else: error_messages.append(f"Missing img file for task {task_id_loaded}: {image_ref_from_manifest}")
                runtime_task = {"id": task_id_loaded, "params": {**params_from_manifest, 'input_image': input_image_np_data}, "status": "pending"}
                newly_loaded_queue.append(runtime_task)
        with queue_lock: queue_state["queue"] = newly_loaded_queue; queue_state["next_id"] = max(max_id_in_file + 1, queue_state.get("next_id", 1))
        gr.Info(f"Loaded {len(newly_loaded_queue)} tasks ({loaded_image_count} images).")
        if error_messages: gr.Warning(" ".join(error_messages))
    except Exception as e: print(f"Error loading queue: {e}"); traceback.print_exc(); gr.Warning(f"Failed to load queue: {str(e)[:200]}")
    finally:
        if uploaded_zip_file_obj and hasattr(uploaded_zip_file_obj, 'name') and uploaded_zip_file_obj.name and tempfile.gettempdir() in os.path.abspath(uploaded_zip_file_obj.name):
            try: os.remove(uploaded_zip_file_obj.name)
            except OSError: pass
    return state_dict_gr_state, update_queue_df_display(queue_state)

global_state_for_autosave_dict = {}
def autosave_queue_on_exit_action(state_dict_gr_state_ref):
    print("Attempting to autosave queue on exit...")
    queue_state = get_queue_state(state_dict_gr_state_ref)
    if not queue_state.get("queue"): print("Autosave: Queue is empty."); return
    try:
        _dummy_state_ignored, zip_b64_for_save = save_queue_to_zip(state_dict_gr_state_ref)
        if zip_b64_for_save:
            with open(AUTOSAVE_FILENAME, "wb") as f: f.write(base64.b64decode(zip_b64_for_save))
            print(f"Autosave successful: Queue saved to {AUTOSAVE_FILENAME}")
        else: print("Autosave failed: Could not generate zip data.")
    except Exception as e: print(f"Error during autosave: {e}"); traceback.print_exc()
atexit.register(autosave_queue_on_exit_action, global_state_for_autosave_dict)

def autoload_queue_on_start_action(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state); df_update = update_queue_df_display(queue_state)
    if not queue_state["queue"] and Path(AUTOSAVE_FILENAME).is_file():
        print(f"Autoloading queue from {AUTOSAVE_FILENAME}...")
        class MockFilepath:
            def __init__(self, name): self.name = name
        temp_state_for_load = {"queue_state": queue_state.copy()}
        loaded_state_result, df_update_after_load = load_queue_from_zip(temp_state_for_load, MockFilepath(AUTOSAVE_FILENAME))
        if loaded_state_result["queue_state"]["queue"]:
            queue_state.update(loaded_state_result["queue_state"]); df_update = df_update_after_load
            print(f"Autoload successful. Loaded {len(queue_state['queue'])} tasks.")
            try: os.remove(AUTOSAVE_FILENAME); print(f"Removed autosave file: {AUTOSAVE_FILENAME}")
            except OSError as e: print(f"Error removing autosave file '{AUTOSAVE_FILENAME}': {e}")
        else:
            print("Autoload: File existed but queue remains empty."); queue_state["queue"] = []; queue_state["next_id"] = 1; df_update = update_queue_df_display(queue_state)
    return state_dict_gr_state, df_update

def extract_metadata_from_pil_image(pil_image: Image.Image) -> dict:
    if pil_image is None: return {}
    pnginfo_data = getattr(pil_image, 'text', None)
    if not isinstance(pnginfo_data, dict): return {}
    params_json_str = pnginfo_data.get('parameters')
    if not params_json_str:
        print("No 'parameters' JSON key found in image metadata.")
        return {}
    try:
        extracted_params = json.loads(params_json_str)
        return extracted_params if isinstance(extracted_params, dict) else {}
    except json.JSONDecodeError as e:
        print(f"Error decoding metadata JSON: {e}")
        return {}

def handle_image_upload_for_metadata(gallery_pil_list):
    if not gallery_pil_list or not isinstance(gallery_pil_list, list):
        return gr.update(visible=False)

    first_image_obj = gallery_pil_list[0]
    # The gallery component might wrap the image in a tuple
    pil_image = first_image_obj[0] if isinstance(first_image_obj, tuple) else first_image_obj

    if not isinstance(pil_image, Image.Image):
        return gr.update(visible=False)
    try:
        metadata = extract_metadata_from_pil_image(pil_image)
        # MERGE: Only show the modal if creative parameters are found in the metadata.
        if metadata and any(key in metadata for key in CREATIVE_PARAM_KEYS):
            return gr.update(visible=True) # Show the modal
    except Exception as e:
        print(f"Error handling image upload for metadata: {e}")

    return gr.update(visible=False)

def ui_load_params_from_image_metadata(gallery_data_list):
    # Loads ONLY creative parameters from image metadata and returns UI updates.
    updates_for_ui = [gr.update()] * len(CREATIVE_PARAM_KEYS)

    try:
        first_image_obj = gallery_data_list[0]
        pil_image = first_image_obj[0] if isinstance(first_image_obj, tuple) else first_image_obj
        extracted_metadata = extract_metadata_from_pil_image(pil_image)
    except Exception:
        return updates_for_ui

    if not extracted_metadata:
        gr.Info("No relevant parameters found in image metadata.")
        return updates_for_ui

    gr.Info(f"Found metadata, applying to creative settings...")
    num_applied = 0
    default_values_map = get_default_values_map()
    params_to_apply = {}

    # Special handling for gs_schedule_active_ui, mapping it back to gs_schedule_shape_ui
    if 'gs_schedule_active_ui' in extracted_metadata:
        params_to_apply['gs_schedule_shape_ui'] = "Linear" if str(extracted_metadata['gs_schedule_active_ui']).lower() == 'true' else "Off"

    # Iterate through only the creative keys to build the update list.
    for i, key in enumerate(CREATIVE_PARAM_KEYS):
        if key in extracted_metadata or key in params_to_apply:
            raw_value = params_to_apply.get(key, extracted_metadata.get(key))
            new_val = raw_value
            try:
                # Apply the same robust type checking as the main settings loader
                if key in ['seed', 'steps', 'preview_frequency_ui']: new_val = int(raw_value)
                elif key in ['total_second_length', 'cfg', 'gs_ui', 'rs', 'gs_final_ui']: new_val = float(raw_value)
                updates_for_ui[i] = gr.update(value=new_val); num_applied += 1
            except (ValueError, TypeError) as ve:
                print(f"Metadata Warning: Could not convert '{raw_value}' for '{key}': {ve}")

    if num_applied > 0: gr.Info(f"Applied {num_applied} parameter(s) from image metadata.")
    return updates_for_ui

def apply_and_hide_modal(gallery_data_list):
    updates = ui_load_params_from_image_metadata(gallery_data_list)
    return [gr.update(visible=False)] + updates

def ui_update_total_segments(total_seconds_ui, latent_window_size_ui):
    if not isinstance(total_seconds_ui, (int, float)) or not isinstance(latent_window_size_ui, (int, float)): return "Segments: Invalid input"
    if latent_window_size_ui <= 0: return "Segments: Invalid window size"
    total_vid_frames = total_seconds_ui * 30; calculated_sections = total_vid_frames / (latent_window_size_ui * 4)
    total_segments = int(max(round(calculated_sections), 1)); return f"Calculated Total Segments: {total_segments}"

def process_task_queue_main_loop(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state)
    abort_event.clear()
    if queue_state["processing"]: gr.Info("Queue is already processing."); yield state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True); return
    if not queue_state["queue"]: gr.Info("Queue is empty."); yield state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False); return
    queue_state["processing"] = True
    yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), gr.update(value="Queue processing started..."), gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True))
    output_stream_for_ui = AsyncStream()

    while queue_state["queue"] and not abort_event.is_set():
        with queue_lock:
            if not queue_state["queue"]: break
            current_task_obj = queue_state["queue"][0]
            task_parameters_for_worker = current_task_obj["params"]
            current_task_id = current_task_obj["id"]

        if task_parameters_for_worker.get('input_image') is None:
            print(f"Skipping task {current_task_id}: Missing input image data.")
            gr.Warning(f"Task {current_task_id} skipped: Input image is missing. Please edit the task to add an image.")
            with queue_lock:
                current_task_obj["status"] = "error"
                current_task_obj["error_message"] = "Missing Image"
            yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True))
            gr.Info("Queue processing halted due to task with missing image. Please remove or fix the task.")
            break
        if task_parameters_for_worker.get('seed') == -1:
            task_parameters_for_worker['seed'] = np.random.randint(0, 2**32 - 1)
            print(f"Task {current_task_id}: Using random seed {task_parameters_for_worker['seed']}")

        print(f"Starting task {current_task_id} (Prompt: {task_parameters_for_worker.get('prompt', '')[:30]}...)."); current_task_obj["status"] = "processing"
        yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), gr.update(value=f"Processing Task {current_task_id}..."), "", gr.update(interactive=False), gr.update(interactive=True))
        worker_args = {**task_parameters_for_worker, 'task_id': current_task_id, 'output_queue_ref': output_stream_for_ui.output_queue, 'abort_event': abort_event, 'text_encoder': text_encoder, 'text_encoder_2': text_encoder_2, 'tokenizer': tokenizer, 'tokenizer_2': tokenizer_2, 'vae': vae, 'feature_extractor': feature_extractor, 'image_encoder': image_encoder, 'transformer': transformer, 'high_vram_flag': high_vram}
        async_run(worker, **worker_args)
        last_known_output_filename = None; task_completed_successfully = False
        while True:
            flag, data_from_worker = output_stream_for_ui.output_queue.next()
            if flag == 'progress':
                msg_task_id, preview_np_array, desc_str, html_str = data_from_worker
                if msg_task_id == current_task_id: yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value=last_known_output_filename), gr.update(visible=(preview_np_array is not None), value=preview_np_array), desc_str, html_str, gr.update(interactive=False), gr.update(interactive=True))
            elif flag == 'file':
                msg_task_id, segment_file_path, segment_info = data_from_worker
                if msg_task_id == current_task_id: last_known_output_filename = segment_file_path; gr.Info(f"Task {current_task_id}: {segment_info}"); yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value=last_known_output_filename), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True))
            elif flag == 'aborted':
                 msg_task_id = data_from_worker; print(f"Task {current_task_id} confirmed aborted by worker."); current_task_obj["status"] = "aborted"; task_completed_successfully = False; break
            elif flag == 'error':
                 msg_task_id, error_message_str = data_from_worker; print(f"Task {current_task_id} failed: {error_message_str}"); gr.Warning(f"Task {current_task_id} Error: {error_message_str}"); current_task_obj["status"] = "error"; current_task_obj["error_message"] = str(error_message_str)[:100]; task_completed_successfully = False; break
            elif flag == 'end':
                 msg_task_id, success_bool, final_video_path = data_from_worker; task_completed_successfully = success_bool; last_known_output_filename = final_video_path if success_bool else last_known_output_filename; current_task_obj["status"] = "done" if success_bool else "error"; print(f"Task {current_task_id} ended. Success: {success_bool}. Output: {last_known_output_filename}"); break

        with queue_lock:
            if queue_state["queue"] and queue_state["queue"][0]["id"] == current_task_id:
                queue_state["queue"].pop(0)
                print(f"Task {current_task_id} popped from queue.")
            else:
                print(f"Warning: Task {current_task_id} not at queue head after processing. It might have been removed already.")
        final_desc = f"Task {current_task_id} {'completed' if task_completed_successfully else 'finished with issues'}."
        yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(value=last_known_output_filename), gr.update(visible=False), gr.update(value=final_desc), gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True))

        if abort_event.is_set(): gr.Info("Queue processing halted by user."); break

    queue_state["processing"] = False;
    print("Queue processing loop finished.")
    final_status_msg = "All tasks processed." if not abort_event.is_set() else "Queue processing aborted."
    yield (state_dict_gr_state, update_queue_df_display(queue_state), gr.update(), gr.update(visible=False), gr.update(value=final_status_msg), gr.update(value=""), gr.update(interactive=True), gr.update(interactive=False))

def abort_current_task_processing_action(state_dict_gr_state):
    queue_state = get_queue_state(state_dict_gr_state);
    if queue_state["processing"]: gr.Info("Abort signal sent. Current task will attempt to stop shortly."); abort_event.set()
    else: gr.Info("Nothing is currently processing.")
    return state_dict_gr_state, gr.update(interactive=not queue_state["processing"])

# --- Gradio UI Definition ---
css_theme = make_progress_bar_css() + """ #queue_df { font-size: 0.85rem; } #queue_df th:nth-child(1), #queue_df td:nth-child(1) { width: 5%; } #queue_df th:nth-child(2), #queue_df td:nth-child(2) { width: 10%; } #queue_df th:nth-child(3), #queue_df td:nth-child(3) { width: 40%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;} #queue_df th:nth-child(4), #queue_df td:nth-child(4) { width: 8%; } #queue_df th:nth-child(5), #queue_df td:nth-child(5) { width: 8%; } #queue_df th:nth-child(6), #queue_df td:nth-child(6) { width: 10%; text-align: center; } #queue_df th:nth-child(7), #queue_df td:nth-child(7), #queue_df th:nth-child(8), #queue_df td:nth-child(8), #queue_df th:nth-child(9), #queue_df td:nth-child(9), #queue_df th:nth-child(10), #queue_df td:nth-child(10) { width: 4%; cursor: pointer; text-align: center; } #queue_df td:hover { background-color: #f0f0f0; } .gradio-container { max-width: 95% !important; margin: auto !important; } """
block = gr.Blocks(css=css_theme, title="FramePack SVC").queue()
with block:
    app_state = gr.State({"queue_state": {"queue": [], "next_id": 1, "processing": False, "editing_task_id": None}})
    gr.Markdown('# FramePack SVC (Stable Video Creation)')

    with Modal(visible=False) as metadata_modal:
        gr.Markdown("Image has saved parameters. Overwrite current creative settings?")
        with gr.Row():
            cancel_metadata_btn = gr.Button("No, Keep Current")
            confirm_metadata_btn = gr.Button("Yes, Apply Settings", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            input_image_gallery_ui = gr.Gallery(type="pil", label="Input Image(s)", height=320, preview=True, allow_preview=True)

            # --- Creative UI Components ---
            prompt_ui = gr.Textbox(label="Prompt", lines=3, placeholder="A detailed description of the motion to generate.")
            example_quick_prompts_ui = gr.Dataset(visible=True, components=[prompt_ui])
            n_prompt_ui = gr.Textbox(label="Negative Prompt", lines=2, placeholder="Concepts to avoid.")
            with gr.Row():
                total_second_length_ui = gr.Slider(label="Total Video Length (Seconds)", minimum=0.1, maximum=120, value=5.0, step=0.1)
                seed_ui = gr.Number(label="Seed", value=-1, precision=0)

            with gr.Accordion("Advanced Settings", open=False):
                total_segments_display_ui = gr.Markdown("Calculated Total Segments: N/A")
                preview_frequency_ui = gr.Slider(label="Preview Frequency", minimum=0, maximum=100, value=5, step=1, info="Produce mp4 preview every N steps (0=off).")
                segments_to_decode_csv_ui = gr.Textbox(label="Preview Segments", placeholder="e.g., 1,5,10. Always saves first & final mp4.", value="")
                with gr.Row():
                    gs_ui = gr.Slider(label="Distilled CFG Start", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                    gs_schedule_shape_ui = gr.Radio(["Off", "Linear"], label="Variable CFG", value="Off")
                    gs_final_ui = gr.Slider(label="Distilled CFG End", minimum=1.0, maximum=32.0, value=10.0, step=0.01, interactive=False)
                    cfg_ui = gr.Slider(label="CFG (Real)", minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                    rs_ui = gr.Slider(label="RS", minimum=0.0, maximum=32.0, value=0.0, step=0.01)
                    steps_ui = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)

            # --- Environment & Debug UI Components ---
            with gr.Accordion("Debug Settings", open=False):
                use_teacache_ui = gr.Checkbox(label='Use TeaCache (Optimize Speed)', value=True)
                use_fp32_transformer_output_checkbox_ui = gr.Checkbox(label="Use FP32 Transformer Output", value=False)
                gpu_memory_preservation_ui = gr.Slider(label="GPU Preserved Memory (GB)", minimum=4, maximum=128, value=6.0, step=0.1)
                mp4_crf_ui = gr.Slider(label="MP4 Compression (CRF)", minimum=0, maximum=51, value=18, step=1)
                latent_window_size_ui = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True)
                output_folder_ui_ctrl = gr.Textbox(label="Output Folder", value=outputs_folder)
                save_as_default_button = gr.Button(value="Save Current as Default", variant="secondary")

            # --- Main Workspace Save/Load Buttons ---
            save_workspace_button = gr.Button(value="Save Workspace", variant="secondary") # Renamed
            load_workspace_button = gr.Button(value="Load Workspace", variant="secondary") # Renamed

        with gr.Column(scale=2):
            gr.Markdown("## Task Queue")
            queue_df_display_ui = gr.DataFrame(headers=["ID", "Status", "Prompt", "Length", "Steps", "Input", "↑", "↓", "✖", "✎"], datatype=["number","markdown","markdown","str","number","markdown","markdown","markdown","markdown","markdown"], col_count=(10,"fixed"), value=[], interactive=False, visible=False, elem_id="queue_df", wrap=True)
            with gr.Row():
                add_task_button = gr.Button(value="Add to Queue", variant="secondary")
                process_queue_button = gr.Button("▶️ Process Queue", variant="primary")
                abort_task_button = gr.Button("⏹️ Abort", variant="stop", interactive=False)
            cancel_edit_task_button = gr.Button("Cancel Edit", visible=False, variant="secondary")
            with gr.Row():
                save_queue_zip_b64_output = gr.Text(visible=False); save_queue_button_ui = gr.DownloadButton("Save Queue", size="sm"); load_queue_button_ui = gr.UploadButton("Load Queue", file_types=[".zip"], size="sm"); clear_queue_button_ui = gr.Button("Clear Pending", size="sm", variant="stop")
            gr.Markdown("## Live Preview")
            current_task_preview_image_ui = gr.Image(interactive=False, visible=False)
            current_task_progress_desc_ui = gr.Markdown('')
            current_task_progress_bar_ui = gr.HTML('')
            gr.Markdown("## Output Video")
            last_finished_video_ui = gr.Video(interactive=True, autoplay=False)

    # --- Define explicit lists of UI components for wiring up events ---
    creative_ui_components = [
        prompt_ui, n_prompt_ui, total_second_length_ui, seed_ui, preview_frequency_ui,
        segments_to_decode_csv_ui, gs_ui, gs_schedule_shape_ui, gs_final_ui, steps_ui, cfg_ui, rs_ui
    ]
    environment_ui_components = [
        use_teacache_ui, use_fp32_transformer_output_checkbox_ui, gpu_memory_preservation_ui,
        mp4_crf_ui, output_folder_ui_ctrl, latent_window_size_ui
    ]
    full_workspace_ui_components = creative_ui_components + environment_ui_components

    task_defining_ui_inputs = [input_image_gallery_ui] + full_workspace_ui_components

    process_queue_outputs_list = [app_state, queue_df_display_ui, last_finished_video_ui, current_task_preview_image_ui, current_task_progress_desc_ui, current_task_progress_bar_ui, process_queue_button, abort_task_button]

    # This list defines all the UI elements that can be updated when a task is selected for editing.
    queue_df_select_outputs_list = [app_state, queue_df_display_ui, input_image_gallery_ui] + full_workspace_ui_components + [add_task_button, cancel_edit_task_button, last_finished_video_ui]

    # Workspace Settings (.json file) Handlers
    save_workspace_button.click(fn=save_workspace, inputs=full_workspace_ui_components, outputs=[])
    load_workspace_button.click(fn=load_workspace, inputs=[], outputs=full_workspace_ui_components)
    save_as_default_button.click(fn=save_as_default_workspace, inputs=full_workspace_ui_components, outputs=[])

    # PNG Metadata (Creative Recipe) Handlers
    input_image_gallery_ui.upload(fn=handle_image_upload_for_metadata, inputs=[input_image_gallery_ui], outputs=[metadata_modal])
    confirm_metadata_outputs = [metadata_modal] + creative_ui_components
    confirm_metadata_btn.click(fn=apply_and_hide_modal, inputs=[input_image_gallery_ui], outputs=confirm_metadata_outputs)
    cancel_metadata_btn.click(lambda: gr.update(visible=False), None, metadata_modal)

    # Queue and Processing Logic
    add_task_button.click(fn=add_or_update_task_in_queue, inputs=[app_state] + task_defining_ui_inputs, outputs=[app_state, queue_df_display_ui, add_task_button, cancel_edit_task_button])
    process_queue_button.click(fn=process_task_queue_main_loop, inputs=[app_state], outputs=process_queue_outputs_list)
    cancel_edit_task_button.click(fn=cancel_edit_mode_action, inputs=[app_state], outputs=[app_state, queue_df_display_ui, add_task_button, cancel_edit_task_button])
    abort_task_button.click(fn=abort_current_task_processing_action, inputs=[app_state], outputs=[app_state, abort_task_button])
    clear_queue_button_ui.click(fn=clear_task_queue_action, inputs=[app_state], outputs=[app_state, queue_df_display_ui])
    save_queue_button_ui.click(fn=save_queue_to_zip, inputs=[app_state], outputs=[app_state, save_queue_zip_b64_output]).then(fn=None, inputs=[save_queue_zip_b64_output], outputs=None, js="""(b64) => { if(!b64) return; const blob = new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], {type: 'application/zip'}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download='framepack_svc_queue.zip'; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url); }""")
    load_queue_button_ui.upload(fn=load_queue_from_zip, inputs=[app_state, load_queue_button_ui], outputs=[app_state, queue_df_display_ui])
    queue_df_display_ui.select(fn=handle_queue_action_on_select, inputs=[app_state] + task_defining_ui_inputs, outputs=queue_df_select_outputs_list)
    def toggle_gs_final(gs_schedule_choice): return gr.update(interactive=(gs_schedule_choice != "Off"))
    gs_schedule_shape_ui.change(fn=toggle_gs_final, inputs=[gs_schedule_shape_ui], outputs=[gs_final_ui])
    for ctrl in [total_second_length_ui, latent_window_size_ui]: ctrl.change(fn=ui_update_total_segments, inputs=[total_second_length_ui, latent_window_size_ui], outputs=[total_segments_display_ui])

    block.load(fn=load_default_workspace_on_start, inputs=[], outputs=full_workspace_ui_components).then(fn=autoload_queue_on_start_action, inputs=[app_state], outputs=[app_state, queue_df_display_ui]).then(lambda s_val: global_state_for_autosave_dict.update(s_val), inputs=[app_state], outputs=None).then(fn=ui_update_total_segments, inputs=[total_second_length_ui, latent_window_size_ui], outputs=[total_segments_display_ui])

expanded_outputs_folder = os.path.abspath(os.path.expanduser(outputs_folder))
if __name__ == "__main__":
    print("Starting FramePack SVC application...")
    block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser, allowed_paths=[expanded_outputs_folder])