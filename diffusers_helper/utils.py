import os
import cv2
import json
import random
import glob
import torch
import einops
import numpy as np
import datetime
import torchvision

import safetensors.torch as sf
from PIL import Image


def min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = max(s1, s0)
    raw_max = max(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def d_resize(x, y):
    H, W, C = y.shape
    new_min = min(H, W)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (W, H), interpolation=interpolation)
    return y


def resize_and_center_crop(image, target_width, target_height):
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_and_center_crop_pytorch(image, target_width, target_height):
    B, C, H, W = image.shape

    if H == target_height and W == target_width:
        return image

    scale_factor = max(target_width / W, target_height / H)
    resized_width = int(round(W * scale_factor))
    resized_height = int(round(H * scale_factor))

    resized = torch.nn.functional.interpolate(image, size=(resized_height, resized_width), mode='bilinear', align_corners=False)

    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    cropped = resized[:, :, top:top + target_height, left:left + target_width]

    return cropped


def resize_without_crop(image, target_width, target_height):
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


def just_crop(image, w, h):
    if h == image.shape[0] and w == image.shape[1]:
        return image

    original_height, original_width = image.shape[:2]
    k = min(original_height / h, original_width / w)
    new_width = int(round(w * k))
    new_height = int(round(h * k))
    x_start = (original_width - new_width) // 2
    y_start = (original_height - new_height) // 2
    cropped_image = image[y_start:y_start + new_height, x_start:x_start + new_width]
    return cropped_image


def write_to_json(data, file_path):
    temp_file_path = file_path + ".tmp"
    with open(temp_file_path, 'wt', encoding='utf-8') as temp_file:
        json.dump(data, temp_file, indent=4)
    os.replace(temp_file_path, file_path)
    return


def read_from_json(file_path):
    with open(file_path, 'rt', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_active_parameters(m):
    return {k: v for k, v in m.named_parameters() if v.requires_grad}


def cast_training_params(m, dtype=torch.float32):
    result = {}
    for n, param in m.named_parameters():
        if param.requires_grad:
            param.data = param.to(dtype)
            result[n] = param
    return result


def separate_lora_AB(parameters, B_patterns=None):
    parameters_normal = {}
    parameters_B = {}

    if B_patterns is None:
        B_patterns = ['.lora_B.', '__zero__']

    for k, v in parameters.items():
        if any(B_pattern in k for B_pattern in B_patterns):
            parameters_B[k] = v
        else:
            parameters_normal[k] = v

    return parameters_normal, parameters_B


def set_attr_recursive(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)
    return


def print_tensor_list_size(tensors):
    total_size = 0
    total_elements = 0

    if isinstance(tensors, dict):
        tensors = tensors.values()

    for tensor in tensors:
        total_size += tensor.nelement() * tensor.element_size()
        total_elements += tensor.nelement()

    total_size_MB = total_size / (1024 ** 2)
    total_elements_B = total_elements / 1e9

    print(f"Total number of tensors: {len(tensors)}")
    print(f"Total size of tensors: {total_size_MB:.2f} MB")
    print(f"Total number of parameters: {total_elements_B:.3f} billion")
    return


@torch.no_grad()
def batch_mixture(a, b=None, probability_a=0.5, mask_a=None):
    batch_size = a.size(0)

    if b is None:
        b = torch.zeros_like(a)

    if mask_a is None:
        mask_a = torch.rand(batch_size) < probability_a

    mask_a = mask_a.to(a.device)
    mask_a = mask_a.reshape((batch_size,) + (1,) * (a.dim() - 1))
    result = torch.where(mask_a, a, b)
    return result


@torch.no_grad()
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


@torch.no_grad()
def supress_lower_channels(m, k, alpha=0.01):
    data = m.weight.data.clone()

    assert int(data.shape[1]) >= k

    data[:, :k] = data[:, :k] * alpha
    m.weight.data = data.contiguous().clone()
    return m


def freeze_module(m):
    if not hasattr(m, '_forward_inside_frozen_module'):
        m._forward_inside_frozen_module = m.forward
    m.requires_grad_(False)
    m.forward = torch.no_grad()(m.forward)
    return m


def get_latest_safetensors(folder_path):
    safetensors_files = glob.glob(os.path.join(folder_path, '*.safetensors'))

    if not safetensors_files:
        raise ValueError('No file to resume!')

    latest_file = max(safetensors_files, key=os.path.getmtime)
    latest_file = os.path.abspath(os.path.realpath(latest_file))
    return latest_file


def generate_random_prompt_from_tags(tags_str, min_length=3, max_length=32):
    tags = tags_str.split(', ')
    tags = random.sample(tags, k=min(random.randint(min_length, max_length), len(tags)))
    prompt = ', '.join(tags)
    return prompt


def interpolate_numbers(a, b, n, round_to_int=False, gamma=1.0):
    numbers = a + (b - a) * (np.linspace(0, 1, n) ** gamma)
    if round_to_int:
        numbers = np.round(numbers).astype(int)
    return numbers.tolist()


def uniform_random_by_intervals(inclusive, exclusive, n, round_to_int=False):
    edges = np.linspace(0, 1, n + 1)
    points = np.random.uniform(edges[:-1], edges[1:])
    numbers = inclusive + (exclusive - inclusive) * points
    if round_to_int:
        numbers = np.round(numbers).astype(int)
    return numbers.tolist()


def soft_append_bcthw(history, current, overlap=0):
    if overlap <= 0:
        return torch.cat([history, current], dim=2)

    assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
    assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
    
    weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output.to(history)


def save_bcthw_as_mp4(x, output_filename, fps=10, crf=0):
    b, c, t, h, w = x.shape

    per_row = b
    for p in [6, 5, 4, 3, 2]:
        if b % p == 0:
            per_row = p
            break

    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    torchvision.io.write_video(output_filename, x, fps=fps, video_codec='libx264', options={'crf': str(int(crf))})
    return x


def save_bcthw_as_png(x, output_filename):
    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, 'b c t h w -> c (b h) (t w)')
    torchvision.io.write_png(x, output_filename)
    return output_filename


def save_bchw_as_png(x, output_filename):
    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, 'b c h w -> c h (b w)')
    torchvision.io.write_png(x, output_filename)
    return output_filename


def add_tensors_with_padding(tensor1, tensor2):
    if tensor1.shape == tensor2.shape:
        return tensor1 + tensor2

    shape1 = tensor1.shape
    shape2 = tensor2.shape

    new_shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))

    padded_tensor1 = torch.zeros(new_shape)
    padded_tensor2 = torch.zeros(new_shape)

    padded_tensor1[tuple(slice(0, s) for s in shape1)] = tensor1
    padded_tensor2[tuple(slice(0, s) for s in shape2)] = tensor2

    result = padded_tensor1 + padded_tensor2
    return result


def print_free_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_mem_mb = free_mem / (1024 ** 2)
        total_mem_mb = total_mem / (1024 ** 2)
        print(f"Free memory: {free_mem_mb:.2f} MB")
        print(f"Total memory: {total_mem_mb:.2f} MB")
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        free_mem_mb = (torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()) / (1024 ** 2)
        total_mem_mb = torch.mps.recommended_max_memory() / (1024 ** 2)
        print(f"Free memory: {free_mem_mb:.2f} MB")
        print(f"Total memory: {total_mem_mb:.2f} MB")
    return


def print_gpu_parameters(device, state_dict, log_count=1):
    summary = {"device": device, "keys_count": len(state_dict)}

    logged_params = {}
    for i, (key, tensor) in enumerate(state_dict.items()):
        if i >= log_count:
            break
        logged_params[key] = tensor.flatten()[:3].tolist()

    summary["params"] = logged_params

    print(str(summary))
    return


def visualize_txt_as_img(width, height, text, font_path='font/DejaVuSans.ttf', size=18):
    from PIL import Image, ImageDraw, ImageFont

    txt = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype(font_path, size=size)

    if text == '':
        return np.array(txt)

    # Split text into lines that fit within the image width
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        line_with_word = f"{current_line} {word}"
        if draw.textbbox((0, 0), line_with_word, font=font)[2] <= width:
            current_line = line_with_word
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)

    # Draw the text line by line
    y = 0
    line_height = draw.textbbox((0, 0), "A", font=font)[3]

    for line in lines:
        if y + line_height > height:
            break  # stop drawing if the next line will be outside the image
        draw.text((0, y), line, fill="black", font=font)
        y += line_height

    return np.array(txt)


def blue_mark(x):
    x = x.copy()
    c = x[:, :, 2]
    b = cv2.blur(c, (9, 9))
    x[:, :, 2] = ((c - b) * 16.0 + b).clip(-1, 1)
    return x


def green_mark(x):
    x = x.copy()
    x[:, :, 2] = -1
    x[:, :, 0] = -1
    return x


def frame_mark(x):
    x = x.copy()
    x[:64] = -1
    x[-64:] = -1
    x[:, :8] = 1
    x[:, -8:] = 1
    return x


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


@torch.no_grad()
def duplicate_prefix_to_suffix(x, count, zero_out=False):
    if zero_out:
        return torch.cat([x, torch.zeros_like(x[:count])], dim=0)
    else:
        return torch.cat([x, x[:count]], dim=0)


def weighted_mse(a, b, weight):
    return torch.mean(weight.float() * (a.float() - b.float()) ** 2)


def clamped_linear_interpolation(x, x_min, y_min, x_max, y_max, sigma=1.0):
    x = (x - x_min) / (x_max - x_min)
    x = max(0.0, min(x, 1.0))
    x = x ** sigma
    return y_min + x * (y_max - y_min)


def expand_to_dims(x, target_dims):
    return x.view(*x.shape, *([1] * max(0, target_dims - x.dim())))


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int):
    if tensor is None:
        return None

    first_dim = tensor.shape[0]

    if first_dim == batch_size:
        return tensor

    if batch_size % first_dim != 0:
        raise ValueError(f"Cannot evenly repeat first dim {first_dim} to match batch_size {batch_size}.")

    repeat_times = batch_size // first_dim

    return tensor.repeat(repeat_times, *[1] * (tensor.dim() - 1))


def dim5(x):
    return expand_to_dims(x, 5)


def dim4(x):
    return expand_to_dims(x, 4)


def dim3(x):
    return expand_to_dims(x, 3)


def crop_or_pad_yield_mask(x, length):
    B, F, C = x.shape
    device = x.device
    dtype = x.dtype

    if F < length:
        y = torch.zeros((B, length, C), dtype=dtype, device=device)
        mask = torch.zeros((B, length), dtype=torch.bool, device=device)
        y[:, :F, :] = x
        mask[:, :F] = True
        return y, mask

    return x[:, :length, :], torch.ones((B, length), dtype=torch.bool, device=device)


def extend_dim(x, dim, minimal_length, zero_pad=False):
    original_length = int(x.shape[dim])

    if original_length >= minimal_length:
        return x

    if zero_pad:
        padding_shape = list(x.shape)
        padding_shape[dim] = minimal_length - original_length
        padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
    else:
        idx = (slice(None),) * dim + (slice(-1, None),) + (slice(None),) * (len(x.shape) - dim - 1)
        last_element = x[idx]
        padding = last_element.repeat_interleave(minimal_length - original_length, dim=dim)

    return torch.cat([x, padding], dim=dim)


def lazy_positional_encoding(t, repeats=None):
    if not isinstance(t, list):
        t = [t]

    from diffusers.models.embeddings import get_timestep_embedding

    te = torch.tensor(t)
    te = get_timestep_embedding(timesteps=te, embedding_dim=256, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1.0)

    if repeats is None:
        return te

    te = te[:, None, :].expand(-1, repeats, -1)

    return te


def state_dict_offset_merge(A, B, C=None):
    result = {}
    keys = A.keys()

    for key in keys:
        A_value = A[key]
        B_value = B[key].to(A_value)

        if C is None:
            result[key] = A_value + B_value
        else:
            C_value = C[key].to(A_value)
            result[key] = A_value + B_value - C_value

    return result


def state_dict_weighted_merge(state_dicts, weights):
    if len(state_dicts) != len(weights):
        raise ValueError("Number of state dictionaries must match number of weights")

    if not state_dicts:
        return {}

    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero")

    normalized_weights = [w / total_weight for w in weights]

    keys = state_dicts[0].keys()
    result = {}

    for key in keys:
        result[key] = state_dicts[0][key] * normalized_weights[0]

        for i in range(1, len(state_dicts)):
            state_dict_value = state_dicts[i][key].to(result[key])
            result[key] += state_dict_value * normalized_weights[i]

    return result


def group_files_by_folder(all_files):
    grouped_files = {}

    for file in all_files:
        folder_name = os.path.basename(os.path.dirname(file))
        if folder_name not in grouped_files:
            grouped_files[folder_name] = []
        grouped_files[folder_name].append(file)

    list_of_lists = list(grouped_files.values())
    return list_of_lists


def generate_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime('%y%m%d_%H%M%S')
    milliseconds = f"{int(now.microsecond / 1000):03d}"
    random_number = random.randint(0, 9999)
    return f"{timestamp}_{milliseconds}_{random_number}"


def write_PIL_image_with_png_info(image, metadata, path):
    from PIL.PngImagePlugin import PngInfo

    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, value)

    image.save(path, "PNG", pnginfo=png_info)
    return image


def torch_safe_save(content, path):
    torch.save(content, path + '_tmp')
    os.replace(path + '_tmp', path)
    return path


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
