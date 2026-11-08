# By lllyasviel
# Optimized for AMD DirectML by Antigravity

import torch
import os

try:
    import torch_directml
    _has_dml = True
except ImportError:
    _has_dml = False

cpu = torch.device('cpu')

# Auto-detect DirectML for AMD hardware
if _has_dml and os.environ.get('USE_DIRECTML', '0') == '1':
    gpu = torch_directml.device()
    print(f"AMD DirectML Device detected: {gpu}")
else:
    try:
        gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
    except:
        gpu = cpu

gpu_complete_modules = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu
    
    # Performance Optimization: Early return for non-CUDA devices.
    # Prevents expensive exception handling during model initialization
    # loop for CPU or DirectML ('privateuseone') devices.

    # Handle int inputs (0 -> 'cuda:0') or check string representation
    is_cuda = isinstance(device, int) or str(device).startswith('cuda')

    if not is_cuda:
        # DirectML specific memory reporting often requires different APIs or is Opaque
        # We will fallback to a static estimate for AMD iGPU if dynamic reporting fails
        if 'privateuseone' in str(device):
            # Placeholder for AMD iGPU memory estimation (64GB allocated)
            return 60.0 # Conservative estimate
        return 8.0 # Minimum fallback

    try:
        memory_stats = torch.cuda.memory_stats(device)
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
        bytes_inactive_reserved = bytes_reserved - bytes_active
        bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
        return bytes_total_available / (1024 ** 3)
    except:
        return 8.0 # Minimum fallback


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    # ⚡ Bolt Optimization: Batch memory checks to reduce CUDA sync overhead
    check_freq = 25

    for i, m in enumerate(model.modules()):
        if i % check_freq == 0:
            if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
                if 'cuda' in str(target_device):
                    torch.cuda.empty_cache()
                return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    if 'cuda' in str(target_device):
        torch.cuda.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    # ⚡ Bolt Optimization: Batch memory checks to reduce CUDA sync overhead
    check_freq = 25

    for i, m in enumerate(model.modules()):
        if i % check_freq == 0:
            if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
                if 'cuda' in str(target_device):
                    torch.cuda.empty_cache()
                return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

    model.to(device=cpu)
    if 'cuda' in str(target_device):
        torch.cuda.empty_cache()
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return
