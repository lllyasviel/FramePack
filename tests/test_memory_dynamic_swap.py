import sys, types, importlib


def _import_memory_with_fake_torch():
    """Import diffusers_helper.memory with a minimal fake torch API."""
    fake_torch = types.ModuleType("torch")

    # torch.device -> echo string/device id
    fake_torch.device = lambda x: str(x)

    # Minimal Parameter
    class Parameter:
        def __init__(self, data=None, requires_grad=False):
            self.data = data
            self.requires_grad = requires_grad

        def to(self, **kwargs):
            return self.data

    # Minimal Module
    class Module:
        def __init__(self):
            self._parameters = {}

        def modules(self):
            return []

    # torch.nn namespace
    class NN:
        pass

    NN.Parameter = Parameter
    NN.Module = Module
    fake_torch.nn = NN()

    # torch.cuda stub
    class _CUDA:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def memory_stats(device):
            return {"active_bytes.all.current": 0, "reserved_bytes.all.current": 0}

        @staticmethod
        def mem_get_info(device):
            return (0, 0)

        @staticmethod
        def empty_cache():
            return None

    fake_torch.cuda = _CUDA()

    sys.modules["torch"] = fake_torch

    if "diffusers_helper.memory" in sys.modules:
        del sys.modules["diffusers_helper.memory"]
    return importlib.import_module("diffusers_helper.memory")


def test_dynamic_swap_install_and_uninstall():
    """Installing should wrap __getattr__ to cast parameters; uninstall should restore the class."""
    memory = _import_memory_with_fake_torch()

    # A tiny fake "module" with _parameters and a .to() method
    class Layer:
        def __init__(self):
            self._parameters = {
                "weight": memory.torch.nn.Parameter(data="W", requires_grad=True)
            }

        def to(self, **kwargs):
            self.last_to = kwargs
            return self

        # modules() is used by DynamicSwapInstaller.install_model
        def modules(self):
            yield self

    layer = Layer()
    original_cls = layer.__class__

    # Install swapper and access a parameter to trigger __getattr__
    memory.DynamicSwapInstaller.install_model(layer, dtype="float16")
    assert layer.__class__ is not original_cls  # class is replaced

    wrapped = layer.weight  # __getattr__ should create a new Parameter with casted data
    assert isinstance(wrapped, memory.torch.nn.Parameter)
    assert wrapped.requires_grad is True
    # After uninstall, class should be restored
    memory.DynamicSwapInstaller.uninstall_model(layer)
    assert layer.__class__ is original_cls
    assert "forge_backup_original_class" not in layer.__dict__
