from pathlib import Path
from typing import Dict, List, Optional, Union
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors"):
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.

    """
    
    state_dict = _fetch_state_dict(
    lora_path,
    weight_name,
    True,
    True,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None)


    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    
    transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=weight_name.split(".")[0])
    print("LoRA weights loaded successfully.")
    return transformer
    
# TODO(neph1): remove when HunyuanVideoTransformer3DModelPacked is in _SET_ADAPTER_SCALE_FN_MAPPING
def set_adapters(
        transformer,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):

    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

    # Expand weights into a list, one entry per adapter
    # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)

    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
        )

    # Set None values to default of 1.0
    # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
    weights = [w if w is not None else 1.0 for w in weights]

    # e.g. [{...}, 7] -> [{expanded dict...}, 7]
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING["HunyuanVideoTransformer3DModel"]
    weights = scale_expansion_fn(transformer, weights)

    set_weights_and_activate_adapters(transformer, adapter_names, weights)