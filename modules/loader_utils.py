import json
import logging
import threading
from contextlib import contextmanager

import torch
import comfy
import comfy.ldm.flux.layers
import comfy.model_detection
import comfy.sd
from comfy.model_detection import unet_prefix_from_state_dict, convert_diffusers_mmdit, detect_unet_config


_MODEL_CONFIG_FACTORY_KEY = "comfyui_piflow_model_config_factory"
_MODEL_DETECTION_LOCK = threading.RLock()

_LORA_REQUIRED_PAIRS = (
    (".lora_A.default.weight", ".lora_B.default.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_A", ".lora_B"),
)
_LORA_PRIMARY_SUFFIXES = tuple(suffix for pair in _LORA_REQUIRED_PAIRS for suffix in pair)
_LORA_COMPANION_SUFFIXES = _LORA_PRIMARY_SUFFIXES + (
    ".alpha",
    ".dora_scale",
    ".lora_A.bias",
    ".lora_B.bias",
    ".lora_mid.weight",
    ".reshape_weight",
)
_LORA_NATIVE_PREFIXES = (
    "base_model.",
    "diffusion_model.",
    "lora_transformer_",
    "lora_unet_",
    "lycoris_",
    "transformer.",
    "unet.",
)
_QUANTIZATION_POSTFIXES = (
    "scale_input",
    "scale_weight",
    "input_scale",
    "weight_scale",
    "weight_scale_2",
    "comfy_quant",
)


def flux_norm_target_suffix():
    """Match the active ComfyUI Flux RMSNorm state-dict name."""
    if getattr(comfy.ldm.flux.layers, "RMSNorm", None) is None:
        return "weight"
    return "scale"


def normalize_flux_norm_keys(state_dict, model_config):
    if model_config.unet_config.get("image_model") not in ("flux", "flux2", "gm_flux", "gm_flux2", "asym_flux2"):
        return

    target_suffix = flux_norm_target_suffix()
    source_suffix = "scale" if target_suffix == "weight" else "weight"
    source_ending = f".{source_suffix}"

    for key in list(state_dict.keys()):
        if not key.endswith(source_ending):
            continue
        if ".norm.query_norm." not in key and ".norm.key_norm." not in key:
            continue
        target_key = f"{key[:-len(source_ending)]}.{target_suffix}"
        if target_key not in state_dict:
            state_dict[target_key] = state_dict.pop(key)


def _key_without_suffix(key, suffixes):
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None


def split_adapter_state_dict(adapter_sd, base_image_model):
    """Separate complete LoRA groups from tensors merged into the base model."""
    lora_roots = set()
    for key in adapter_sd:
        root = _key_without_suffix(key, _LORA_PRIMARY_SUFFIXES)
        if root is not None:
            lora_roots.add(root)

    for root in lora_roots:
        matching_pairs = [
            pair for pair in _LORA_REQUIRED_PAIRS
            if root + pair[0] in adapter_sd or root + pair[1] in adapter_sd
        ]
        if not any(root + left in adapter_sd and root + right in adapter_sd
                   for left, right in matching_pairs):
            raise ValueError("Incomplete LoRA pair for adapter layer: {}".format(root))

    lora_sd = {}
    full_sd = {}

    for key, value in adapter_sd.items():
        root = _key_without_suffix(key, _LORA_COMPANION_SUFFIXES)
        if root not in lora_roots:
            full_sd[key] = value
            continue

        lora_key = key
        if base_image_model in ("flux", "flux2") and not key.startswith(_LORA_NATIVE_PREFIXES):
            lora_key = "transformer." + key
        lora_sd[lora_key] = value

    return full_sd, lora_sd


def _quantization_layer_from_key(key):
    for postfix in _QUANTIZATION_POSTFIXES:
        suffix = "." + postfix
        if key.endswith(suffix):
            return key[:-len(suffix)], postfix
    return None, None


def _mapped_weight_key(key_mapping, weight_key):
    mapped = key_mapping.get(weight_key)
    if mapped is None:
        return weight_key
    if isinstance(mapped, str):
        return mapped
    return mapped[0]


def _mapped_weight_target(key_mapping, weight_key):
    mapped = key_mapping.get(weight_key)
    if not isinstance(mapped, tuple):
        return None, None
    return mapped[0], mapped[1]


def _quantized_layers(state_dict, metadata):
    layers = set()
    for key in state_dict:
        layer, postfix = _quantization_layer_from_key(key)
        if postfix is not None:
            layers.add(layer)

    quant_metadata = metadata.get("_quantization_metadata")
    if isinstance(quant_metadata, str):
        quant_metadata = json.loads(quant_metadata)
    if quant_metadata is not None:
        layers.update(quant_metadata.get("layers", {}))
    return layers


def _fully_replaced_mapped_weights(full_sd, key_mapping, base_model_sd):
    mapped_slices = {}
    for key in full_sd:
        target_key, offset = _mapped_weight_target(key_mapping, key)
        if target_key is None or offset is None or target_key not in base_model_sd:
            continue
        dim, start, length = offset
        group = mapped_slices.setdefault((target_key, dim), [])
        group.append((start, start + length))

    fully_replaced = set()
    for (target_key, dim), intervals in mapped_slices.items():
        cursor = 0
        for start, end in sorted(intervals):
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor >= base_model_sd[target_key].shape[dim]:
            fully_replaced.add(target_key)
    return fully_replaced


def merge_model_metadata(base_metadata, adapter_metadata, updated_weight_layers, key_mapping):
    metadata = base_metadata.copy()
    metadata.update({k: v for k, v in adapter_metadata.items() if k != "_quantization_metadata"})

    base_quant = base_metadata.get("_quantization_metadata")
    adapter_quant = adapter_metadata.get("_quantization_metadata")
    if base_quant is None and adapter_quant is None:
        return metadata

    if isinstance(base_quant, str):
        base_quant = json.loads(base_quant)
    if isinstance(adapter_quant, str):
        adapter_quant = json.loads(adapter_quant)

    quant_config = (base_quant or {}).copy()
    quant_config.update({k: v for k, v in (adapter_quant or {}).items() if k != "layers"})
    quant_layers = (base_quant or {}).get("layers", {}).copy()
    for layer in updated_weight_layers:
        quant_layers.pop(layer, None)

    for layer, config in (adapter_quant or {}).get("layers", {}).items():
        mapped_weight = _mapped_weight_key(key_mapping, layer + ".weight")
        mapped_layer = mapped_weight[:-len(".weight")] if mapped_weight.endswith(".weight") else layer
        quant_layers[mapped_layer] = config

    if quant_layers:
        quant_config["layers"] = quant_layers
        metadata["_quantization_metadata"] = json.dumps(quant_config)
    else:
        metadata.pop("_quantization_metadata", None)
    return metadata


@contextmanager
def use_model_config_factory(model_config_factory):
    """Route one marked ComfyUI load through piFlow's model detection."""
    with _MODEL_DETECTION_LOCK:
        original = comfy.model_detection.model_config_from_unet

        def model_config_from_unet(
                state_dict, key_prefix, use_base_if_no_match=False, metadata=None):
            if metadata is None or metadata.get(_MODEL_CONFIG_FACTORY_KEY) is not model_config_factory:
                return original(
                    state_dict, key_prefix,
                    use_base_if_no_match=use_base_if_no_match, metadata=metadata)

            original_from_config = comfy.model_detection.model_config_from_unet_config

            def model_config_from_unet_config(
                    unet_config, candidate_state_dict=None, unet_key_prefix=""):
                if candidate_state_dict is not state_dict:
                    return original_from_config(
                        unet_config, candidate_state_dict, unet_key_prefix=unet_key_prefix)
                return model_config_factory(
                    candidate_state_dict, unet_key_prefix, metadata=metadata)

            comfy.model_detection.model_config_from_unet_config = model_config_from_unet_config
            try:
                model_config = original(
                    state_dict, key_prefix,
                    use_base_if_no_match=use_base_if_no_match, metadata=metadata)
            finally:
                comfy.model_detection.model_config_from_unet_config = original_from_config
            if model_config is not None:
                normalize_flux_norm_keys(state_dict, model_config)
            return model_config

        comfy.model_detection.model_config_from_unet = model_config_from_unet
        try:
            yield
        finally:
            comfy.model_detection.model_config_from_unet = original


def convert_diffusers_to_comfyui(state_dict, diffusers_weight, comfy_weight_map, cloned_weight_keys=None):
    """Modified from convert_diffusers_mmdit.

    This updates state_dict in place. Source tensors are never modified.
    """
    if cloned_weight_keys is None:
        cloned_weight_keys = set()

    if isinstance(comfy_weight_map, str):
        comfy_weight_key = comfy_weight_map
        state_dict[comfy_weight_key] = diffusers_weight
    else:
        comfy_weight_key = comfy_weight_map[0]
        if len(comfy_weight_map) > 2:
            weight_convert_fun = comfy_weight_map[2]
        else:
            weight_convert_fun = lambda a: a
        offset = comfy_weight_map[1]
        converted_weight = weight_convert_fun(diffusers_weight)
        if offset is not None:
            updated_weight = state_dict.get(comfy_weight_key, None)
            if updated_weight is None:
                updated_shape = list(diffusers_weight.shape)
                updated_shape[offset[0]] = offset[1] + offset[2]
                updated_weight = torch.empty(
                    updated_shape, device=diffusers_weight.device, dtype=diffusers_weight.dtype)
            elif comfy_weight_key not in cloned_weight_keys:
                updated_weight = updated_weight.clone()
                cloned_weight_keys.add(comfy_weight_key)
            if updated_weight.shape[offset[0]] < offset[1] + offset[2]:
                expanded_shape = list(diffusers_weight.shape)
                expanded_shape[offset[0]] = offset[1] + offset[2]
                expanded_weight = torch.empty(
                    expanded_shape, device=diffusers_weight.device, dtype=diffusers_weight.dtype)
                _updated_weight = expanded_weight.narrow(offset[0], 0, updated_weight.shape[offset[0]])
                _updated_weight[:] = updated_weight
                updated_weight = expanded_weight
                cloned_weight_keys.add(comfy_weight_key)
            target_slice = updated_weight.narrow(offset[0], offset[1], offset[2])
            target_slice[:] = converted_weight
        else:
            updated_weight = converted_weight
        state_dict[comfy_weight_key] = updated_weight
    return comfy_weight_key


def prepare_base_model_state_dict(base_model_sd, base_metadata=None):
    if base_metadata is None:
        base_metadata = {}

    diffusion_model_prefix = unet_prefix_from_state_dict(base_model_sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(base_model_sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        base_model_sd = temp_sd
    base_unet_config = detect_unet_config(base_model_sd, "", metadata=base_metadata)
    if base_unet_config is None:
        base_model_sd = convert_diffusers_mmdit(base_model_sd, "")
        base_unet_config = detect_unet_config(base_model_sd, "", metadata=base_metadata)

    return base_model_sd, base_unet_config


def merge_adapter_state_dict(
        base_model_sd, base_unet_config, adapter_sd=None,
        base_metadata=None, adapter_metadata=None):
    if base_metadata is None:
        base_metadata = {}
    if adapter_metadata is None:
        adapter_metadata = {}

    new_sd = base_model_sd.copy()

    if adapter_sd is None:
        return new_sd, {}, base_metadata.copy()

    updated_weight_layers = set()
    updated_keys = set()
    cloned_weight_keys = set()

    key_mapping = {}
    base_image_model = base_unet_config["image_model"]
    if base_image_model in ("flux", "flux2"):
        key_mapping = comfy.utils.flux_to_diffusers(base_unet_config, output_prefix="")

    full_sd, lora_sd = split_adapter_state_dict(adapter_sd, base_image_model)
    fully_replaced = _fully_replaced_mapped_weights(full_sd, key_mapping, base_model_sd)
    quantized_layers = _quantized_layers(base_model_sd, base_metadata)
    for key in full_sd:
        target_key, offset = _mapped_weight_target(key_mapping, key)
        if target_key is None or offset is None:
            continue
        target_layer = target_key[:-len(".weight")] if target_key.endswith(".weight") else target_key
        if target_layer in quantized_layers and target_key not in fully_replaced:
            raise ValueError(
                "Cannot partially replace quantized weight {} from adapter key {}. "
                "The adapter must replace every mapped slice of that weight.".format(target_key, key))

    for target_key in fully_replaced:
        new_sd.pop(target_key, None)

    for key, value in full_sd.items():
        if key in key_mapping:
            comfy_weight_key = convert_diffusers_to_comfyui(
                new_sd, value, key_mapping[key], cloned_weight_keys=cloned_weight_keys)
        else:
            source_layer, quant_postfix = _quantization_layer_from_key(key)
            if quant_postfix is not None:
                source_weight_key = source_layer + ".weight"
                mapped_weight_key = _mapped_weight_key(key_mapping, source_weight_key)
                if mapped_weight_key != source_weight_key:
                    comfy_layer = mapped_weight_key[:-len(".weight")]
                    comfy_weight_key = ".".join([comfy_layer, quant_postfix])
                    new_sd[comfy_weight_key] = value
                else:
                    new_sd[key] = value
                    comfy_weight_key = key
            else:
                new_sd[key] = value
                comfy_weight_key = key
        updated_keys.add(comfy_weight_key)
        if comfy_weight_key.endswith(".weight"):
            updated_weight_layers.add(comfy_weight_key[:-len(".weight")])

    for layer in updated_weight_layers:
        for postfix in _QUANTIZATION_POSTFIXES:
            auxiliary_key = ".".join([layer, postfix])
            if auxiliary_key in new_sd and auxiliary_key not in updated_keys:
                del new_sd[auxiliary_key]

    metadata = merge_model_metadata(
        base_metadata, adapter_metadata, updated_weight_layers, key_mapping)
    return new_sd, lora_sd, metadata


def build_model_from_state_dict(
        new_sd, metadata, model_options, model_config_factory, disable_dynamic=False):
    metadata = metadata.copy() if metadata is not None else {}
    metadata[_MODEL_CONFIG_FACTORY_KEY] = model_config_factory

    with use_model_config_factory(model_config_factory):
        return comfy.sd.load_diffusion_model_state_dict(
            new_sd, model_options=model_options, metadata=metadata,
            disable_dynamic=disable_dynamic)


def load_lakonlab_model_state_dict(
        base_model_sd, adapter_sd=None, model_options=None,
        base_metadata=None, adapter_metadata=None, model_config_factory=None,
        disable_dynamic=False):
    if model_options is None:
        model_options = {}
    if base_metadata is None:
        base_metadata = {}
    if adapter_metadata is None:
        adapter_metadata = {}

    base_model_sd, base_unet_config = prepare_base_model_state_dict(base_model_sd, base_metadata)
    if base_unet_config is None:
        return None, None

    new_sd, lora_sd, metadata = merge_adapter_state_dict(
        base_model_sd, base_unet_config, adapter_sd,
        base_metadata=base_metadata, adapter_metadata=adapter_metadata)

    model = build_model_from_state_dict(
        new_sd, metadata, model_options, model_config_factory,
        disable_dynamic=disable_dynamic)
    if model is None:
        return None, None
    return model, lora_sd


def load_lakonlab_model_from_files(
        base_model_path, adapter_path, model_options=None, adapter_strength=1.0,
        model_config_factory=None, error_label="model", disable_dynamic=False):
    if model_options is None:
        model_options = {}

    base_model_sd, base_metadata = comfy.utils.load_torch_file(base_model_path, return_metadata=True)
    adapter_sd = adapter_metadata = None
    if adapter_path is not None:
        adapter_sd, adapter_metadata = comfy.utils.load_torch_file(adapter_path, return_metadata=True)

    model, lora_sd = load_lakonlab_model_state_dict(
        base_model_sd, adapter_sd=adapter_sd, model_options=model_options,
        base_metadata=base_metadata, adapter_metadata=adapter_metadata,
        model_config_factory=model_config_factory, disable_dynamic=disable_dynamic)
    if model is None:
        logging.error("ERROR UNSUPPORTED %s MODEL", error_label.upper())
        raise RuntimeError("ERROR: Could not detect {} model type of: {}\n".format(error_label, base_model_path))
    if len(lora_sd) > 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, adapter_strength, None)
    model.cached_patcher_init = (
        load_lakonlab_model_from_files,
        (base_model_path, adapter_path, model_options, adapter_strength,
         model_config_factory, error_label),
    )
    return model


def set_gguf_linear_dtypes(ops, dequant_dtype=None, patch_dtype=None):
    if dequant_dtype in ("default", None):
        ops.Linear.dequant_dtype = None
    elif dequant_dtype in ["target"]:
        ops.Linear.dequant_dtype = dequant_dtype
    else:
        ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

    if patch_dtype in ("default", None):
        ops.Linear.patch_dtype = None
    elif patch_dtype in ["target"]:
        ops.Linear.patch_dtype = patch_dtype
    else:
        ops.Linear.patch_dtype = getattr(torch, patch_dtype)


def load_lakonlab_model_from_gguf(
        base_model_path, adapter_path, model_options=None, adapter_strength=1.0,
        dequant_dtype=None, patch_dtype=None, patch_on_device=None,
        model_config_factory=None, error_label="model",
        gguf_model_patcher=None, gguf_sd_loader=None, ggml_ops_class=None):
    if gguf_model_patcher is None:
        raise RuntimeError(
            "ComfyUI-GGUF not found. Please install the ComfyUI-GGUF custom nodes to enable GGUF loading.")
    if model_options is None:
        model_options = {}

    ops = ggml_ops_class()
    set_gguf_linear_dtypes(ops, dequant_dtype=dequant_dtype, patch_dtype=patch_dtype)

    loaded_gguf_data = gguf_sd_loader(base_model_path)
    if isinstance(loaded_gguf_data, tuple):
        base_model_sd, extra = loaded_gguf_data
        base_metadata = extra.get("metadata", None)
    else:
        base_model_sd = loaded_gguf_data
        base_metadata = None

    model_options = model_options.copy()
    model_options.update(custom_operations=ops)

    adapter_sd = adapter_metadata = None
    if adapter_path is not None:
        adapter_sd, adapter_metadata = comfy.utils.load_torch_file(adapter_path, return_metadata=True)

    model, lora_sd = load_lakonlab_model_state_dict(
        base_model_sd, adapter_sd=adapter_sd, model_options=model_options,
        base_metadata=base_metadata, adapter_metadata=adapter_metadata,
        model_config_factory=model_config_factory)
    if model is None:
        logging.error("ERROR UNSUPPORTED %s MODEL", error_label.upper())
        raise RuntimeError("ERROR: Could not detect {} model type of: {}\n".format(error_label, base_model_path))

    model = gguf_model_patcher.clone(model)
    model.patch_on_device = patch_on_device

    if len(lora_sd) > 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, adapter_strength, None)

    return model
