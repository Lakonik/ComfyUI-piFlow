import logging

import torch
import comfy
import comfy.ldm.flux.layers
from comfy import model_management
from comfy.model_detection import unet_prefix_from_state_dict, convert_diffusers_mmdit, detect_unet_config

try:
    from comfy.utils import convert_old_quants
except ImportError:
    convert_old_quants = None


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


def merge_adapter_state_dict(base_model_sd, base_unet_config, adapter_sd=None, adapter_metadata=None):
    metadata = {}
    if adapter_metadata is None:
        adapter_metadata = {}

    new_sd = base_model_sd.copy()
    lora_sd = {}

    if adapter_sd is None:
        return new_sd, lora_sd, metadata

    updated_weight_layers = set()
    updated_keys = set()
    cloned_weight_keys = set()

    key_mapping = {}
    base_image_model = base_unet_config["image_model"]
    if base_image_model in ("flux", "flux2"):
        key_mapping = comfy.utils.flux_to_diffusers(base_unet_config, output_prefix="")

    for k in adapter_sd.keys():
        if "lora" in k:
            if base_image_model in ("flux", "flux2") and not k.startswith("transformer."):
                lora_sd["transformer." + k] = adapter_sd[k]
            else:
                lora_sd[k] = adapter_sd[k]
        else:
            if k in key_mapping:
                comfy_weight_key = convert_diffusers_to_comfyui(
                    new_sd, adapter_sd[k], key_mapping[k], cloned_weight_keys=cloned_weight_keys)
            else:
                new_sd[k] = adapter_sd[k]
                comfy_weight_key = k
            updated_keys.add(comfy_weight_key)
            if comfy_weight_key.endswith(".weight"):
                updated_weight_layers.add(comfy_weight_key[:-7])

    quantization_postfixes = [
        "scale_input",
        "scale_weight",
        "input_scale",
        "weight_scale",
        "weight_scale_2",
        "comfy_quant",
    ]
    for layer in updated_weight_layers:
        for postfix in quantization_postfixes:
            auxiliary_key = ".".join([layer, postfix])
            if auxiliary_key in new_sd and auxiliary_key not in updated_keys:
                del new_sd[auxiliary_key]

    metadata.update(adapter_metadata)
    return new_sd, lora_sd, metadata


def build_model_from_state_dict(new_sd, metadata, weight_dtype, model_options, model_config_factory):
    parameters = comfy.utils.calculate_parameters(new_sd)

    if convert_old_quants is not None:
        if model_options.get("custom_operations", None) is None:
            new_sd, metadata = convert_old_quants(new_sd, "", metadata=metadata)

    model_config = model_config_factory(new_sd, "", metadata=metadata)
    if model_config is None:
        return None
    normalize_flux_norm_keys(new_sd, model_config)

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if hasattr(model_config, "quant_config"):
        has_quant_config = model_config.quant_config is not None
        if has_quant_config:
            weight_dtype = None
    else:
        has_quant_config = getattr(model_config, "layer_quant_config", None) is not None
        if getattr(model_config, "scaled_fp8", None) is not None:
            weight_dtype = None

    dtype = model_options.get("dtype", None)
    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    load_device = model_management.get_torch_device()
    if has_quant_config:
        manual_cast_dtype = model_management.unet_manual_cast(
            None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = new_sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device)


def load_lakonlab_model_state_dict(
        base_model_sd, adapter_sd=None, model_options=None,
        base_metadata=None, adapter_metadata=None, model_config_factory=None):
    if model_options is None:
        model_options = {}
    if base_metadata is None:
        base_metadata = {}
    if adapter_metadata is None:
        adapter_metadata = {}

    base_model_sd, base_unet_config = prepare_base_model_state_dict(base_model_sd, base_metadata)
    if base_unet_config is None:
        return None, None

    weight_dtype = comfy.utils.weight_dtype(base_model_sd)
    metadata = base_metadata.copy()
    new_sd, lora_sd, adapter_metadata = merge_adapter_state_dict(
        base_model_sd, base_unet_config, adapter_sd, adapter_metadata)
    metadata.update(adapter_metadata)

    model = build_model_from_state_dict(
        new_sd, metadata, weight_dtype, model_options, model_config_factory)
    if model is None:
        return None, None
    return model, lora_sd


def load_lakonlab_model_from_files(
        base_model_path, adapter_path, model_options=None, adapter_strength=1.0,
        model_config_factory=None, error_label="model"):
    if model_options is None:
        model_options = {}

    base_model_sd, base_metadata = comfy.utils.load_torch_file(base_model_path, return_metadata=True)
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
    if len(lora_sd) > 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, adapter_strength, None)
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
