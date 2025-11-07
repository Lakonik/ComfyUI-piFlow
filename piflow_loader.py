import logging
import torch
import comfy
import folder_paths
from comfy import model_management
from comfy.model_detection import unet_prefix_from_state_dict, convert_diffusers_mmdit, model_config_from_diffusers_unet, detect_unet_config
from .modules.model_detection import model_config_from_piflow


def convert_diffusers_to_comfyui(state_dict, diffusers_weight, comfy_weight_map):
    """Modified from convert_diffusers_mmdit

    Note: This is an in-place operation. Tensors in state_dict and diffusers_weight may be updated in place.
    """
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
        if offset is not None:
            updated_weight = state_dict.get(comfy_weight_key, None)
            if updated_weight is None:
                updated_weight = diffusers_weight
            if updated_weight.shape[offset[0]] < offset[1] + offset[2]:
                expanded_shape = list(diffusers_weight.shape)
                expanded_shape[offset[0]] = offset[1] + offset[2]
                expanded_weight = torch.empty(expanded_shape, device=diffusers_weight.device, dtype=diffusers_weight.dtype)
                _updated_weight = expanded_weight.narrow(offset[0], 0, updated_weight.shape[offset[0]])
                _updated_weight[:] = updated_weight
                updated_weight = expanded_weight
            target_slice = updated_weight.narrow(offset[0], offset[1], offset[2])
        else:
            target_slice = updated_weight = diffusers_weight
        target_slice[:] = weight_convert_fun(diffusers_weight)
        state_dict[comfy_weight_key] = updated_weight


def load_piflow_model_state_dict(
        base_model_sd, adapter_sd=None, model_options={},
        base_metadata=None, adapter_metadata=None, adapter_strength=1.0):
    if base_metadata is None:
        base_metadata = {}
    if adapter_metadata is None:
        adapter_metadata = {}

    # prepare base_model_sd first
    diffusion_model_prefix = unet_prefix_from_state_dict(base_model_sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(base_model_sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        base_model_sd = temp_sd
    base_unet_config = detect_unet_config(base_model_sd, "", metadata=base_metadata)
    if base_unet_config is None:
        base_model_sd = convert_diffusers_mmdit(base_model_sd, "")
        base_unet_config = detect_unet_config(base_model_sd, "", metadata=base_metadata)
        if base_unet_config is None:
            return None

    weight_dtype = comfy.utils.weight_dtype(base_model_sd)

    metadata = base_metadata.copy()
    new_sd = base_model_sd.copy()
    lora_sd = {}

    if adapter_sd is not None:
        key_mapping = {}
        base_image_model = base_unet_config["image_model"]
        if base_image_model == 'flux':  # requires conversion
            key_mapping = comfy.utils.flux_to_diffusers(base_unet_config, output_prefix="")

        for k in adapter_sd.keys():
            if "lora" in k:
                if base_image_model == 'flux' and not k.startswith("transformer."):
                    lora_sd["transformer." + k] = adapter_sd[k]
                else:
                    lora_sd[k] = adapter_sd[k]
            else:
                if k in key_mapping:  # convert_diffusers_mmdit
                    convert_diffusers_to_comfyui(new_sd, adapter_sd[k], key_mapping[k])
                else:
                    new_sd[k] = adapter_sd[k]

        metadata.update(adapter_metadata)

    parameters = comfy.utils.calculate_parameters(new_sd)

    model_config = model_config_from_piflow(new_sd, "", metadata=metadata)
    if model_config is None:
        return None

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    dtype = model_options.get("dtype", None)

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    load_device = model_management.get_torch_device()
    if model_config.layer_quant_config is not None:
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
    model = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

    if len(lora_sd) > 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, adapter_strength, None)

    return model


def load_piflow_model(base_model_path, adapter_path, model_options={}, adapter_strength=1.0):
    base_model_sd, base_metadata = comfy.utils.load_torch_file(base_model_path, return_metadata=True)
    adapter_sd = adapter_metadata = None
    if adapter_path is not None:
        adapter_sd, adapter_metadata = comfy.utils.load_torch_file(adapter_path, return_metadata=True)
    model = load_piflow_model_state_dict(
        base_model_sd, adapter_sd=adapter_sd, model_options=model_options,
        base_metadata=base_metadata, adapter_metadata=adapter_metadata, adapter_strength=adapter_strength)
    if model is None:
        logging.error("ERROR UNSUPPORTED PIFLOW MODEL")
        raise RuntimeError("ERROR: Could not detect model type of: {}\n".format(base_model_path))
    return model


class PiFlowLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
            },
            "optional": {
                "adapter_name": (folder_paths.get_filename_list("loras"), {"default": None}),
                "adapter_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_piflow"

    CATEGORY = "piflow"

    def load_piflow(self, model_name, weight_dtype, adapter_name=None, adapter_strength=1.0):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        base_model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        if adapter_name is not None:
            adapter_path = folder_paths.get_full_path_or_raise("loras", adapter_name)
        else:
            adapter_path = None
        model = load_piflow_model(
            base_model_path, adapter_path, 
            model_options=model_options, adapter_strength=adapter_strength)

        return (model, )
