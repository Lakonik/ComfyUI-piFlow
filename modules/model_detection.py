import json
import logging
import torch
from typing import Dict
from comfy.model_detection import detect_unet_config
from .supported_models import models


def _has_gm_heads(sd: Dict[str, object], prefix: str) -> bool:
    return f"{prefix}proj_out_means.weight" in sd and f"{prefix}proj_out_logweights.weight" in sd


def _infer_gm_cfg(sd: Dict[str, torch.Tensor], prefix: str, base_config: Dict):
    """
    Infer patch_size, num_gaussians, out_channels from GM heads.
    proj_out_logweights: (num_gaussians * (patch_size * patch_size), inner_dim)
    proj_out_means:      (num_gaussians * out_channels, inner_dim)
    """
    base_image_model = base_config["image_model"]
    if base_image_model in ('qwen_image', 'flux', 'flux2'):
        patch_size = 2
    else:
        raise ValueError(f"Unknown base image model: {base_image_model}")
    proj_out_logweights = sd[f"{prefix}proj_out_logweights.weight"]
    proj_out_means = sd[f"{prefix}proj_out_means.weight"]
    proj_out_logweights_out_channels = proj_out_logweights.shape[0]
    num_gaussians = proj_out_logweights_out_channels // (patch_size * patch_size)
    out_channels = proj_out_means.shape[0] // max(1, num_gaussians)
    if base_image_model in ('flux', 'flux2'):
        out_channels = out_channels // 4
    return patch_size, num_gaussians, out_channels


def _infer_gm_logstd_cfg(sd: dict, prefix: str):
    weight_shapes = []
    base = f"{prefix}proj_out_logstds."
    for k, v in sd.items():
        if k.startswith(base) and k.endswith(".weight"):
            # v is a Linear weight: (out_features, in_features)
            weight_shapes.append(v.shape)

    gm_num_logstd_layers = len(weight_shapes)

    if gm_num_logstd_layers == 0:
        constant_logstd = '-inf'
        logstd_inner_dim = None

    elif gm_num_logstd_layers == 1:
        constant_logstd = logstd_inner_dim = None

    else:
        constant_logstd = None
        non_final_outs = [weight_shape[0] for weight_shape in weight_shapes if weight_shape[0] != 1]
        assert len(non_final_outs) > 0
        logstd_inner_dim = non_final_outs[0]

    return gm_num_logstd_layers, logstd_inner_dim, constant_logstd


def detect_policy_config(metadata=None):
    if metadata is not None and 'policy_config' in metadata:
        policy_config = json.loads(metadata["policy_config"])
        if policy_config.get("mode") == "polynomial":
            raise ValueError("DX polynomial policies are not supported by ComfyUI-piFlow.")
        return policy_config
    return {}


def detect_piflow_config(state_dict, key_prefix, metadata=None):
    base_config = detect_unet_config(state_dict, key_prefix, metadata=metadata)
    if base_config is None:
        return None, None

    base_image_model = base_config["image_model"]

    if not _has_gm_heads(state_dict, key_prefix):  # not a GMFlow model
        if base_image_model == 'qwen_image':
            # infer out_channels from proj_out.weight
            patch_size = base_config.get("patch_size", 2)
            proj_out_weight = state_dict[f"{key_prefix}proj_out.weight"]
            out_channels = proj_out_weight.shape[0] // (patch_size * patch_size)
            unet_config = base_config.copy()
            unet_config["out_channels"] = out_channels
        elif base_image_model in ('flux', 'flux2'):
            # infer out_channels from proj_out.weight
            patch_size = base_config.get("patch_size", 2)
            proj_out_weight = state_dict[f"{key_prefix}final_layer.linear.weight"]
            out_channels = proj_out_weight.shape[0] // (patch_size * patch_size)
            unet_config = base_config.copy()
            unet_config["out_channels"] = out_channels
        else:
            unet_config = base_config

    else:
        patch_size, num_gaussians, out_channels = _infer_gm_cfg(state_dict, key_prefix, base_config)
        gm_num_logstd_layers, logstd_inner_dim, constant_logstd = _infer_gm_logstd_cfg(state_dict, key_prefix)
        unet_config = base_config.copy()
        unet_config["image_model"] = f"gm_{base_image_model}"
        unet_config["patch_size"] = patch_size
        unet_config["num_gaussians"] = num_gaussians
        unet_config["out_channels"] = out_channels
        unet_config["gm_num_logstd_layers"] = gm_num_logstd_layers
        unet_config["logstd_inner_dim"] = logstd_inner_dim
        unet_config["constant_logstd"] = constant_logstd

    return unet_config, detect_policy_config(metadata)


def model_config_from_piflow_config(unet_config, policy_config, state_dict=None):
    for model_config in models:
        if model_config.matches(unet_config, state_dict):
            model_config = model_config(unet_config)
            if policy_config:
                if getattr(model_config, "policy_config", None) is None:
                    model_config.policy_config = {}
                model_config.policy_config.update(policy_config)
            return model_config

    logging.error("no match {}".format(unet_config))
    return None


def model_config_from_piflow(state_dict, key_prefix, metadata=None):
    unet_config, policy_config = detect_piflow_config(
        state_dict, key_prefix, metadata=metadata)
    if unet_config is None:
        return None
    model_config = model_config_from_piflow_config(
        unet_config, policy_config, state_dict)
    if model_config is None:
        return None

    scaled_fp8_key = "{}scaled_fp8".format(key_prefix)
    if scaled_fp8_key in state_dict:
        scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
        model_config.scaled_fp8 = scaled_fp8_weight.dtype
        if model_config.scaled_fp8 == torch.float32:
            model_config.scaled_fp8 = torch.float8_e4m3fn
        if scaled_fp8_weight.nelement() == 2:
            model_config.optimizations["fp8"] = False
        else:
            model_config.optimizations["fp8"] = True

    return model_config


def detect_asymflow_config(state_dict, key_prefix, metadata=None):
    base_config = detect_unet_config(state_dict, key_prefix, metadata=metadata)
    if base_config is None:
        return None

    has_asym_buffers = (
        f"{key_prefix}proj_buffer" in state_dict
        and f"{key_prefix}scale_buffer" in state_dict
    )
    if not has_asym_buffers:
        return None

    base_image_model = base_config.get("image_model")
    if base_image_model != "flux2":
        logging.error("AsymFlow detected, but only Flux2 Klein is supported right now: %s", base_config)
        return None

    unet_config = base_config.copy()
    unet_config["image_model"] = "asym_flux2"

    proj_buffer = state_dict.get(f"{key_prefix}proj_buffer")
    if proj_buffer is not None:
        patch_dim = proj_buffer.shape[0]
        unet_config["base_rank"] = proj_buffer.shape[1]
    else:
        patch_dim = state_dict[f"{key_prefix}img_in.weight"].shape[1]
        unet_config["base_rank"] = min(128, patch_dim)

    # First supported AsymFlow architecture is RGB/Oklab pixel diffusion.
    unet_config["in_channels"] = 3
    unet_config["out_channels"] = 3
    unet_config["patch_size"] = int(round((patch_dim // 3) ** 0.5))
    unet_config["sigma_min"] = float(metadata.get("sigma_min", 1e-4)) if metadata is not None else 1e-4
    unet_config["num_timesteps"] = float(metadata.get("num_timesteps", 1.0)) if metadata is not None else 1.0
    return unet_config


def model_config_from_asymflow(state_dict, key_prefix, metadata=None):
    unet_config = detect_asymflow_config(state_dict, key_prefix, metadata=metadata)
    if unet_config is None:
        return None
    model_config = model_config_from_piflow_config(unet_config, {}, state_dict)
    if model_config is None:
        return None

    return model_config
