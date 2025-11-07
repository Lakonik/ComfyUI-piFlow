import json
import logging
import torch
from typing import Dict
from comfy.model_detection import detect_unet_config
from .supported_models import models
try:
    from comfy.model_detection import detect_layer_quantization
except ImportError:
    detect_layer_quantization = None


def _has_gm_heads(sd: Dict[str, object], prefix: str) -> bool:
    return f"{prefix}proj_out_means.weight" in sd and f"{prefix}proj_out_logweights.weight" in sd


def _infer_gm_cfg(sd: Dict[str, torch.Tensor], prefix: str, base_config: Dict):
    """
    Infer patch_size, num_gaussians, out_channels from GM heads.
    proj_out_logweights: (num_gaussians * (patch_size * patch_size), inner_dim)
    proj_out_means:      (num_gaussians * out_channels, inner_dim)
    """
    base_image_model = base_config["image_model"]
    if base_image_model in ('qwen_image', 'flux'):
        patch_size = 2
    else:
        raise ValueError(f"Unknown base image model: {base_image_model}")
    proj_out_logweights = sd[f"{prefix}proj_out_logweights.weight"]
    proj_out_means = sd[f"{prefix}proj_out_means.weight"]
    proj_out_logweights_out_channels = proj_out_logweights.shape[0]
    num_gaussians = proj_out_logweights_out_channels // (patch_size * patch_size)
    out_channels = proj_out_means.shape[0] // max(1, num_gaussians)
    if base_image_model == 'flux':
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
        elif base_image_model == 'flux':
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

    if metadata is not None and 'policy_config' in metadata:
        policy_config = json.loads(metadata["policy_config"])
    else:
        policy_config = {}

    return unet_config, policy_config


def model_config_from_piflow_config(unet_config, policy_config, state_dict=None):
    for model_config in models:
        if model_config.matches(unet_config, state_dict):
            model_config = model_config(unet_config)
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

    if detect_layer_quantization is not None:
        # Detect per-layer quantization (mixed precision)
        layer_quant_config = detect_layer_quantization(metadata)
        if layer_quant_config:
            model_config.layer_quant_config = layer_quant_config
            logging.info(f"Detected mixed precision quantization: {len(layer_quant_config)} layers quantized")

    return model_config
