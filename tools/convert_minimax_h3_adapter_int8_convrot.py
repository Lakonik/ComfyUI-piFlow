"""Convert a diffusers-format LakonLab MiniMax H3 adapter into a ComfyUI-native INT8 ConvRot adapter.

All keys become ComfyUI MiniMax H3 names, LoRA included; LoRA tensors are never quantized. The non-LoRA weights of
the layers that ComfyUI's own INT8 ConvRot checkpoints quantize (the H3 block attention and MLP linears, and the
Sol refiner attention and FFN linears, as in the INT8 ConvRot video VAE decoder) become comfy_kitchen
int8_tensorwise ConvRot weights with per-row scales. Everything else keeps its dtype.

Run it from the ComfyUI directory, with ComfyUI-piFlow installed in custom_nodes:
    python custom_nodes/ComfyUI-piFlow/tools/convert_minimax_h3_adapter_int8_convrot.py ADAPTER.safetensors OUTPUT.safetensors
"""
import importlib.util
import json
import os
import re
import sys

import torch
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from safetensors import safe_open
from safetensors.torch import save_file

# the key rule of the ComfyUI-piFlow nodes, imported as a package from the ComfyUI directory
sys.path.insert(0, os.getcwd())
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "comfyui_piflow", os.path.join(package_dir, "__init__.py"), submodule_search_locations=[package_dir])
sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules[spec.name])
from comfyui_piflow.lakonlab_loaders import minimax_h3_to_diffusers  # noqa: E402

# the MiniMax H3 (pruned) geometry the adapters are trained on
H3_CONFIG = dict(num_layers=50, token_refiner_num_layers=2, num_attention_heads=56, attention_head_dim=128)
COMFY_QUANT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
QUANTIZED_LAYERS = re.compile(
    r"blocks\.\d+\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))"
    r"|dwd_pixel_refiner\.blocks\.\d+\.(attn\.(to_qkv|to_out)|ff\.(w1|w2))")
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def main(src, dst):
    key_map = minimax_h3_to_diffusers(H3_CONFIG)
    with safe_open(src, "pt") as f:
        metadata = f.metadata()
        sd = {k: f.get_tensor(k) for k in f.keys()}
    out = {}
    for key, value in sd.items():
        suffix = next((s for s in LORA_SUFFIXES if key.endswith(s)), None)
        name = key[:-len(suffix)] + ".weight" if suffix else key
        target = key_map.get(name, name)  # the DWD head is a new module and keeps its names
        if not isinstance(target, str):
            target, offset, fn = (*target, None)[:3]
            if offset is not None:
                raise NotImplementedError("{}: adapters that replace a slice of a fused tensor".format(key))
            if fn is not None and suffix != ".lora_A.weight":  # weight and lora_B rows follow the target's rows
                value = fn(value)
        if suffix is not None:
            out[target[:-len(".weight")] + suffix] = value.contiguous()
        elif target.endswith(".weight") and QUANTIZED_LAYERS.fullmatch(target[:-len(".weight")]):
            layer = target[:-len(".weight")]
            qdata, params = TensorWiseINT8Layout.quantize(
                value.float(), per_channel=True, convrot=True, convrot_groupsize=COMFY_QUANT["convrot_groupsize"])
            out[target] = qdata.contiguous()
            out[layer + ".weight_scale"] = params.scale.float().reshape(-1, 1).contiguous()
            out[layer + ".comfy_quant"] = torch.tensor(list(json.dumps(COMFY_QUANT).encode()), dtype=torch.uint8)
        else:
            out[target] = value.contiguous()
    save_file(out, dst, metadata=metadata)
    quantized = sorted(k[:-len(".comfy_quant")] for k in out if k.endswith(".comfy_quant"))
    print("{}: {} tensors, {} quantized layers".format(dst, len(out), len(quantized)))


if __name__ == "__main__":
    main(*sys.argv[1:])
