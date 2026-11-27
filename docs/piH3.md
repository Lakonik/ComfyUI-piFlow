# pi-H3: Video Generation in 2–4 Steps Without a VAE Decoder

<a href="https://hanshengchen.com/blog/pi-h3/"><img src="https://hanshengchen.com/blog/pi-h3/teaser.jpg" width="500" alt=""/></a>

pi-H3 distills [MiniMax H3](https://www.minimax.io/blog/minimax-h3) into a 2- or 4-step generator of video with synchronized audio, and can predict the video frames directly without a VAE decoder. Read more in our [blog post](https://hanshengchen.com/blog/pi-h3/).

## Speed

The pi-H3 DWD workflows are the fastest way to generate MiniMax H3 videos. End-to-end time at 2 steps (INT8 ConvRot) on an **unlocked RTX 3090** ([gpu-unlock](https://github.com/zagitta/gpu-unlock), recommended) with **CUDA 13**, with the models loaded and the prompt encoded:

pi-H3-DWD-Spark workflow ([Spark-Attn](https://github.com/zechengtang/Spark-H3), recommended):

| Resolution | 5 s | 10 s | 15 s |
|---|---|---|---|
| 864x480 | 14.6 s | 36.4 s | 73.2 s |
| 1344x768 | 34.6 s | 80.6 s | 147.7 s |

pi-H3-DWD-KJ-Sol workflow (Kijai's [Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton)):

| Resolution | 5 s | 10 s | 15 s |
|---|---|---|---|
| 864x480 | 14.6 s | 40.2 s | 81.7 s |
| 1344x768 | 41.0 s | 114.2 s | 234.1 s |

pi-H3-DWD workflow (ComfyUI's built-in Sol-Attn):

| Resolution | 5 s | 10 s | 15 s |
|---|---|---|---|
| 864x480 | 14.3 s | 36.7 s | 74.0 s |
| 1344x768 | 35.9 s | 87.2 s | 164.6 s |

## Workflows

This repo provides video generation [workflows](../workflows) based on MiniMax H3.

Supports the MiniMax H3 FL2VA model (first-last-frame to video and audio), which generates video with synchronized audio from a text prompt, optionally conditioned on a first frame, a last frame, or both.

pi-H3 provides 2-step and 4-step adapters. We recommend the DWD workflow, whose adapters predict the video frames directly at the last step, so the video skips the VAE decoder.

Please download the workflow below and drag it into ComfyUI to load it. Set `steps` on the `pi-H3 Adapter (2 or 4 steps)` node to switch between the 2-step and 4-step adapters.

- **pi-H3-DWD-Spark workflow**: video predicted by DWD, with [Spark-Attn](#spark-attn) (**strongly recommended**, needs [Spark-H3](#spark-attn)). Download the version for your GPU: [RTX 30 series (SM86)](../workflows/pi-H3-DWD-Spark-SM86.json), [RTX 40 series (SM89)](../workflows/pi-H3-DWD-Spark-SM89.json), or [RTX 50 series and RTX PRO Blackwell (SM120)](../workflows/pi-H3-DWD-Spark-SM120.json).

All workflows are listed below. Pick a row (with or without DWD) and a sparse attention, and for Spark-Attn the version for your GPU. The workflows without DWD decode the video with the VAE.

| | Spark-Attn (recommended) | Kijai's Sol-Attn | ComfyUI's built-in Sol-Attn |
|---|---|---|---|
| DWD | [SM86](../workflows/pi-H3-DWD-Spark-SM86.json), [SM89](../workflows/pi-H3-DWD-Spark-SM89.json), [SM120](../workflows/pi-H3-DWD-Spark-SM120.json) | [pi-H3-DWD-KJ-Sol.json](../workflows/pi-H3-DWD-KJ-Sol.json) | [pi-H3-DWD.json](../workflows/pi-H3-DWD.json) |
| Without DWD | [SM86](../workflows/pi-H3-Spark-SM86.json), [SM89](../workflows/pi-H3-Spark-SM89.json), [SM120](../workflows/pi-H3-Spark-SM120.json) | [pi-H3-KJ-Sol.json](../workflows/pi-H3-KJ-Sol.json) | [pi-H3.json](../workflows/pi-H3.json) |

### Sparse attention

All pi-H3 workflows use sparse attention to speed up sampling. Bypass the sparse attention node in any workflow to use dense attention.

#### Spark-Attn

[Spark-Attn](https://github.com/zechengtang/Spark-H3) is a block sparse attention method for MiniMax H3 that groups similar tokens into blocks and reweights the tokens it approximates. Of the three options, it keeps pi-H3 videos the closest to dense attention. The Spark-Attn workflows run its `MiniMax H3 Spark Attention` node in `speed` mode, so every step is sparse. The node is built for one GPU architecture, so use the workflow for your GPU.

To install it, download the [Spark-H3 release](https://github.com/zechengtang/Spark-H3/releases) archive for your operating system and GPU (e.g. `ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm86-cu130.zip` for an RTX 30 series GPU on Linux), extract it into your ComfyUI `custom_nodes` directory, and run its installer with the Python that runs ComfyUI:
```bash
# run the following commands in your ComfyUI `custom_nodes` directory
unzip ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm86-cu130.zip
python ComfyUI-Spark-H3/install.py
```

The Linux SM86 and SM89 v0.2.0 archives were repackaged on 2026-11-24 to restore the matching Spark Attention nodes. If the SM86 or SM89 node is missing after installing an older copy of the archive, [re-download the current ZIP](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.2.0), extract it over the existing `ComfyUI-Spark-H3` directory, and restart ComfyUI.

Spark-H3 lists ComfyUI up to v0.44 as supported; we tested it with ComfyUI v0.46.0.

#### Kijai's Sol-Attn

The KJ-Sol workflows use the BF16 Sol-Attn of Kijai's [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) (the `Patch Sol-Attn` node), which works on any GPU that Triton supports. To install it, clone it into your ComfyUI `custom_nodes` directory. It needs [Triton](https://github.com/triton-lang/triton) (on Windows, [triton-windows](https://github.com/woct0rdho/triton-windows)).
```bash
# run the following command in your ComfyUI `custom_nodes` directory
git clone https://github.com/kijai/ComfyUI-SolAttn_triton
```

Its Triton kernels are compiled and tuned on the first run, so the first video takes longer.

#### ComfyUI's built-in Sol-Attn

ComfyUI's built-in Sol-Attn (the `Model Sparse Attention` node) needs no extra installation, but it runs in INT8 and visibly degrades pi-H3 videos.

### Model links

Base model

- Download [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) and save it to
<br>`models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`

pi-H3 DWD adapters

- Download [pi_h3_2step_dwd_adapter_int8_convrot.safetensors](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_2step_dwd_adapter_int8_convrot.safetensors) and save it to 
<br>`models/loras/pi_h3_2step_dwd_adapter_int8_convrot.safetensors`

  Alternative full-precision version: [pi_h3_2step_dwd_adapter/diffusion_pytorch_model.safetensors](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_2step_dwd_adapter/diffusion_pytorch_model.safetensors)

- Download [pi_h3_4step_dwd_adapter_int8_convrot.safetensors](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_4step_dwd_adapter_int8_convrot.safetensors) and save it to 
<br>`models/loras/pi_h3_4step_dwd_adapter_int8_convrot.safetensors`

  Alternative full-precision version: [pi_h3_4step_dwd_adapter/diffusion_pytorch_model.safetensors](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_4step_dwd_adapter/diffusion_pytorch_model.safetensors)

pi-H3 adapters for the workflow without DWD

- Download [pi_h3_2step_adapter/diffusion_pytorch_model.safetensors](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_2step_adapter/diffusion_pytorch_model.safetensors) and save it to 
<br>`models/loras/pi_h3_2step_adapter.safetensors`

- Download [pi_h3_4step_adapter/diffusion_pytorch_model.safetensors](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_4step_adapter/diffusion_pytorch_model.safetensors) and save it to 
<br>`models/loras/pi_h3_4step_adapter.safetensors`

Text encoder

- Download [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) and save it to
<br>`models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`

VAE

- Download [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) and save it to
<br>`models/vae/minimax_h3_video_vae_int8_convrot.safetensors`

  The video VAE only encodes the first and last frames, and decodes the video in the workflow without DWD.

- Download [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) and save it to
<br>`models/vae/minimax_h3_audio_vae_fp32.safetensors`

### Sampler steps

`steps` on `pi-H3 Adapter (2 or 4 steps)` sets the recommended `pi-Flow Scheduler` settings: `max_raw_t` 0.93 for 2 steps and 0.96 for 4 steps, and `final_step_size_scale` 0.6.
