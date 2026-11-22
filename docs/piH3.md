# pi-H3: Video Generation in 2–4 Steps Without a VAE Decoder

<a href="https://hanshengchen.com/blog/pi-h3/"><img src="https://hanshengchen.com/blog/pi-h3/teaser.jpg" width="500" alt=""/></a>

pi-H3 distills [MiniMax H3](https://www.minimax.io/blog/minimax-h3) into a 2- or 4-step generator of video with synchronized audio, and can predict the video frames directly without a VAE decoder. Read more in our [blog post](https://hanshengchen.com/blog/pi-h3/).

## Speed

The pi-H3 DWD workflows are the fastest way to generate MiniMax H3 videos. End-to-end time at 2 steps (INT8 ConvRot) on an **unlocked RTX 3090** ([gpu-unlock](https://github.com/zagitta/gpu-unlock), recommended) with **CUDA 13**, with the models loaded and the prompt encoded:

pi-H3-DWD-KJ-Sol workflow (Kijai's [Sol sparse attention](https://github.com/kijai/ComfyUI-SolAttn_triton), recommended):

| Resolution | 5 s | 10 s | 15 s |
|---|---|---|---|
| 864x480 | 14.6 s | 40.2 s | 81.7 s |
| 1344x768 | 41.0 s | 114.2 s | 234.1 s |

pi-H3-DWD workflow (ComfyUI's built-in Sol sparse attention):

| Resolution | 5 s | 10 s | 15 s |
|---|---|---|---|
| 864x480 | 14.3 s | 36.7 s | 74.0 s |
| 1344x768 | 35.9 s | 87.2 s | 164.6 s |

## Workflows

This repo provides video generation [workflows](../workflows) based on MiniMax H3.

Supports the MiniMax H3 FL2VA model (first-last-frame to video and audio), which generates video with synchronized audio from a text prompt, optionally conditioned on a first frame, a last frame, or both.

pi-H3 provides 2-step and 4-step adapters. We recommend the DWD workflow, whose adapters predict the video frames directly at the last step, so the video skips the VAE decoder.

Please download the workflow below and drag it into ComfyUI to load it. Set `steps` on the `pi-H3 Adapter (2 or 4 steps)` node to switch between the 2-step and 4-step adapters.

- **[pi-H3-DWD-KJ-Sol.json](../workflows/pi-H3-DWD-KJ-Sol.json)**: video predicted by DWD, with Kijai's Sol sparse attention (**strongly recommended**, needs [ComfyUI-SolAttn_triton](#sparse-attention))

Additional workflows are listed below. The workflows without DWD decode the video with the VAE. ComfyUI's built-in Sol is faster than Kijai's, but noticeably worse in quality.

| | Kijai's Sol (recommended) | ComfyUI's built-in Sol |
|---|---|---|
| DWD | [pi-H3-DWD-KJ-Sol.json](../workflows/pi-H3-DWD-KJ-Sol.json) | [pi-H3-DWD.json](../workflows/pi-H3-DWD.json) |
| Without DWD | [pi-H3-KJ-Sol.json](../workflows/pi-H3-KJ-Sol.json) | [pi-H3.json](../workflows/pi-H3.json) |

### Sparse attention

All pi-H3 workflows use Sol sparse attention to speed up sampling. ComfyUI's built-in Sol (the `Model Sparse Attention` node) runs in INT8 and visibly degrades pi-H3 videos. The KJ Sol workflows use the BF16 Sol of Kijai's [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) (the `Patch Sol-Attn` node) instead, which is much closer to the Sol of the official pi-H3 demo.

To install it, clone it into your ComfyUI `custom_nodes` directory. It needs [Triton](https://github.com/triton-lang/triton) (on Windows, [triton-windows](https://github.com/woct0rdho/triton-windows)).
```bash
# run the following command in your ComfyUI `custom_nodes` directory
git clone https://github.com/kijai/ComfyUI-SolAttn_triton
```

Its Triton kernels are compiled and tuned on the first run, so the first video takes longer. In either workflow, bypass the sparse attention node to use dense attention.

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
