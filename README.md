# ComfyUI pi-Flow Nodes for Fast Few-Step Sampling

<img src="https://raw.githubusercontent.com/Lakonik/piFlow/refs/heads/main/assets/teaser.jpg" alt=""/>


**ComfyUI-piFlow** is a collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implement the pi-Flow few-step sampling workflow. All images in the above example were generated using pi-Flow with only 4 sampling steps.

[pi-Flow](https://arxiv.org/abs/2510.14974) is a novel method for flow-based few-step generation. It achieves both high quality and diversity in generated images with as few as 4 sampling steps. Notably,  pi-Flow’s results generally align with the base model’s outputs and exhibit significantly higher diversity than those from DMD models (e.g., [Qwen-Image Lightning](https://github.com/ModelTC/Qwen-Image-Lightning)), as shown below.

<img src="https://raw.githubusercontent.com/Lakonik/piFlow/refs/heads/main/assets/diversity_comparison.jpg" width="1000" alt=""/>

In addition, when using photorealistic style LoRAs, pi-Flow produces significantly better texture details than DMD models, as shown below (zoom in for best view).

<img src="https://raw.githubusercontent.com/Lakonik/piFlow/refs/heads/main/assets/piflow_dmd_texture_comparison.jpg" width="1000" alt=""/>

## Installation

**This repo requires ComfyUI version 0.3.64 or higher**. Make sure your ComfyUI is up to date before installing.

### ComfyUI Manager

If you are using [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager), you can load a [workflow](#workflows) first, and then install the missing nodes via ComfyUI Manager.

### Manual Installation

For manual installation, simply clone this repo into your ComfyUI `custom_nodes` directory.
```bash
# run the following command in your ComfyUI `custom_nodes` directory
git clone https://github.com/Lakonik/ComfyUI-piFlow
```

## Workflows

This repo provides text-to-image [workflows](workflows) based on FLUX.1 dev and Qwen-Image. 

### pi-Qwen-Image

Please download the image below and drag it into ComfyUI to load the pi-Qwen-Image workflow.  

<img src="workflows/pi-Qwen-Image.png" width="600" alt=""/>

### pi-Flux

Please download the image below and drag it into ComfyUI to load the pi-Flux workflow.  

<img src="workflows/pi-Flux.png" width="600" alt=""/>

## Training Your Own pi-Flow Models

Please visit the official [piFlow](https://github.com/lakonik/piflow) repo for more information on training.
