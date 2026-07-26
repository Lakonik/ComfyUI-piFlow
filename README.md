# ComfyUI Nodes for AsymFlow and pi-Flow

**ComfyUI-piFlow** is a collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implement the following model families:
- **AsymFlow**
  <br>
  Generates realistic images in pixel space. 
  <br>
  **[[Workflows](docs/AsymFlow.md)]**

  <img src="https://raw.githubusercontent.com/Lakonik/LakonLab/refs/heads/main/docs/assets/asymflow/asymflow_teaser.jpg" width=250 alt=""/>

- **pi-Flow**
  <br>
  Fast few-step generation.
  <br>
  **[[Workflows](docs/piFlow.md)]**

  <img src="https://raw.githubusercontent.com/Lakonik/LakonLab/refs/heads/main/docs/assets/piflow/piflow_teaser.jpg" width=250 alt=""/>

## Installation

**This extension requires ComfyUI version 0.23.0 or higher**. Older ComfyUI releases are no longer supported.

**Please uninstall any non-official AsymFlow or pi-Flow extensions before installing this extension**, as they may cause compatibility issues.

### ComfyUI Manager

If you are using [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager), you can load a [workflow](#workflows) first, and then install the missing nodes via ComfyUI Manager.

### Manual Installation

For manual installation, simply clone this repo into your ComfyUI `custom_nodes` directory.
```bash
# run the following command in your ComfyUI `custom_nodes` directory
git clone https://github.com/Lakonik/ComfyUI-piFlow
```

## License

This code repository is licensed under the Apache-2.0 License. Models used in the workflows are subject to their own respective licenses.

## Changelog

- **v1.3.3** (2026-07-25)
  - Require ComfyUI v0.23.0 or newer and fix pi-Flow sampling with the current `ModelPatcher.prepare_state` API ([#36](https://github.com/Lakonik/ComfyUI-piFlow/issues/36), [#37](https://github.com/Lakonik/ComfyUI-piFlow/issues/37)).

- **v1.3.2** (2026-05-23)
  - Fix AsymFLUX.2 patchification rounding.

- **v1.3.1** (2026-05-20)
  - Fix AsymFLUX.2 VRAM predictor ([#32](https://github.com/Lakonik/ComfyUI-piFlow/pull/32)).

- **v1.3.0** (2026-05-20)
  - Add AsymFlow nodes and workflows for AsymFLUX.2 klein 9B.

- **v1.2.0** (2026-05-17)
  - Target latest ComfyUI only (`requires-comfyui >= 0.17.0`); older releases are no longer supported.
  - Update Qwen, Flux, and Flux.2 model shims for current ComfyUI reference/edit conditioning APIs.
  - Fix Flux/Flux.2 loader compatibility with current normalization key names and latest model loading behavior.
  - Add sampler support for multiple active/regional pi-Flow conditionings through composite policy blending.

- **v1.1.5** (2026-01-18)
  - Fix a compatibility issue with ComfyUI-GGUF commit `58625e1`.
  - Add support for loading metadata from GGUF pi-Flow models.

- **v1.1.4** (2025-12-18)
  - Fix a bug in the example pi-Flux.2 editing workflow where a load image node is disconnected from the main graph. 

- **v1.1.3** (2025-12-18)
  - Add pi-Flux.2 models and workflow for text-to-image generation and multi-image editing.
  - Add GGUF support for pi-Flow models.
  - Fix compatibility for ComfyUI v0.4.0 (new quantization)
  - Fix dtype mismatch issues in GMFlow output layers
  - Improve GMFlow numerical stability

- **v1.0.5** (2025-11-11)
  - Add experimental support for polynomial-based DX policy.
  - Update README.md and pi-Flux workflow (highlighting the FluxGuidance setting).

- **v1.0.4** (2025-11-09)
  - Fix a bug in GM-Qwen when running in BF16 precision.

- **v1.0.3** (2025-11-09)
  - Add support for scaled FP8 base models.
