# Copyright (c) 2026 Hansheng Chen

import weakref

import torch
from torch import nn

import comfy.model_patcher
import comfy.ops
import comfy.patcher_extension

from .dwd_head import DWD_PIXELS_KEY, DWDPixelHead, dwd_outer_sample_wrapper

# a MiniMax H3 video token is 1x2x2 latents of the 16x VAE: 32x32 pixels per latent frame
PIXEL_PATCH_SIZE = (32, 32)


class MiniMaxH3DWDNormOut(nn.Module):
    """The DWD output norm: affine-free LayerNorm modulated by (scale, shift) = linear(silu(t_emb))."""

    def __init__(self, hidden, t_dim, eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.eps = eps
        self.linear = operations.Linear(t_dim, 2 * hidden, dtype=dtype, device=device)

    def forward(self, x, t_emb, row):
        scale, shift = self.linear(nn.functional.silu(t_emb)).to(x.dtype).chunk(2, dim=-1)
        return nn.functional.layer_norm(x, x.shape[-1:], eps=self.eps).mul_(1 + scale[row]).add_(shift[row])


class MiniMaxH3DWD(DWDPixelHead):
    """The DWD output norm and pixel head of a LakonLab DWD adapter on MiniMax H3.

    At the last sampling step, the final hidden states of the target video rows decode directly to pixels.
    """

    def __init__(self, dwd_sd, eps, dtype=None, device=None, operations=None):
        hidden, t_dim = dwd_sd["dwd_norm_out.linear.weight"].shape
        hidden //= 2
        super().__init__(dwd_sd, hidden, PIXEL_PATCH_SIZE, dtype=dtype, device=device, operations=operations)
        self.dwd_norm_out = MiniMaxH3DWDNormOut(hidden, t_dim, eps, dtype=dtype, device=device, operations=operations)

    def decode(self, h, t_emb, mod_segments, layout):
        """The (F, H, W, 3) IMAGE in [0, 1], fp16 on the intermediate device, of the final hidden states h."""
        _, latent_t, latent_h, latent_w, _ = layout.signature
        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        row = next(r for a, b, r in mod_segments if (a, b) == (va, vb)) // 3  # block mod rows are timestep * 3 + tag
        # the features go straight in, so the head can release them once expanded
        pixels = self.forward_dwd_head(
            self.dwd_norm_out(h[va:vb], t_emb, row)[None], (latent_t, latent_h // 2, latent_w // 2))[0]
        return pixels.add_(1).mul_(0.5).permute(0, 2, 3, 1).half()


def patch_minimax_h3_dwd(model, dwd_sd, load_module):
    """Install the DWD head of a pi-H3 DWD adapter on a clone of a native MiniMax H3 model.

    The head is its own ModelPatcher, registered as an additional model so that ComfyUI loads and offloads it with
    the denoiser while the denoiser's own weights stay untouched. ComfyUI's ``_run_blocks`` is the only place
    holding the final hidden states, so it is wrapped to decode them with the head at the last step of the
    sampling schedule, on the positive pass only, into the holder the outer-sample wrapper gave this sampling run.

    ComfyUI sets object patches on the shared denoiser, where the wrapper can outlive its patchers, so it holds the head
    only weakly: the head is freed with the last of its patchers.
    """
    dwd = MiniMaxH3DWD(
        dwd_sd, model.model.diffusion_model.final_layer.norm.eps, dtype=dwd_sd["dwd_norm_out.linear.weight"].dtype,
        device=model.offload_device, operations=comfy.ops.mixed_precision_ops())
    load_module(dwd, dwd_sd)
    head = comfy.model_patcher.CoreModelPatcher(dwd, load_device=model.load_device, offload_device=model.offload_device)

    model = model.clone()
    model.set_additional_models("lakonlab_dwd", [head])
    run_blocks = model.get_model_object("diffusion_model._run_blocks")
    dwd_ref = weakref.ref(dwd)

    def run_blocks_and_decode(h, t_emb, mod_segments, layout, transformer_options):
        h = run_blocks(h, t_emb, mod_segments, layout, transformer_options)
        holder = transformer_options.get(DWD_PIXELS_KEY)
        sigmas = transformer_options.get("sigmas")
        sample_sigmas = transformer_options.get("sample_sigmas")
        if holder is not None and transformer_options.get("cond_or_uncond") == [0] and torch.isclose(
                sigmas.flatten()[0].float(), sample_sigmas[sample_sigmas > 0][-1].float().to(sigmas.device)):
            holder.pixels = dwd_ref().decode(h, t_emb, mod_segments, layout)
        return h

    model.add_object_patch("diffusion_model._run_blocks", run_blocks_and_decode)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "lakonlab_dwd", dwd_outer_sample_wrapper)
    return model
