# Copyright (c) 2026 Hansheng Chen

import collections
import hashlib

import torch
from torch import nn

import comfy.utils

from .sol_h3 import SolH3DWDHead

# pixel refiners of LakonLab DWD adapters, each identified by a tensor only it has (under dwd_pixel_refiner.)
DWD_REFINERS = (
    ("register_tokens", SolH3DWDHead),
)


class DWDPixelHead(nn.Module):
    """The pixel head of a LakonLab DWD adapter (LakonLab's DWDHeadMixin), built from the adapter's tensors.

    The modules keep their adapter names. The head maps post-norm final-layer features on a (t, h, w) token grid,
    each token covering ``pixel_patch_size`` pixels, to pixels in [-1, 1]. A host model subclasses it and supplies
    the features.
    """

    def __init__(self, dwd_sd, in_channels, pixel_patch_size, dtype=None, device=None, operations=None):
        super().__init__()
        if any(k.startswith(("dwd_linear_head.", "first_frame_dwd_linear_head.")) for k in dwd_sd):
            raise NotImplementedError("DWD linear heads are not supported yet")
        refiner_sd = {k[len("dwd_pixel_refiner."):]: v for k, v in dwd_sd.items() if k.startswith("dwd_pixel_refiner.")}
        refiner_cls = next((cls for key, cls in DWD_REFINERS if key in refiner_sd), None)
        if refiner_cls is None:
            raise ValueError("Unknown DWD pixel refiner: {}".format(sorted(refiner_sd)[:8]))
        self.dwd_pixel_refiner = refiner_cls.from_state_dict(
            refiner_sd, in_channels, pixel_patch_size, dtype=dtype, device=device, operations=operations)

    def forward_dwd_head(self, features, token_grid_size):
        """(B, t * h * w, in_channels) features -> (B, F, 3, H, W) pixels in [-1, 1] on the intermediate device.

        The features are released once expanded: pass them without keeping a reference.
        """
        expanded = self.dwd_pixel_refiner.expand(features, token_grid_size)
        del features
        return self.dwd_pixel_refiner.refine(*expanded)


# DWD pixels of the latest sampling runs, by the fingerprint of the video latent each run returned
DWD_PIXELS = collections.OrderedDict()
MAX_DWD_PIXELS = 4
DWD_PIXELS_KEY = "lakonlab_dwd_pixels"


class DWDPixelsHolder:
    """The decoded pixels of one sampling run. Not a dict, which ComfyUI copies with the model options."""

    def __init__(self):
        self.pixels = None


def latent_fingerprint(video_latent):
    return hashlib.sha1(video_latent.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def dwd_outer_sample_wrapper(executor, noise, latent_image, sampler, sigmas, *args, latent_shapes=None, **kwargs):
    """Give every sampling run of a DWD model its own holder for the decoded pixels, and file the pixels under the
    run's output video latent for the DWD decode node. Any ComfyUI sampler and guider run through here."""
    holder = DWDPixelsHolder()
    executor.class_obj.model_options.setdefault("transformer_options", {})[DWD_PIXELS_KEY] = holder
    output = executor(noise, latent_image, sampler, sigmas, *args, latent_shapes=latent_shapes, **kwargs)
    if holder.pixels is not None:
        video = output if latent_shapes is None or len(latent_shapes) < 2 else comfy.utils.unpack_latents(
            output, latent_shapes)[0]
        DWD_PIXELS[latent_fingerprint(video)] = holder.pixels
        while len(DWD_PIXELS) > MAX_DWD_PIXELS:
            DWD_PIXELS.popitem(last=False)
    return output


def get_dwd_pixels(samples):
    """The DWD pixels of the sampling run that returned the LATENT ``samples`` (joint audio-video or video)."""
    latent = samples["samples"]
    video = latent.unbind()[0] if latent.is_nested else latent
    pixels = DWD_PIXELS.get(latent_fingerprint(video))
    if pixels is None:
        raise RuntimeError("No DWD pixels for this latent: connect the sampler's output from a model patched with "
                           "a LakonLab DWD adapter (the latest {} runs are kept)".format(MAX_DWD_PIXELS))
    return pixels.float()
