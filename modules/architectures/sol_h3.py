# Copyright (c) 2026 Hansheng Chen

import torch

import comfy.model_management
import comfy.ops
from comfy.ldm.minimax.vae import NonCausalViT3DDecoder
from comfy.ldm.modules.attention import optimized_attention

PATCH, PATCH_T, OUT_CH = 16, 4, 3
FRAMES_TO_TRIM = 3  # frames dropped from the first latent of every native block

# affine RGB color codec of Sol's ImageNet-normalized RGB output -> [-1, 1]
COLOR_MEAN = (-0.03, -0.088, -0.188)
COLOR_STD = (0.458, 0.448, 0.45)


class SolH3DWDHead(NonCausalViT3DDecoder):
    """DWD pixel refiner made of the last Sol-H3-VAE decoder layers, final norm and pixel projection, run by
    ComfyUI's native Sol decoder (block-major rows, layers streamed over native 5-latent blocks).

    In place of the decoder's latent embedding, a full-rank linear expansion maps every denoiser token onto
    16-pixel Sol tokens, one per latent frame. The first latent of every native block has its own one-frame
    projection.
    """

    def __init__(self, in_channels, pixel_patch_size, num_layers, dtype=None, device=None, operations=None):
        super().__init__(num_layers=num_layers, operations=operations)
        del self.x_embedder  # the tokens come from feature_expand
        self.expansion = (pixel_patch_size[0] // PATCH, pixel_patch_size[1] // PATCH)
        self.feature_expand = operations.Linear(
            in_channels, self.dim * self.expansion[0] * self.expansion[1], dtype=dtype, device=device)
        self.first_frame_proj_out = operations.Linear(
            self.dim, OUT_CH * (PATCH_T - FRAMES_TO_TRIM) * PATCH * PATCH, dtype=dtype, device=device)

    @classmethod
    def from_state_dict(cls, sd, in_channels, pixel_patch_size, dtype=None, device=None, operations=None):
        num_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        return cls(in_channels, pixel_patch_size, num_layers, dtype=dtype, device=device, operations=operations)

    def expand(self, features, token_grid_size):
        """(1, t * h * w, in_channels) features on a (t, h, w) token grid -> the Sol tokens in the decoder's
        block-major rows (global rows initialized), with the geometry tables of their (T, H, W) latent shape."""
        t, h, w = token_grid_size
        e_h, e_w = self.expansion
        T, H, W = t, h * e_h, w * e_w
        D, S = self.dim, self.num_global
        geo = self._build_geometry(T, H, W)
        rows = geo["rows"]
        x = torch.empty(rows[-1][1], D, dtype=features.dtype, device=features.device)
        frames = features[0].view(t, h * w, -1)
        for b, (r0, r1) in enumerate(rows):
            # a block's rows hold its tokens in (t, y, x) order, then its global rows
            f = frames[b * self.block:(b + 1) * self.block]
            x[r0:r1 - S] = self.feature_expand(f).view(-1, h, w, e_h, e_w, D).permute(0, 1, 3, 2, 4, 5).reshape(-1, D)
        x[geo["glob"].to(x.device)] = torch.cat(
            [comfy.ops.cast_to_input(self.register_tokens[0], x), x.new_zeros(1, D)], 0).repeat(len(rows), 1)
        return x, geo, (T, H, W)

    def refine(self, x, geo, latent_shape):
        """Refine expanded Sol tokens in place into pixels in [-1, 1], (1, F, 3, 16H, 16W), on the intermediate
        device. The layer loop is ComfyUI's ``NonCausalViT3DDecoder.iter_decode``."""
        T, H, W = latent_shape
        heads, dh, D, S, R = self.heads, self.dim_head, self.dim, self.num_global, self.ring
        dev = x.device
        rows, slot_rows = geo["rows"], geo["slot_rows"]
        gslot = geo["gslot"].to(dev)
        shifts = [(sh["ids"].to(dev), [[(qi.to(dev), ki.to(dev)) for qi, ki in wins] for wins in sh["windows_at"]],
                   sh["blocks_at"]) for sh in geo["shifts"]]
        q, k, v = (torch.empty(R * slot_rows, heads, dh, dtype=x.dtype, device=dev) for _ in range(3))
        attn_out = torch.empty(R * slot_rows, D, dtype=x.dtype, device=dev)
        for li, blk in enumerate(self.blocks):
            attn, (ids, windows_at, blocks_at) = blk.attn, shifts[li % 2]
            for b, (r0, r1) in enumerate(rows):
                s0 = (b % R) * slot_rows
                s1 = s0 + r1 - r0
                qkv = comfy.ops.linear_input_act(attn.to_qkv, x[r0:r1], "rms_norm", act_weight=blk.norm1)
                qq, kk, vv = qkv.view(1, r1 - r0, heads, 3 * dh).chunk(3, -1)
                qq, kk = attn.rms_rope(qq, kk, self.pos_embed(ids[None, r0:r1].to(x.dtype)))
                q[s0:s1], k[s0:s1], v[s0:s1] = qq[0], kk[0], vv[0]
                for qi, ki in windows_at[b]:
                    attn_out[qi] = attn.attention(*(t.transpose(1, 2) for t in (q[qi], k[ki], v[ki])))
                gs = gslot[b]
                attn_out[gs] = optimized_attention(*(t[None].transpose(1, 2) for t in (q[gs], k[s0:s1], v[s0:s1])),
                                                   heads, skip_reshape=True)[0]
                for c in blocks_at[b]:
                    c0, c1 = rows[c]
                    cs = (c % R) * slot_rows
                    x[c0:c1] = blk.post_attention(attn_out[cs:cs + c1 - c0], x[c0:c1])
        del q, k, v, attn_out, shifts

        first_t = PATCH_T - FRAMES_TO_TRIM
        num_frames = sum(first_t + ((r1 - r0 - S) // (H * W) - 1) * PATCH_T for r0, r1 in rows)
        pixels = torch.empty(
            num_frames, OUT_CH, H * PATCH, W * PATCH, device=comfy.model_management.intermediate_device())
        f0 = 0
        for r0, r1 in rows:
            n = (r1 - r0 - S) // (H * W)
            y = self.norm_out(x[r0:r1 - S]).view(n, H, W, D)
            out = [self.first_frame_proj_out(y[:1]).view(H, W, OUT_CH, first_t, PATCH, PATCH).permute(
                3, 2, 0, 4, 1, 5).reshape(first_t, OUT_CH, H * PATCH, W * PATCH)]
            if n > 1:
                out.append(self.proj_out(y[1:]).view(n - 1, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(
                    0, 4, 3, 1, 5, 2, 6).reshape((n - 1) * PATCH_T, OUT_CH, H * PATCH, W * PATCH))
            out = torch.cat(out)
            pixels[f0:f0 + out.shape[0]].copy_(out)
            f0 += out.shape[0]
        mean = torch.tensor(COLOR_MEAN, device=pixels.device).view(-1, 1, 1)
        std = torch.tensor(COLOR_STD, device=pixels.device).view(-1, 1, 1)
        return pixels.mul_(std).add_(mean).clamp_(-1, 1)[None]
