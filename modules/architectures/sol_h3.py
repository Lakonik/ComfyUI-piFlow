# Copyright (c) 2026 Hansheng Chen
# Adapted from the Apache-2.0 Sol-H3-VAE reference decoder:
# https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE/blob/6a8a33fc61e092281f32bf28cde236760e2caaa4/code/sol_h3_vae/model.py

import math

import torch
from torch import nn

import comfy.model_management
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_masked

# Sol-H3-VAE decoder constants (MiniMax-H3 vae/config.json and the Sol reference decoder)
NUM_HEADS = 32
HEAD_DIM = 64
DIM = NUM_HEADS * HEAD_DIM
FFN_MULT = 4
NUM_LAYERS = 24
NORM_EPS = 1e-5
NUM_REGISTER_TOKENS = 4
ROPE_THETA = 100.0
ROPE_DIM = int(HEAD_DIM * 0.75)
WINDOW = 16  # spatial window in tokens; odd layers shift it by half
BLOCK = 5  # native latent frames per block (17 video frames)
T_SHIFT = 2  # temporal window shift of odd layers
TIME_NORM = 5
PATCH, PATCH_T, OUT_CH = 16, 4, 3
FRAMES_TO_TRIM = 3  # frames dropped from the first latent of every block

# affine RGB color codec of Sol's ImageNet-normalized RGB output -> [-1, 1]
COLOR_MEAN = (-0.03, -0.088, -0.188)
COLOR_STD = (0.458, 0.448, 0.45)


def _rope(T, H, W, s, st, device):
    """cos/sin (1, T*H*W, 1, ROPE_DIM // 2) for tokens in (T, H, W) order with window-local time and space ids
    for shifts (st, s); both halves of the rotary dims share them."""
    lt = ((torch.arange(T, device=device) + st) % BLOCK).float()
    ly = ((torch.arange(H, device=device) + s) % WINDOW).float()
    lx = ((torch.arange(W, device=device) + s) % WINDOW).float()
    t = (lt + 0.5) / TIME_NORM * 2 - 1
    hy = (ly + 0.5) / WINDOW * 2 - 1
    wx = (lx + 0.5) / WINDOW * 2 - 1
    grid = torch.stack(torch.meshgrid(t, hy, wx, indexing='ij'), -1).reshape(1, -1, 3)
    inv_freq = 1.0 / ROPE_THETA ** torch.arange(0, 1, 6 / ROPE_DIM, dtype=torch.float32, device=device)
    angles = 2.0 * math.pi * grid[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).unsqueeze(2)
    return angles.cos(), angles.sin()


def _rotate_(x, emb):
    """Apply rotary embeddings in place to (B, L, heads, head_dim) x."""
    cos, sin = (t.to(x.dtype) for t in emb)
    half = cos.size(-1)
    first, second = x[..., :half], x[..., half:2 * half]
    first_sin, second_sin = first * sin, second * sin
    first.mul_(cos).sub_(second_sin)
    second.mul_(cos).add_(first_sin)


def _partition(x, s, st):
    """x: (B, T, H, W, ...) -> windows (B*nW, BLOCK*WINDOW*WINDOW, ...), valid (nW, BLOCK*WINDOW*WINDOW), info.
    Window order is (time window, row, col)."""
    B, T, H, W = x.shape[:4]
    ws, tb = WINDOW, BLOCK
    rest = x.shape[4:]
    Tp = ((T + st + tb - 1) // tb) * tb
    Hp = ((H + s + ws - 1) // ws) * ws
    Wp = ((W + s + ws - 1) // ws) * ws
    xp = x.new_zeros(B, Tp, Hp, Wp, *rest)
    xp[:, st:st + T, s:s + H, s:s + W] = x
    valid = torch.zeros(Tp, Hp, Wp, dtype=torch.bool, device=x.device)
    valid[st:st + T, s:s + H, s:s + W] = True
    nt, nh, nw = Tp // tb, Hp // ws, Wp // ws
    r = len(rest)
    xw = xp.view(B, nt, tb, nh, ws, nw, ws, *rest).permute(0, 1, 3, 5, 2, 4, 6, *range(7, 7 + r))
    xw = xw.reshape(B * nt * nh * nw, tb * ws * ws, *rest)
    vw = valid.view(nt, tb, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, tb * ws * ws)
    return xw, vw, (B, T, H, W, Tp, Hp, Wp, nt, nh, nw)


def _window_rows(B, T, H, W, nb, S, s, st, device):
    """Padded window attention indices over the B*T*H*W token rows and the (B, nb, S) global-token rows after
    them: query rows, key rows, the key mask and the gather that restores token order. Windows run (batch, time
    window, row, col) and keep their tokens in (t, y, x) order; the keys of a window append the global tokens of
    the native blocks it overlaps."""
    N = T * H * W
    tokens, valid, info = _partition(torch.arange(1, N + 1, device=device).view(1, T, H, W), s, st)
    Tp, Hp, Wp, nt, nh, nw = info[4:]
    # native blocks of the first and last latent of temporal window j, which covers [j*BLOCK - st, (j+1)*BLOCK - st)
    blocks = [(max(j * BLOCK - st, 0) // BLOCK, (min((j + 1) * BLOCK - st, T) - 1) // BLOCK) for j in range(nt)]
    global_rows = torch.tensor([[b * S + i for b in pair for i in range(S)] for pair in blocks], device=device)
    global_valid = torch.tensor([[True] * S + [b1 != b0] * S for b0, b1 in blocks], device=device)
    global_rows, global_valid = (
        t[:, None, None].expand(nt, nh, nw, 2 * S).reshape(-1, 2 * S) for t in (global_rows, global_valid))
    batch = torch.arange(B, device=device)[:, None, None]
    token_rows = batch * N + tokens - 1
    key_rows = torch.cat([token_rows, B * N + batch * nb * S + global_rows], -1)
    key_valid = torch.cat([valid, global_valid], -1)
    output_rows = torch.arange(token_rows.numel(), device=device).view(
        B, nt, nh, nw, BLOCK, WINDOW, WINDOW).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, Tp, Hp, Wp)
    output_rows = output_rows[:, st:st + T, s:s + H, s:s + W].reshape(-1)
    return (token_rows.masked_fill(~valid, 0).flatten(0, 1),
            key_rows.masked_fill(~key_valid, 0).flatten(0, 1),
            key_valid.expand_as(key_rows).flatten(0, 1).contiguous(), output_rows)


def _global_rows(B, T, H, W, nb, S, device):
    """Padded keys of the (B, nb, S) global tokens and their mask: every token of their native block, then the
    block's global tokens."""
    N = T * H * W
    tokens = torch.arange(nb * BLOCK * H * W, device=device).view(nb, -1)
    batch = torch.arange(B, device=device)[:, None, None]
    globals_ = torch.arange(nb * S, device=device).view(nb, S)
    rows = torch.cat([batch * N + tokens, B * N + batch * nb * S + globals_], -1)
    valid = torch.cat([tokens < N, torch.ones(nb, S, dtype=torch.bool, device=device)], -1)
    return rows.masked_fill(~valid, 0).flatten(0, 1), valid.expand_as(rows).flatten(0, 1).contiguous()


def _attention(q, k, v, key_mask):
    """(n, Lq, heads, dh) queries over (n, Lk, heads, dh) keys with a (n, Lk) key mask -> (n, Lq, heads * dh)."""
    return optimized_attention_masked(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), NUM_HEADS,
        mask=key_mask[:, None, None, :], skip_reshape=True)


class SolH3Attention(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.to_qkv = operations.Linear(DIM, 3 * DIM, dtype=dtype, device=device)
        self.to_out = operations.Linear(DIM, DIM, dtype=dtype, device=device)


class SolH3SwiGLU(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.w1 = operations.Linear(DIM, 2 * DIM * FFN_MULT, dtype=dtype, device=device)
        self.w2 = operations.Linear(DIM * FFN_MULT, DIM, dtype=dtype, device=device)

    def forward(self, x):
        x = self.w1(x)
        gate, up = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
        nn.functional.silu(gate, inplace=True).mul_(up)  # in place: the gate half becomes silu(gate) * up
        return self.w2(gate)


class SolH3Block(nn.Module):
    """One Sol decoder layer, updating the tokens x (B, T, H, W, D) and the per-block global tokens g (B, nb, S, D)
    in place."""

    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(DIM, eps=NORM_EPS, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(DIM, eps=NORM_EPS, dtype=dtype, device=device)
        self.scale1 = nn.Parameter(torch.empty(DIM, dtype=dtype, device=device))
        self.scale2 = nn.Parameter(torch.empty(DIM, dtype=dtype, device=device))
        self.attn = SolH3Attention(dtype=dtype, device=device, operations=operations)
        self.ff = SolH3SwiGLU(dtype=dtype, device=device, operations=operations)

    def forward(self, x, g, shifted, global_indices, window_indices):
        self._attention(x, g, shifted, global_indices, window_indices)
        scale2 = comfy.model_management.cast_to(self.scale2, dtype=x.dtype, device=x.device)
        x.addcmul_(self.ff(self.norm2(x)), scale2)
        g.addcmul_(self.ff(self.norm2(g)), scale2)

    def _attention(self, x, g, shifted, global_indices, window_indices):
        """Window attention of the tokens (each window also sees the global tokens of the native blocks it
        overlaps) and global attention of the global tokens over their native block, with the residual."""
        B, T, H, W, _ = x.shape
        nb, S = g.shape[1:3]
        N = B * T * H * W
        s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
        # token rows, then global-token rows: one gather packs the queries or keys of either attention
        qkv = self.attn.to_qkv(self.norm1(torch.cat([x.flatten(0, 3), g.flatten(0, 2)])))
        qkv = qkv.view(-1, NUM_HEADS, 3, HEAD_DIM)
        q = nn.functional.rms_norm(qkv[:, :, 0], (HEAD_DIM,), eps=NORM_EPS)
        k = nn.functional.rms_norm(qkv[:, :, 1], (HEAD_DIM,), eps=NORM_EPS)
        v = qkv[:, :, 2]
        rope = _rope(T, H, W, s, st, x.device)
        _rotate_(q[:N].view(B, -1, NUM_HEADS, HEAD_DIM), rope)
        _rotate_(k[:N].view(B, -1, NUM_HEADS, HEAD_DIM), rope)
        rows, key_mask = global_indices
        og = _attention(q[N:].unflatten(0, (B * nb, S)), k[rows], v[rows], key_mask).flatten(0, 1)
        q_rows, k_rows, key_mask, output_rows = window_indices
        # v is a view of qkv: gather it first so qkv dies before the larger padded windows of q and k exist
        v = v[k_rows]
        del qkv
        q = q[q_rows]
        k = k[k_rows]
        o = _attention(q, k, v, key_mask)
        del q, k, v
        o = o.flatten(0, 1)[output_rows]
        scale1 = comfy.model_management.cast_to(self.scale1, dtype=x.dtype, device=x.device)
        x.addcmul_(self.attn.to_out(o).view_as(x), scale1)
        g.addcmul_(self.attn.to_out(og).view_as(g), scale1)


class SolH3DWDHead(nn.Module):
    """DWD pixel refiner made of the last Sol-H3-VAE decoder layers, final norm and pixel projection.

    A full-rank linear expansion maps every token onto 16-pixel Sol tokens, one per latent frame; the whole latent
    sequence is refined at once with Sol attention (shifted 5-latent x 16x16-token windows plus per-block global
    tokens).
    """

    def __init__(self, in_channels, pixel_patch_size, num_layers, dtype=None, device=None, operations=None):
        super().__init__()
        self.expansion = (pixel_patch_size[0] // PATCH, pixel_patch_size[1] // PATCH)
        self.feature_expand = operations.Linear(
            in_channels, DIM * self.expansion[0] * self.expansion[1], dtype=dtype, device=device)
        self.register_tokens = nn.Parameter(torch.empty(1, NUM_REGISTER_TOKENS, DIM, dtype=dtype, device=device))
        self.blocks = nn.ModuleList([
            SolH3Block(dtype=dtype, device=device, operations=operations) for _ in range(num_layers)])
        self.norm_out = operations.LayerNorm(DIM, eps=NORM_EPS, dtype=dtype, device=device)
        self.proj_out = operations.Linear(DIM, OUT_CH * PATCH_T * PATCH * PATCH, dtype=dtype, device=device)
        self.first_frame_proj_out = operations.Linear(
            DIM, OUT_CH * (PATCH_T - FRAMES_TO_TRIM) * PATCH * PATCH, dtype=dtype, device=device)

    @classmethod
    def from_state_dict(cls, sd, in_channels, pixel_patch_size, dtype=None, device=None, operations=None):
        num_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        return cls(in_channels, pixel_patch_size, num_layers, dtype=dtype, device=device, operations=operations)

    def expand(self, features, token_grid_size):
        """(B, t * h * w, in_channels) features on a (t, h, w) token grid -> Sol tokens (B, T, H, W, D)."""
        B = features.size(0)
        t, h, w = token_grid_size
        e_h, e_w = self.expansion
        x = self.feature_expand(features)
        return x.view(B, t, h, w, e_h, e_w, DIM).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, t, h * e_h, w * e_w, DIM)

    def refine(self, x):
        """Refine Sol tokens (B, T, H, W, D) in place into pixels in [-1, 1], (B, F, 3, 16H, 16W), on the
        intermediate device, through Sol's affine RGB color codec."""
        B, T, H, W, _ = x.shape
        register_tokens = comfy.model_management.cast_to(self.register_tokens, dtype=x.dtype, device=x.device)
        g = torch.cat([register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, DIM)], 1)
        g = g.unsqueeze(1).expand(B, (T + BLOCK - 1) // BLOCK, -1, -1).contiguous()
        nb, S = g.shape[1:3]
        global_indices = _global_rows(B, T, H, W, nb, S, x.device)
        window_indices = {
            shifted: _window_rows(B, T, H, W, nb, S, *((WINDOW // 2, T_SHIFT) if shifted else (0, 0)), x.device)
            for shifted in (False, True)}
        for index, block in enumerate(self.blocks, NUM_LAYERS - len(self.blocks)):
            block(x, g, index % 2 == 1, global_indices, window_indices[index % 2 == 1])
        del g
        x = self.norm_out(x)

        # every native block of BLOCK latents covers a trimmed first latent and full later latents
        first_t = PATCH_T - FRAMES_TO_TRIM
        num_chunks, tail = divmod(T, BLOCK)
        chunk_frames = first_t + (BLOCK - 1) * PATCH_T
        tail_frames = first_t + (tail - 1) * PATCH_T if tail else 0
        out = x.new_empty(B, num_chunks * chunk_frames + tail_frames, OUT_CH, H * PATCH, W * PATCH)
        if num_chunks:
            self._project_pixels(
                x[:, :num_chunks * BLOCK].unflatten(1, (num_chunks, BLOCK)),
                out[:, :num_chunks * chunk_frames].unflatten(1, (num_chunks, chunk_frames)))
        if tail:
            self._project_pixels(x[:, num_chunks * BLOCK:].unsqueeze(1), out[:, num_chunks * chunk_frames:].unsqueeze(1))
        del x
        pixels = out.to(comfy.model_management.intermediate_device())
        del out
        pixels = pixels.float()
        mean = torch.tensor(COLOR_MEAN).view(-1, 1, 1).to(pixels.device)
        std = torch.tensor(COLOR_STD).view(-1, 1, 1).to(pixels.device)
        return pixels.mul_(std).add_(mean).clamp_(-1, 1)

    def _project_pixels(self, x, out):
        """Project chunks (B, n, t, H, W, D) of Sol tokens into their pixel frames out (B, n, F, 3, 16H, 16W)."""
        B, n, t, H, W, _ = x.shape
        first_t = PATCH_T - FRAMES_TO_TRIM
        out[:, :, :first_t].unflatten(-2, (H, PATCH)).unflatten(-1, (W, PATCH)).copy_(
            self.first_frame_proj_out(x[:, :, :1]).view(B, n, H, W, OUT_CH, first_t, PATCH, PATCH).permute(
                0, 1, 5, 4, 2, 6, 3, 7))
        if t > 1:
            # over every latent of the chunk: slicing off the first would copy the input
            rest = self.proj_out(x).view(B, n, t, H, W, OUT_CH, PATCH_T, PATCH, PATCH)[:, :, 1:]
            out[:, :, first_t:].unflatten(2, (t - 1, PATCH_T)).unflatten(-2, (H, PATCH)).unflatten(-1, (W, PATCH)).copy_(
                rest.permute(0, 1, 2, 6, 5, 3, 7, 4, 8))
