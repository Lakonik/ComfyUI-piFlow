import torch
import torch.nn as nn
from comfy.ldm.flux.layers import apply_mod, timestep_embedding
from comfy.ldm.flux.model import Flux


class LastLayer(nn.Module):
    """Modulation only, no linear projection to out_channels."""

    def __init__(self, hidden_size: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype,
                                               device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype,
                                                                device=device))

    def forward(self, x, vec, modulation_dims=None):
        if vec.ndim == 2:
            vec = vec[:, None, :]

        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = apply_mod(self.norm_final(x), (1 + scale), shift, modulation_dims)
        return x


class GMFlux(Flux):
    def __init__(
            self,
            *args,
            num_gaussians=8,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            final_layer=True,
            dtype=None,
            device=None,
            operations=None,
            **kwargs):
        super().__init__(
            *args, final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)

        if final_layer:
            self.num_gaussians = num_gaussians
            self.constant_logstd = constant_logstd
            self.logstd_inner_dim = logstd_inner_dim
            self.gm_num_logstd_layers = gm_num_logstd_layers

            self.final_layer = LastLayer(
                self.hidden_size, dtype=dtype, device=device, operations=operations)
            self.proj_out_means = operations.Linear(
                self.hidden_size, self.num_gaussians * self.out_channels,
                bias=True, device=device)
            self.proj_out_logweights = operations.Linear(
                self.hidden_size, self.num_gaussians * self.patch_size * self.patch_size,
                bias=True, device=device)
            self.constant_logstd = constant_logstd

            if self.constant_logstd is None:
                assert gm_num_logstd_layers >= 1
                in_dim = self.hidden_size
                logstd_layers = []
                for _ in range(gm_num_logstd_layers - 1):
                    logstd_layers.extend([
                        nn.SiLU(),
                        operations.Linear(in_dim, logstd_inner_dim, bias=True, device=device)])
                    in_dim = logstd_inner_dim
                self.proj_out_logstds = nn.Sequential(
                    *logstd_layers,
                    nn.SiLU(),
                    operations.Linear(in_dim, 1, bias=True, device=device))

    def forward_orig(
            self,
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance=None,
            control=None,
            transformer_options={},
            attn_mask=None):

        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        txt = self.txt_in(txt)

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"),
                                                   transformer_options=args.get("transformer_options"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask,
                                 transformer_options=transformer_options)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:, :add.shape[1]] += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"),
                                       transformer_options=args.get("transformer_options"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:txt.shape[1] + add.shape[1], ...] += add

        img = img[:, txt.shape[1]:, ...]

        hidden_states = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return hidden_states, vec

    def _forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None,
                 transformer_options={}, **kwargs):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        img, img_ids = self.process_img(x)
        img_tokens = img.shape[1]
        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get("ref_latents_method", "offset")
            for ref in ref_latents:
                if ref_latents_method == "index":
                    index += 1
                    h_offset = 0
                    w_offset = 0
                elif ref_latents_method == "uxo":
                    index = 0
                    h_offset = h_len * patch_size + h
                    w_offset = w_len * patch_size + w
                    h += ref.shape[-2]
                    w += ref.shape[-1]
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        hidden_states, vec = self.forward_orig(
            img, img_ids, context, txt_ids, timestep, y, guidance, control,
            transformer_options, attn_mask=kwargs.get("attention_mask", None))
        hidden_states = hidden_states[:, :img_tokens].float()

        bs = hidden_states.size(0)
        k = self.num_gaussians
        c = self.out_channels
        out_means = self.proj_out_means(hidden_states).reshape(
            bs, h_len, w_len, k, c // (patch_size * patch_size), patch_size, patch_size
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, c // (patch_size * patch_size), h_len * patch_size, w_len * patch_size
        )[..., :h_orig, :w_orig]
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, h_len, w_len, k, 1, patch_size, patch_size
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, 1, h_len * patch_size, w_len * patch_size
        )[..., :x.shape[-2], :x.shape[-1]].log_softmax(dim=1)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(vec.detach().float()).reshape(bs, 1, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1, 1), float(self.constant_logstd))

        return dict(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)
