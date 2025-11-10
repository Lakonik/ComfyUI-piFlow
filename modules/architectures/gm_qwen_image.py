import torch
import torch.nn as nn
import comfy.ldm.qwen_image.model
from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel, LastLayer


class GMQwenImageTransformer2DModel(QwenImageTransformer2DModel):
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

            self.norm_out = LastLayer(
                self.inner_dim, self.inner_dim,
                dtype=dtype, device=device, operations=operations)
            self.proj_out_means = operations.Linear(
                self.inner_dim, self.num_gaussians * self.out_channels,
                bias=True, device=device)
            self.proj_out_logweights = operations.Linear(
                self.inner_dim, self.num_gaussians * self.patch_size * self.patch_size,
                bias=True, device=device)
            self.constant_logstd = constant_logstd

            if self.constant_logstd is None:
                assert gm_num_logstd_layers >= 1
                in_dim = self.inner_dim
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

    def _forward(
            self,
            x,
            timesteps,
            context,
            attention_mask=None,
            guidance: torch.Tensor = None,
            ref_latents=None,
            transformer_options={},
            control=None,
            **kwargs):
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
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

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids)
        if not hasattr(comfy.ldm.qwen_image.model, 'apply_rope1'):  # old version check
            image_rotary_emb = image_rotary_emb.squeeze(1).unsqueeze(2)
        image_rotary_emb = image_rotary_emb.to(x.dtype).contiguous()

        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})

        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        hidden_states = self.norm_out(hidden_states, temb)[:, :num_embeds].float()

        bs = hidden_states.size(0)
        k = self.num_gaussians
        c = self.out_channels  # should be the same as self.in_channels
        h, w = orig_shape[-2] // self.patch_size, orig_shape[-1] // self.patch_size
        out_means = self.proj_out_means(hidden_states).reshape(
                bs, h, w, k, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size
            ).permute(
                0, 3, 4, 1, 5, 2, 6
            ).reshape(
                bs, k, c // (self.patch_size * self.patch_size), 1, h * self.patch_size, w * self.patch_size
            )[..., :x.shape[-2], :x.shape[-1]]
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
                bs, h, w, k, 1, self.patch_size, self.patch_size
            ).permute(
                0, 3, 4, 1, 5, 2, 6
            ).reshape(
                bs, k, 1, 1, h * self.patch_size, w * self.patch_size
            )[..., :x.shape[-2], :x.shape[-1]].log_softmax(dim=1)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(temb.detach().float()).reshape(bs, 1, 1, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1, 1, 1), float(self.constant_logstd))

        return dict(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)
