import collections
import numpy as np
import torch
import comfy
import comfy.sampler_helpers
import comfy.patcher_extension
import comfy.model_patcher
from typing import Optional
from comfy.samplers import Sampler, CFGGuider, finalize_default_conds, can_concat_cond, get_area_and_mult, cond_cat
from comfy import model_management
from tqdm.auto import trange
from .model_base import BasePiFlow
from .piflow_policies import GMFlowPolicy
from .piflow_policies.base import BasePolicy


def calc_cond_batch(model: BasePiFlow, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options: dict[str]):
    handler: comfy.context_windows.ContextHandlerABC = model_options.get("context_handler", None)
    if handler is None or not handler.should_use_context(model, conds, x_in, timestep, model_options):
        return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
    return handler.execute(_calc_cond_batch_outer, model, conds, x_in, timestep, model_options)


def _calc_cond_batch_outer(model: BasePiFlow, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _calc_cond_batch,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True)
    )
    return executor.execute(model, conds, x_in, timestep, model_options)


def _calc_cond_batch(model: BasePiFlow, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    out_conds = []
    # separate conds by matching hooks
    hooked_to_run: dict[comfy.hooks.HookGroup, list[tuple[tuple, int]]] = {}
    default_conds = []
    has_default_conds = False

    for i in range(len(conds)):
        cond = conds[i]
        default_c = []
        if cond is not None:
            for x in cond:
                if 'default' in x:
                    default_c.append(x)
                    has_default_conds = True
                    continue
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                if p.hooks is not None:
                    model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
                hooked_to_run.setdefault(p.hooks, list())
                hooked_to_run[p.hooks] += [(p, i)]
        default_conds.append(default_c)

    if has_default_conds:
        finalize_default_conds(model, hooked_to_run, default_conds, x_in, timestep, model_options)

    model.current_patcher.prepare_state(timestep)

    # run every hooked_to_run separately
    for hooks, to_run in hooked_to_run.items():
        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            free_memory = model_management.get_free_memory(x_in.device)
            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
                cond_shapes = collections.defaultdict(list)
                for tt in batch_amount:
                    for k, v in to_run[tt][0].conditioning.items():
                        cond_shapes[k].append(v.size())

                if model.memory_required(input_shape, cond_shapes=cond_shapes) * 1.5 < free_memory:
                    to_batch = batch_amount
                    break

            input_x = []
            c = []
            cond_or_uncond = []
            uuids = []
            control = None
            patches = None
            for x in to_batch:
                o = to_run.pop(x)
                p = o[0]
                input_x.append(p.input_x)
                c.append(p.conditioning)
                cond_or_uncond.append(o[1])
                uuids.append(p.uuid)
                control = p.control
                patches = p.patches

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)

            transformer_options = model.current_patcher.apply_hooks(hooks=hooks)
            if 'transformer_options' in model_options:
                transformer_options = comfy.patcher_extension.merge_nested_dicts(
                    transformer_options, model_options['transformer_options'], copy_dict1=False)

            if patches is not None:
                transformer_options["patches"] = comfy.patcher_extension.merge_nested_dicts(
                    transformer_options.get("patches", {}),
                    patches
                )

            transformer_options["cond_or_uncond"] = cond_or_uncond[:]
            transformer_options["uuids"] = uuids[:]
            transformer_options["sigmas"] = timestep

            c['transformer_options'] = transformer_options

            if control is not None:
                c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond), transformer_options)

            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond})
            else:
                output = model.apply_model(input_x, timestep_, **c)

            out_conds.append(output)

    assert len(out_conds) == 1
    return out_conds[0]  # return a single policy


def sampling_function(model, x, timestep, cond, model_options={}, seed=None):
    return calc_cond_batch(model, [cond], x, timestep, model_options)


class _PolicySampler(CFGGuider):

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(
            self.inner_model, x, timestep, self.conds.get("positive", None), model_options=model_options, seed=seed)


class _PiFlowSampler(Sampler):

    def __init__(
            self, model_sampling, h=0.0, substeps=128, gm_temperature=1.0):
        self.model_sampling = model_sampling
        self.h = h
        self.substeps = substeps
        self.gm_temperature = gm_temperature

    def calculate_sigmas_dst(self, sigmas, eps=1e-6):
        sigmas_src = sigmas[:-1]
        sigmas_to = sigmas[1:]
        alphas_src = 1 - sigmas_src
        alphas_to = 1 - sigmas_to
        if self.h <= 0.0:
            m = torch.ones_like(sigmas_src)
        else:
            h2 = self.h * self.h
            m = (sigmas_to * alphas_src / (sigmas_src * alphas_to).clamp(min=eps)) ** h2

        sigmas_to_mul_m = sigmas_to * m
        sigmas_dst = sigmas_to_mul_m / (alphas_to + sigmas_to_mul_m).clamp(min=eps)

        return sigmas_dst, m

    def policy_rollout(
            self,
            x_t_start: torch.Tensor,  # (B, C, *, H, W)
            sigma_t_start: torch.Tensor,  # (B, 1, *, 1, 1)
            raw_t_start: torch.Tensor,  # (B, )
            raw_t_end: torch.Tensor,  # (B, )
            policy: BasePolicy,
            denoise_mask: Optional[torch.Tensor] = None,
            latent_image: Optional[torch.Tensor] = None):
        num_batches = x_t_start.size(0)
        ndim = x_t_start.dim()
        raw_t_start = raw_t_start.reshape(num_batches, *((ndim - 1) * [1]))
        raw_t_end = raw_t_end.reshape(num_batches, *((ndim - 1) * [1]))

        delta_raw_t = raw_t_start - raw_t_end
        num_substeps = (delta_raw_t * self.substeps).round().to(torch.long).clamp(min=1)
        substep_size = delta_raw_t / num_substeps
        max_num_substeps = num_substeps.max()

        raw_t = raw_t_start
        sigma_t = sigma_t_start
        x_t = x_t_start

        for substep_id in range(max_num_substeps.item()):
            u = policy.pi(x_t, sigma_t)
            if denoise_mask is not None and latent_image is not None:
                u_from_latent_image = (x_t - latent_image) / sigma_t
                u = u * denoise_mask + u_from_latent_image * (1 - denoise_mask)

            raw_t_minus = (raw_t - substep_size).clamp(min=0)
            sigma_t_minus = self.model_sampling.warp_t(raw_t_minus)
            x_t_minus = x_t + u * (sigma_t_minus - sigma_t)

            active_mask = num_substeps > substep_id
            x_t = torch.where(active_mask, x_t_minus, x_t)
            sigma_t = torch.where(active_mask, sigma_t_minus, sigma_t)
            raw_t = torch.where(active_mask, raw_t_minus, raw_t)

        x_t_end = x_t
        return x_t_end

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        x_t_src = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image)

        sigmas_dst, m_vals = self.calculate_sigmas_dst(sigmas)

        seed = extra_args.get("seed", None)
        if seed is not None:
            seed = (seed + 1) & 0xffffffffffffffff
            generator = torch.Generator(device=x_t_src.device).manual_seed(seed)
        else:
            generator = None

        num_batches = x_t_src.size(0)
        ndim = x_t_src.dim()

        # pi-Flow sampling loop
        nfe = len(sigmas) - 1

        for step_id in trange(nfe, disable=disable_pbar):
            sigma_t_src = sigmas[step_id].expand(num_batches).reshape(num_batches, *((ndim - 1) * [1]))
            sigma_t_dst = sigmas_dst[step_id].expand(num_batches).reshape(num_batches, *((ndim - 1) * [1]))
            t_src = self.model_sampling.timestep(sigma_t_src.flatten())
            raw_t_src = self.model_sampling.unwarp_t(sigma_t_src.flatten())
            raw_t_dst = self.model_sampling.unwarp_t(sigma_t_dst.flatten())
            sigma_t_to = sigmas[step_id + 1]
            alpha_t_to = 1 - sigma_t_to
            m = m_vals[step_id]

            policy = model_wrap(x_t_src, t_src, **extra_args)
            if isinstance(policy, GMFlowPolicy) and step_id != nfe - 1:
                policy.temperature_(self.gm_temperature)

            x_t_dst = self.policy_rollout(
                x_t_src, sigma_t_src, raw_t_src, raw_t_dst, policy,
                denoise_mask=denoise_mask, latent_image=latent_image)

            noise = torch.randn(
                x_t_dst.size(), dtype=x_t_dst.dtype, device=x_t_dst.device, generator=generator)

            x_t_to = (alpha_t_to + sigma_t_to * m) * x_t_dst + sigma_t_to * (1 - m.square()).clamp(
                min=0).sqrt() * noise

            if callback is not None:
                u = (x_t_src - x_t_to) / (sigma_t_src - sigma_t_to).clamp(min=1e-6)
                x_t_0 = x_t_to - u * sigma_t_to
                callback(step_id, x_t_0, x_t_src, nfe)

            x_t_src = x_t_to

        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x_t_src)
        return samples


class PiFlowSampler:
    def __init__(
            self, model, steps, device, h=0.0, substeps=128, final_step_size_scale=1.0, denoise=1.0, model_options={}):
        self.model = model
        self.device = device
        self.h = h
        self.substeps = substeps
        self.set_steps(steps, final_step_size_scale, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def set_steps(self, steps, final_step_size_scale=1.0, denoise=1.0):
        self.steps = steps
        if denoise <= 0.0:
            self.sigmas = torch.FloatTensor([])
        else:
            model_sampling = self.model.get_model_object("model_sampling")
            raw_timesteps = torch.from_numpy(np.linspace(
                1, (final_step_size_scale - 1) / (steps + final_step_size_scale - 1),
                steps, dtype=np.float32, endpoint=False)).clamp(min=0) * min(denoise, 1.0)
            self.sigmas = torch.cat(
                [model_sampling.warp_t(raw_timesteps), torch.FloatTensor([0.0])], dim=0)

    def sample(
            self, noise, conditioning, temperature, latent_image=None,
            denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        policy_sampler = _PolicySampler(self.model)
        policy_sampler.set_conds(conditioning, [])
        policy_sampler.set_cfg(1.0)
        sampler = _PiFlowSampler(
            self.model.get_model_object("model_sampling"),
            h=self.h, substeps=self.substeps, gm_temperature=temperature)
        return policy_sampler.sample(noise, latent_image, sampler, self.sigmas, denoise_mask, callback, disable_pbar, seed)


def sample(
        model, noise, steps, substeps, final_step_size_scale, diffusion_coefficient,
        gm_temperature, manual_gm_temperature,
        conditioning, latent_image, denoise=1.0,
        noise_mask=None, callback=None, disable_pbar=None, seed=None):
    sampler = PiFlowSampler(
        model, steps, device=model.load_device, h=diffusion_coefficient, substeps=substeps,
        final_step_size_scale=final_step_size_scale, denoise=denoise)
    if gm_temperature == 'auto':
        temperature = min(max(0.1 * (steps - 1), 0), 1)
    elif gm_temperature == 'manual':
        temperature = manual_gm_temperature
    else:
        temperature = 1.0
    samples = sampler.sample(
        noise, conditioning, temperature=temperature, latent_image=latent_image,
        denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples
