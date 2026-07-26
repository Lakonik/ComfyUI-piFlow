import copy
import numpy as np
import torch
import comfy
import comfy.sampler_helpers
import comfy.patcher_extension
import comfy.model_patcher
from typing import Optional
from comfy.samplers import Sampler, CFGGuider, finalize_default_conds, get_area_and_mult, cond_cat
from tqdm.auto import trange
from .model_base import BasePiFlow
from .piflow_policies.base import BasePolicy


def deepcopy_no_tensors(x):
    memo = {}

    # Pre-register all tensors we can reach so deepcopy will reuse them.
    def register_tensors(obj):
        if torch.is_tensor(obj):
            memo[id(obj)] = obj
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                register_tensors(k)
                register_tensors(v)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for v in obj:
                register_tensors(v)

    register_tensors(x)
    return copy.deepcopy(x, memo)


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


class CompositePolicy(BasePolicy):
    def __init__(self, entries: list[tuple[BasePolicy, list[int] | None, torch.Tensor]]):
        self.entries = entries

    @staticmethod
    def _narrow_area(tensor: torch.Tensor, area: list[int] | None):
        if area is None:
            return tensor

        dims = len(area) // 2
        for i in range(dims):
            tensor_dim = i + 2
            if tensor.ndim > tensor_dim and tensor.shape[tensor_dim] > 1:
                tensor = tensor.narrow(tensor_dim, area[i + dims], area[i])
        return tensor

    def pi(self, x_t, sigma_t):
        out = torch.zeros_like(x_t)
        counts = torch.ones_like(x_t) * 1e-37

        for policy, area, mult in self.entries:
            if area is None:
                out += policy.pi(x_t, sigma_t) * mult
                counts += mult
                continue

            out_view = out
            counts_view = counts
            dims = len(area) // 2
            for i in range(dims):
                out_view = out_view.narrow(i + 2, area[i + dims], area[i])
                counts_view = counts_view.narrow(i + 2, area[i + dims], area[i])

            x_view = self._narrow_area(x_t, area)
            sigma_view = self._narrow_area(sigma_t, area)
            out_view += policy.pi(x_view, sigma_view) * mult
            counts_view += mult

        return out / counts

    def detach(self):
        return CompositePolicy([(policy.detach(), area, mult) for policy, area, mult in self.entries])

    def temperature_(self, temperature):
        for policy, _, _ in self.entries:
            if hasattr(policy, "temperature_"):
                policy.temperature_(temperature)
        return self


def _calc_cond_batch(model: BasePiFlow, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    out_conds = [[] for _ in conds]
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

    model.current_patcher.prepare_state(timestep, model_options)

    # run every hooked_to_run separately
    for hooks, to_run in hooked_to_run.items():
        while len(to_run) > 0:
            p, cond_index = to_run.pop()
            input_x = p.input_x
            c = cond_cat([p.conditioning])
            timestep_ = timestep

            transformer_options = model.current_patcher.apply_hooks(hooks=hooks)
            if 'transformer_options' in model_options:
                transformer_options = comfy.patcher_extension.merge_nested_dicts(
                    transformer_options, model_options['transformer_options'], copy_dict1=False)

            if p.patches is not None:
                transformer_options["patches"] = comfy.patcher_extension.merge_nested_dicts(
                    transformer_options.get("patches", {}),
                    p.patches
                )

            transformer_options["cond_or_uncond"] = [cond_index]
            transformer_options["uuids"] = [p.uuid]
            transformer_options["sigmas"] = timestep

            c['transformer_options'] = transformer_options

            if p.control is not None:
                c['control'] = p.control.get_control(input_x, timestep_, c, 1, transformer_options)

            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": [cond_index]})
            else:
                output = model.apply_model(input_x, timestep_, **c)

            out_conds[cond_index].append((output, p.area, p.mult))

    policies = []
    for entries in out_conds:
        if len(entries) == 0:
            raise RuntimeError("No active pi-Flow conditioning was available for this sampling step.")
        if len(entries) == 1 and entries[0][1] is None and not torch.any(entries[0][2] == 0).item():
            policies.append(entries[0][0])
        else:
            policies.append(CompositePolicy(entries))

    if len(policies) == 1:
        return policies[0]
    return policies


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
        noise = self.model_sampling.unpatchify(noise)
        if latent_image is not None:
            latent_image = self.model_sampling.unpatchify(latent_image)
        if denoise_mask is not None:
            denoise_mask = self.model_sampling.unpatchify(denoise_mask)

        x_t_src = self.model_sampling.noise_scaling(
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
            if hasattr(policy, "temperature_") and step_id != nfe - 1:
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
                callback(  # for preview
                    step_id,
                    self.model_sampling.patchify(x_t_0),
                    self.model_sampling.patchify(x_t_src),
                    nfe)

            x_t_src = x_t_to

        samples = self.model_sampling.inverse_noise_scaling(sigmas[-1], x_t_src)

        samples = self.model_sampling.patchify(samples)
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

        model_sampling = self.model.get_model_object("model_sampling")
        conditioning = deepcopy_no_tensors(conditioning)
        for conditioning_single in conditioning:
            conditioning_dict = conditioning_single[1]
            if 'reference_latents' in conditioning_dict:
                conditioning_dict['reference_latents'] = [
                    model_sampling.unpatchify(x) for x in conditioning_dict['reference_latents']]

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
