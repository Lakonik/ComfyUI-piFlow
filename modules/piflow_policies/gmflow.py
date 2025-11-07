# Copyright (c) 2025 Hansheng Chen

import math
import torch

from typing import Dict
from .base import BasePolicy


@torch.jit.script
def gmflow_posterior_mean_jit(
        sigma_t_src, sigma_t, x_t_src, x_t,
        gm_means, gm_vars, gm_logweights,
        eps: float, gm_dim: int = -4, channel_dim: int = -3):
    alpha_t_src = 1 - sigma_t_src
    alpha_t = 1 - sigma_t

    sigma_t_src_sq = sigma_t_src.square()
    sigma_t_sq = sigma_t.square()

    # compute gaussian params
    denom = (alpha_t.square() * sigma_t_src_sq - alpha_t_src.square() * sigma_t_sq).clamp(min=eps)  # ζ
    g_mean = (alpha_t * sigma_t_src_sq * x_t - alpha_t_src * sigma_t_sq * x_t_src) / denom  # ν / ζ
    g_var = sigma_t_sq * sigma_t_src_sq / denom

    # gm_mul_iso_gaussian
    g_mean = g_mean.unsqueeze(gm_dim)  # (bs, *, 1, out_channels, h, w)
    g_var = g_var.unsqueeze(gm_dim)  # (bs, *, 1, 1, 1, 1)

    gm_diffs = gm_means - g_mean  # (bs, *, num_gaussians, out_channels, h, w)
    norm_factor = (g_var + gm_vars).clamp(min=eps)

    out_means = (g_var * gm_means + gm_vars * g_mean) / norm_factor
    # (bs, *, num_gaussians, 1, h, w)
    logweights_delta = gm_diffs.square().sum(dim=channel_dim, keepdim=True) * (-0.5 / norm_factor)
    out_weights = (gm_logweights + logweights_delta).softmax(dim=gm_dim)

    out_mean = (out_means * out_weights).sum(dim=gm_dim)

    return out_mean


def gm_temperature(gm, temperature, gm_dim=-4, eps=1e-6):
    gm = gm.copy()
    temperature = max(temperature, eps)
    gm['logweights'] = (gm['logweights'] / temperature).log_softmax(dim=gm_dim)
    if 'logstds' in gm:
        gm['logstds'] = gm['logstds'] + (0.5 * math.log(temperature))
    if 'gm_vars' in gm:
        gm['gm_vars'] = gm['gm_vars'] * temperature
    return gm


class GMFlowPolicy(BasePolicy):
    """GMFlow policy. The number of components K is inferred from the denoising output.

    Args:
        denoising_output (dict): The output of the denoising model, containing:
            means (torch.Tensor): The means of the Gaussian components. Shape (B, K, C, H, W) or (B, K, C, T, H, W).
            logstds (torch.Tensor): The log standard deviations of the Gaussian components. Shape (B, K, 1, 1, 1)
                or (B, K, 1, 1, 1, 1).
            logweights (torch.Tensor): The log weights of the Gaussian components. Shape (B, K, 1, H, W) or
                (B, K, 1, T, H, W).
        x_t_src (torch.Tensor): The initial noisy sample. Shape (B, C, H, W) or (B, C, T, H, W).
        sigma_t_src (torch.Tensor): The initial noise level. Shape (B,).
        checkpointing (bool): Whether to use gradient checkpointing to save memory. Defaults to True.
        eps (float): A small value to avoid numerical issues. Defaults to 1e-4.
    """

    def __init__(
            self,
            denoising_output: Dict[str, torch.Tensor],
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            checkpointing: bool = True,
            eps: float = 1e-4):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.checkpointing = checkpointing
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.denoising_output_x_0 = self._u_to_x_0(
            denoising_output, self.x_t_src, self.sigma_t_src)

    @staticmethod
    def _u_to_x_0(denoising_output, x_t, sigma_t):
        x_t = x_t.unsqueeze(1)
        sigma_t = sigma_t.unsqueeze(1)
        means_x_0 = x_t - sigma_t * denoising_output['means']
        gm_vars = (denoising_output['logstds'] * 2).exp() * sigma_t.square()
        return dict(
            means=means_x_0,
            gm_vars=gm_vars,
            logweights=denoising_output['logweights'])

    def pi(self, x_t, sigma_t):
        """Compute the flow velocity at (x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at time t.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        means = self.denoising_output_x_0['means']
        gm_vars = self.denoising_output_x_0['gm_vars']
        logweights = self.denoising_output_x_0['logweights']
        if (sigma_t == self.sigma_t_src).all() and (x_t == self.x_t_src).all():
            x_0 = (logweights.softmax(dim=1) * means).sum(dim=1)
        else:
            if self.checkpointing and torch.is_grad_enabled():
                x_0 = torch.utils.checkpoint.checkpoint(
                    gmflow_posterior_mean_jit,
                    self.sigma_t_src, sigma_t, self.x_t_src, x_t,
                    means,
                    gm_vars,
                    logweights,
                    self.eps, 1, 2,
                    use_reentrant=True)  # use_reentrant=False does not work with jit
            else:
                x_0 = gmflow_posterior_mean_jit(
                    self.sigma_t_src, sigma_t, self.x_t_src, x_t,
                    means,
                    gm_vars,
                    logweights,
                    self.eps, 1, 2)
        u = (x_t - x_0) / sigma_t.clamp(min=self.eps)
        return u

    def copy(self):
        new_policy = GMFlowPolicy.__new__(GMFlowPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.ndim = self.ndim
        new_policy.checkpointing = self.checkpointing
        new_policy.eps = self.eps
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.denoising_output_x_0 = self.denoising_output_x_0.copy()
        return new_policy

    def detach_(self):
        self.denoising_output_x_0 = {k: v.detach() for k, v in self.denoising_output_x_0.items()}
        return self

    def detach(self):
        new_policy = self.copy()
        return new_policy.detach_()

    def dropout_(self, p):
        if p <= 0 or p >= 1:
            return self
        logweights = self.denoising_output_x_0['logweights']
        dropout_mask = torch.rand(
            (*logweights.shape[:2], *((self.ndim - 1) * [1])), device=logweights.device) < p
        is_all_dropout = dropout_mask.all(dim=1, keepdim=True)
        dropout_mask &= ~is_all_dropout
        self.denoising_output_x_0['logweights'] = logweights.masked_fill(
            dropout_mask, float('-inf'))
        return self

    def dropout(self, p):
        new_policy = self.copy()
        return new_policy.dropout_(p)

    def temperature_(self, temp):
        if temp >= 1.0:
            return self
        self.denoising_output_x_0 = gm_temperature(
            self.denoising_output_x_0, temp, gm_dim=1, eps=self.eps)
        return self

    def temperature(self, temp):
        new_policy = self.copy()
        return new_policy.temperature_(temp)
