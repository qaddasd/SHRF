"""Bounded soft routing over a fixed-shape 1-indexed heap."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn, Tensor
from .stats import StatLayout, contribution, ridge


@dataclass
class RouteLevel:
    node: Tensor
    mass: Tensor
    parent: Tensor | None
    right_probability: Tensor | None = None


class Routing(nn.Module):
    def __init__(self, trees, depth, r, tau, mix, beam, seed):
        super().__init__()
        self.E, self.D, self.B = trees, depth, beam
        self.tau, self.mix = tau, mix
        gen = torch.Generator().manual_seed(seed)
        base = torch.nn.functional.normalize(torch.randn(trees, 2**depth, r, generator=gen), dim=-1)
        self.register_buffer("base_dir", base)
        self.dirs = nn.Parameter(base.clone())
        self.theta = nn.Parameter(torch.zeros(trees, 2**depth))

    def root(self, keys):
        shape = (*keys.shape[:-1], 1)
        return RouteLevel(torch.ones(shape, dtype=torch.long, device=keys.device),
                          keys.new_ones(shape), None)

    def descend(self, keys, level):
        trees = torch.arange(self.E, device=keys.device).unsqueeze(-1)
        direction = self.dirs[trees, level.node].to(keys.dtype)
        threshold = self.theta[trees, level.node].to(keys.dtype)
        right = (((direction * keys.unsqueeze(-2)).sum(-1) - threshold) / self.tau).sigmoid()
        child_mass = (level.mass.unsqueeze(-1) * torch.stack((1-right, right), -1)).flatten(-2)
        child_node = torch.stack((2*level.node, 2*level.node + 1), -1).flatten(-2)
        width = min(self.B, child_node.shape[-1])
        # Stable tie break by candidate order (parent order, then left before right).
        order = child_mass.argsort(dim=-1, descending=True, stable=True)[..., :width]
        return RouteLevel(child_node.gather(-1, order), child_mass.gather(-1, order),
                          order // 2, right)

    def forward(self, keys):
        levels = [self.root(keys)]
        for _ in range(self.D):
            levels.append(self.descend(keys, levels[-1]))
        return levels

    @torch.no_grad()
    def calibrate(self, keys, values, rho, min_support=8):
        """Use projected [observations, trees, r] keys and [observations, q] values."""
        layout = StatLayout(keys.shape[-1], values.shape[-1])
        level = self.root(keys)
        new_dirs, new_theta = self.dirs.clone(), self.theta.clone()
        values_e = values[:, None, :].expand(-1, self.E, -1)
        rows = contribution(keys, values_e, layout)
        for depth in range(self.D):
            for e in range(self.E):
                for u in range(2**depth, 2**(depth + 1)):
                    mask = (level.node[:, e] == u) & (level.mass[:, e] > 0)
                    if int(mask.sum()) < min_support:
                        continue
                    ids, slots = torch.where(mask)
                    k, v, w = keys[ids, e], values[ids], level.mass[ids, e, slots]
                    stats = (rows[ids, e] * w.unsqueeze(-1)).sum(0)
                    fit = ridge(stats.expand(len(k), -1), k, layout, rho)
                    err = (v - fit["pred"]).square().sum(-1)
                    centered = k - fit["mu_k"]
                    covariance = centered.T @ (centered * (w * err).unsqueeze(-1))
                    if covariance.trace() > 1e-10:
                        top = torch.linalg.eigh(covariance).eigenvectors[:, -1]
                        base = self.base_dir[e, u].to(top.dtype)
                        top = torch.where(top @ base < 0, -top, top)
                        direction = torch.nn.functional.normalize((1-self.mix)*base + self.mix*top, dim=0)
                    else:
                        direction = self.base_dir[e, u].to(k.dtype)
                    scores, order = (k @ direction).sort()
                    mid = torch.searchsorted(w[order].cumsum(0), 0.5*w.sum()).clamp_max(len(k)-1)
                    new_dirs[e, u], new_theta[e, u] = direction, scores[mid]
            start, end = 2**depth, 2**(depth+1)
            self.dirs[:, start:end].copy_(new_dirs[:, start:end])
            self.theta[:, start:end].copy_(new_theta[:, start:end])
            level = self.descend(keys, level)
