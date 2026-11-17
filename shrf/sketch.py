"""Mergeable nonlinear moment sketch for leaf ridge residuals."""
import math

import torch
from torch import nn

from .stats import fit, outer


class ResidualSketch(nn.Module):
    def __init__(self, trees, r, q, m, bandwidth, seed):
        super().__init__()
        self.r, self.q, self.m = r, q, m
        self.dim = m + r*m + m*m + m*q
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("omega", torch.randn(trees, r, m, generator=gen) * bandwidth)
        self.register_buffer("phase", 2 * math.pi * torch.rand(trees, m, generator=gen))

    def features(self, k):
        angle = torch.einsum("...er,erm->...em", k, self.omega.to(k.dtype))
        return math.sqrt(2 / self.m) * (angle + self.phase.to(k.dtype)).cos()

    def tree_features(self, tree, k):
        angle = k @ self.omega[tree].to(k) + self.phase[tree].to(k)
        return math.sqrt(2 / self.m) * angle.cos()

    def contribution(self, k, phi, v):
        return torch.cat((phi, outer(k, phi).flatten(-2), outer(phi, phi).flatten(-2),
                          outer(phi, v).flatten(-2)), dim=-1)

    def joint(self, base, extra, k, phi, layout, rho, sketch_rho):
        r, m, q = self.r, self.m, self.q
        u = layout.unpack(base)
        a, j, g, c = extra.split((m, r*m, m*m, m*q), dim=-1)
        j = j.unflatten(-1, (r, m))
        g = g.unflatten(-1, (m, m))
        c = c.unflatten(-1, (m, q))
        gram = torch.cat((torch.cat((u["G"], j), -1),
                          torch.cat((j.transpose(-1, -2), g), -1)), -2)
        key = torch.cat((k, phi), -1)
        reg = key.new_tensor([rho] * r + [sketch_rho] * m)
        return fit(u["n"], torch.cat((u["a"], a), -1), u["b"], gram,
                   torch.cat((u["C"], c), -2), u["e"], key, reg)
