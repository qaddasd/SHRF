"""Additive weighted sufficient statistics and batched affine ridge fits."""
from __future__ import annotations
import torch
from torch import Tensor


class StatLayout:
    def __init__(self, r: int, q: int):
        self.r, self.q = r, q
        self.n = 0
        self.a = slice(1, 1 + r)
        self.b = slice(1 + r, 1 + r + q)
        gram = self.b.stop
        self.G = slice(gram, gram + r*r)
        self.C = slice(self.G.stop, self.G.stop + r*q)
        self.e = self.C.stop
        self.dim = self.e + 1

    def unpack(self, s: Tensor) -> dict:
        return dict(n=s[..., self.n], a=s[..., self.a], b=s[..., self.b],
                    G=s[..., self.G].unflatten(-1, (self.r, self.r)),
                    C=s[..., self.C].unflatten(-1, (self.r, self.q)), e=s[..., self.e])


def outer(x, y):
    return x.unsqueeze(-1) * y.unsqueeze(-2)


def contribution(k: Tensor, v: Tensor, lay: StatLayout) -> Tensor:
    energy = v.square().sum(-1, keepdim=True)
    return torch.cat((torch.ones_like(energy), k, v, outer(k, k).flatten(-2),
                      outer(k, v).flatten(-2), energy), dim=-1)


def fit(n, a, b, gram, cross, energy, key, diagonal):
    """Fit row-weighted ridge with an unpenalized intercept."""
    safe = n.clamp_min(1e-8).unsqueeze(-1)
    mu_k, mu_v = a / safe, b / safe
    centered_gram = gram - outer(a, mu_k)
    centered_gram = 0.5 * (centered_gram + centered_gram.transpose(-1, -2))
    centered_cross = cross - outer(a, mu_v)
    reg = torch.maximum(diagonal, 1e-5 * n.unsqueeze(-1))
    system = centered_gram + torch.diag_embed(reg)
    coef = torch.linalg.solve_ex(system, centered_cross, check_errors=False).result
    pred = mu_v + ((key - mu_k).unsqueeze(-2) @ coef).squeeze(-2)
    gain = (coef * centered_cross).sum((-1, -2))
    rss = (energy - (b * mu_v).sum(-1) - gain -
           (reg.unsqueeze(-1) * coef.square()).sum((-1, -2))).clamp_min(0)
    return dict(pred=pred.masked_fill(n.unsqueeze(-1) <= 1e-8, 0),
                n=n, sigma2=rss / (safe.squeeze(-1) * b.shape[-1]),
                mu_v=mu_v, mu_k=mu_k, rss=rss, gain=gain, M=coef)


def ridge(stats: Tensor, k: Tensor, lay: StatLayout, rho: float) -> dict:
    u = lay.unpack(stats)
    return fit(u["n"], u["a"], u["b"], u["G"], u["C"], u["e"], k, k.new_full((lay.r,), rho))


def shrinkage_lambda(n, sigma2, tau2, kappa_min, kappa_max):
    kappa = (sigma2 / tau2.clamp_min(1e-8)).clamp(kappa_min, kappa_max)
    return n / (n + kappa)
