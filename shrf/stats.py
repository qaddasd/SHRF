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
        return {"n": s[..., self.n], "a": s[..., self.a], "b": s[..., self.b],
                    "G": s[..., self.G].unflatten(-1, (self.r, self.r)),
                    "C": s[..., self.C].unflatten(-1, (self.r, self.q)), "e": s[..., self.e]}


def outer(x, y):
    return x.unsqueeze(-1) * y.unsqueeze(-2)


def contribution(k: Tensor, v: Tensor, lay: StatLayout) -> Tensor:
    energy = v.square().sum(-1, keepdim=True)
    return torch.cat((torch.ones_like(energy), k, v, outer(k, k).flatten(-2),
                      outer(k, v).flatten(-2), energy), dim=-1)


def fit(n, a, b, gram, cross, energy, key, diagonal):
    """Fit row-weighted ridge with an unpenalized intercept."""
    dtype = key.dtype
    safe = n.double().clamp_min(1e-8).unsqueeze(-1)
    a, b, gram, cross = (x.double() for x in (a, b, gram, cross))
    mu_k, mu_v = a / safe, b / safe
    centered_gram = gram - outer(a, mu_k)
    centered_gram = 0.5 * (centered_gram + centered_gram.transpose(-1, -2))
    centered_cross = cross - outer(a, mu_v)
    diagonal = diagonal.double()
    system = centered_gram + torch.diag_embed(diagonal)
    factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
    failed = (info != 0)[..., None, None]
    if bool(failed.any()):
        # Rebuild failed systems before backward so malformed factors never enter autograd.
        resized = torch.eye(system.shape[-1], device=system.device, dtype=system.dtype)
        factor = torch.linalg.cholesky_ex(torch.where(failed, resized, system)).L
        centered_cross = centered_cross.masked_fill(failed, 0)
    coef64 = torch.cholesky_solve(centered_cross, factor)
    pred = (mu_v + ((key.double() - mu_k).unsqueeze(-2) @ coef64).squeeze(-2)).to(dtype)
    gain = (coef64 * centered_cross).sum((-1, -2))
    rss = (energy.double() - (b * mu_v).sum(-1) - gain
           - (diagonal.unsqueeze(-1) * coef64.square()).sum((-1, -2))).clamp_min(0)
    return {"pred": pred.masked_fill(n.unsqueeze(-1) <= 1e-8, 0),
                "n": n, "sigma2": (rss / (safe.squeeze(-1) * b.shape[-1])).to(dtype),
                "mu_v": mu_v.to(dtype), "mu_k": mu_k.to(dtype), "rss": rss.to(dtype),
                "gain": gain.to(dtype), "M": coef64.to(dtype)}


def ridge(stats: Tensor, k: Tensor, lay: StatLayout, rho: float) -> dict:
    u = lay.unpack(stats)
    return fit(u["n"], u["a"], u["b"], u["G"], u["C"], u["e"], k, k.new_full((lay.r,), rho))


def shrinkage_lambda(n, sigma2, tau2, kappa_min, kappa_max):
    kappa = (sigma2 / tau2.clamp_min(1e-8)).clamp(kappa_min, kappa_max)
    return n / (n + kappa)
