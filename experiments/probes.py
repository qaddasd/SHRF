"""Controlled numeric probes for the nonlinear sketch and shrinkage."""
import json
from pathlib import Path

import torch

from shrf.sketch import ResidualSketch
from shrf.stats import StatLayout, contribution, ridge, shrinkage_lambda


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    layout = StatLayout(1, 1)
    x = torch.linspace(-1, 1, 150, dtype=torch.float64).reshape(-1, 1, 1)
    y = (5*x).cos()
    train, query = x[::2], x[1::2]
    target = y[1::2]
    base = contribution(train, y[::2], layout).sum(0)
    baseline = ridge(base.expand(len(query), 1, -1), query, layout, .05)["pred"]
    errors = {"0": (baseline-target).square().mean().item()}
    for m in (2, 4, 8, 16):
        seed_errors = []
        for seed in range(10):
            sketch = ResidualSketch(1, 1, 1, m, 3., seed)
            extra = sketch.contribution(train, sketch.features(train), y[::2]).sum(0)
            pred = sketch.joint(base.expand(len(query), 1, -1), extra.expand(len(query), 1, -1),
                                query, sketch.features(query), layout, .05, .05)["pred"]
            seed_errors.append((pred-target).square().mean().item())
        errors[str(m)] = sum(seed_errors)/len(seed_errors)

    # Oracle parent mean isolates the bias-variance tradeoff.
    gen = torch.Generator().manual_seed(7)
    shrink = {}
    for n in (2, 4, 16, 64):
        samples = .1 + torch.randn(8192, n, generator=gen, dtype=torch.float64)
        local = samples.mean(-1)
        noise = samples.var(-1, correction=0)
        w = shrinkage_lambda(torch.full_like(local, n), noise, local.square() + .00101, .25, 256.)
        shrink[str(n)] = {
            "raw": (local-.1).square().mean().item(),
            "shrunken": (w*local-.1).square().mean().item(),
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/probes.json").write_text(
        json.dumps({"nonlinear_mse": errors, "shrinkage_mse": shrink}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
