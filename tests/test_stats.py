import unittest

import torch

from shrf.sketch import ResidualSketch
from shrf.stats import StatLayout, contribution, ridge, shrinkage_lambda


class StatsTest(unittest.TestCase):
    def test_anchored_fp32_rss_matches_fp64_with_large_offset(self):
        gen = torch.Generator().manual_seed(64)
        layout = StatLayout(1, 1)
        k = torch.randn(100_000, 1, generator=gen)
        v = 20_000 + 0.25 * k + 0.04 * torch.randn(k.shape, generator=gen)
        kc, vc = k - k[:1], v - v[:1]
        fp32 = ridge(contribution(kc, vc, layout).sum(0), kc[0], layout, .05)
        fp64 = ridge(contribution(kc.double(), vc.double(), layout).sum(0), kc[0].double(), layout, .05)
        torch.testing.assert_close(fp32["rss"].double(), fp64["rss"], rtol=2e-4, atol=0)

    def test_configured_ridge_does_not_grow_with_mass(self):
        layout = StatLayout(1, 1)
        k = torch.tensor([[-.001], [.001]], dtype=torch.float64)
        raw = contribution(k, k, layout).sum(0) * 500_000
        result = ridge(raw, k[0], layout, .05)
        torch.testing.assert_close(result["M"], torch.tensor([[1/1.05]], dtype=k.dtype))

    def test_weighted_ridge_and_rss(self):
        torch.manual_seed(18)
        layout = StatLayout(4, 6)
        k, v, w = torch.randn(80,4), torch.randn(80,6), torch.rand(80,1)
        q = torch.randn(4)
        s = (contribution(k,v,layout)*w).sum(0)
        result = ridge(s,q,layout,.05)
        kc, vc = k-(k*w).sum(0)/w.sum(), v-(v*w).sum(0)/w.sum()
        ref = torch.linalg.solve(kc.T @ (kc*w)+.05*torch.eye(4), kc.T @ (vc*w))
        torch.testing.assert_close(result["M"], ref)
        torch.testing.assert_close(result["rss"], ((vc-kc@ref).square()*w).sum())
        self.assertEqual(ridge(torch.zeros(layout.dim),q,layout,.05)["pred"].abs().sum().item(), 0)
        n = torch.tensor([0.,1.,5.,10.])
        lam = shrinkage_lambda(n, torch.ones_like(n), torch.ones_like(n), .25, 100)
        self.assertTrue(torch.all(lam[1:] > lam[:-1]))

    def test_sketch_learns_nonlinear_residual(self):
        layout = StatLayout(1, 1)
        f = ResidualSketch(1, 1, 1, 16, 3., 9)
        x = torch.linspace(-1, 1, 150).reshape(-1,1,1)
        y = torch.cos(5*x)
        train = torch.arange(len(x)) % 2 == 0
        base = contribution(x[train],y[train],layout).sum(0)
        extra = f.contribution(x[train],f.features(x[train]),y[train]).sum(0)
        xq = x[~train]
        plain = ridge(base.expand(len(xq),1,-1),xq,layout,.05)["pred"]
        joint = f.joint(base.expand(len(xq),1,-1),extra.expand(len(xq),1,-1),
                        xq,f.features(xq),layout,.05,.05)["pred"]
        self.assertLess((joint-y[~train]).square().mean().item(),
                        .1*(plain-y[~train]).square().mean().item())
