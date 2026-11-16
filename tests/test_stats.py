import unittest
import torch
from shrf.stats import StatLayout, contribution, ridge, shrinkage_lambda
from shrf.sketch import ResidualSketch


class StatsTest(unittest.TestCase):
    def test_weighted_ridge_and_rss(self):
        torch.manual_seed(18)
        l = StatLayout(4, 6)
        k, v, w = torch.randn(80,4), torch.randn(80,6), torch.rand(80,1)
        q = torch.randn(4)
        s = (contribution(k,v,l)*w).sum(0)
        result = ridge(s,q,l,.05)
        kc, vc = k-(k*w).sum(0)/w.sum(), v-(v*w).sum(0)/w.sum()
        ref = torch.linalg.solve(kc.T @ (kc*w)+.05*torch.eye(4), kc.T @ (vc*w))
        torch.testing.assert_close(result["M"], ref)
        torch.testing.assert_close(result["rss"], ((vc-kc@ref).square()*w).sum())
        self.assertEqual(ridge(torch.zeros(l.dim),q,l,.05)["pred"].abs().sum().item(), 0)
        n = torch.tensor([0.,1.,5.,10.])
        lam = shrinkage_lambda(n, torch.ones_like(n), torch.ones_like(n), .25, 100)
        self.assertTrue(torch.all(lam[1:] > lam[:-1]))

    def test_sketch_learns_nonlinear_residual(self):
        l = StatLayout(1, 1)
        f = ResidualSketch(1, 1, 1, 16, 3., 9)
        x = torch.linspace(-1, 1, 150).reshape(-1,1,1)
        y = torch.cos(5*x)
        train = torch.arange(len(x)) % 2 == 0
        base = contribution(x[train],y[train],l).sum(0)
        extra = f.contribution(x[train],f.features(x[train]),y[train]).sum(0)
        xq = x[~train]
        plain = ridge(base.expand(len(xq),1,-1),xq,l,.05)["pred"]
        joint = f.joint(base.expand(len(xq),1,-1),extra.expand(len(xq),1,-1),
                        xq,f.features(xq),l,.05,.05)["pred"]
        self.assertLess((joint-y[~train]).square().mean().item(),
                        .1*(plain-y[~train]).square().mean().item())
