import unittest

import torch

from shrf import SHRFConfig, SHRFLayer
from shrf.rust_backend import RustSHRF


class NativeTest(unittest.TestCase):
    @torch.no_grad()
    def test_weighted_native_decode_and_reset(self):
        torch.manual_seed(41)
        x = torch.randn(12, 8, dtype=torch.float64)
        w = torch.rand(12, dtype=torch.float64)
        w[2] = 0
        for extras in ({}, dict(use_sketch=False, use_shrinkage=False, write_lag=0),
                       dict(context_lags=3, gate_power=4)):
            layer = SHRFLayer(SHRFConfig(d_model=8, depth=3, r_key=4, d_value=6, **extras)).double()
            if layer.context_mix is not None:
                layer.context_mix.normal_(mean=.1, std=.2)
            try:
                decoder = RustSHRF(layer)
            except FileNotFoundError as error:
                self.skipTest(str(error))
            with decoder:
                st = layer.init_state(1)
                expected = torch.stack([layer.step(row[None], st, write_weight=weight[None])[0][0]
                                        for row, weight in zip(x, w)])
                for _ in range(2):
                    result = torch.stack([decoder.step(row, write_weight=float(weight))
                                          for row, weight in zip(x, w)])
                    torch.testing.assert_close(expected, result, atol=1e-10, rtol=1e-9)
                    decoder.reset()
                self.assertEqual(decoder.prefill(x[:0]).shape, (0, x.shape[1]))
                self.assertEqual(decoder.prefill(x[:2], write_weight=w[:2]).shape, (2, x.shape[1]))
                torch.testing.assert_close(expected[2:], decoder.prefill(x[2:], write_weight=w[2:]),
                                           atol=1e-10, rtol=1e-9)
