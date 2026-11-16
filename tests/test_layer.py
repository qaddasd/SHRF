import unittest
import torch
from shrf import SHRFConfig, SHRFLayer


class LayerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.x = torch.randn(2, 17, 16)
        self.cfg = dict(d_model=16, depth=3, r_key=4, d_value=8, sketch_rank=5, beam=3)

    def test_parallel_recurrent_and_causality(self):
        for lag in (0, 1):
            with torch.no_grad():
                model = SHRFLayer(SHRFConfig(**self.cfg, chunk_size=5, write_lag=lag))
                y, end = model(self.x)
                state = model.init_state(2)
                recurrent = torch.stack([model.step(z, state)[0] for z in self.x.unbind(1)], dim=1)
                torch.testing.assert_close(y, recurrent, atol=1e-5, rtol=3e-5)
                torch.testing.assert_close(end.heap, state.heap, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(end.extra, state.extra, atol=3e-6, rtol=3e-5)
                changed = self.x.clone()
                changed[:, 8:] += 5
                torch.testing.assert_close(model(changed)[0][:, :8], y[:, :8], rtol=0, atol=0)

    def test_full_gradient_across_chunks(self):
        small = SHRFLayer(SHRFConfig(**self.cfg, chunk_size=4, detach_chunks=False)).double()
        large = SHRFLayer(SHRFConfig(**self.cfg, chunk_size=32, detach_chunks=False)).double()
        large.load_state_dict(small.state_dict())
        gradients = []
        for layer in (small, large):
            x = self.x.double().requires_grad_()
            y, _ = layer(x)
            y[:, -1].square().mean().backward()
            self.assertGreater(x.grad[:, 1].abs().sum().item(), 0)
            self.assertGreater(layer.routing.dirs.grad.abs().sum().item(), 0)
            gradients.append(x.grad)
        torch.testing.assert_close(*gradients, rtol=1e-9, atol=1e-10)

    def test_precision_empty_and_write_mask(self):
        layer = SHRFLayer(SHRFConfig(**self.cfg)).half()
        x = self.x.half()
        with torch.no_grad():
            y, state = layer(x, write_weight=torch.zeros(x.shape[:2]))
            self.assertEqual(state.heap.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.float16)
            self.assertEqual(y.abs().max().item(), 0)
            self.assertEqual(layer(x[:, :0])[0].shape[1], 0)

    def test_context_and_power_gate_match_stream(self):
        layer = SHRFLayer(SHRFConfig(**self.cfg, context_lags=3, gate_power=4, chunk_size=2)).double()
        layer.context_mix.data.normal_(mean=.1, std=.2)
        x = self.x.double()
        with torch.no_grad():
            full, expected = layer(x)
            _, state = layer(x[:, :3])
            suffix, state = layer(x[:, 3:], state)
            torch.testing.assert_close(full[:, 3:], suffix, atol=2e-11, rtol=1e-9)
            state = layer.init_state(2)
            recurrent = torch.stack([layer.step(v, state)[0] for v in x.unbind(1)], dim=1)
            torch.testing.assert_close(full, recurrent, atol=2e-11, rtol=1e-9)
            torch.testing.assert_close(expected.key_history, state.key_history)

    def test_coalescing_retains_small_writes_after_large_prefix(self):
        heap = torch.tensor([[[[1e6, 1e6]]]])
        group = torch.zeros(64, dtype=torch.long)
        rows = torch.full((64, 2), .01)
        original = heap.clone()
        for _ in range(80):
            SHRFLayer._commit(heap, [group], [rows], False, True)
            SHRFLayer._commit(original, [group], [rows], False, False)
        self.assertEqual(original[0,0,0,0].item(), 1e6)
        self.assertLess(abs(float(heap[0,0,0,0])-1_000_051.2), 1.3)


if __name__ == "__main__":
    unittest.main()
