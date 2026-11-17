import unittest

import torch

from shrf import SHRFConfig, SHRFLayer
from shrf.tape import IDTape


class ConsolidationTest(unittest.TestCase):
    @torch.no_grad()
    def test_empty_leaves_stay_unassigned_and_evicted_leaf_can_return(self):
        cfg = SHRFConfig(d_model=1, r_key=1, d_value=1, n_trees=1,
                         depth=2, beam=1, write_lag=0, decay=.95)
        layer = SHRFLayer(cfg)
        layer.W_k.fill_(1.)
        layer.routing.dirs.fill_(1.)
        self.assertIsNotNone(layer.sketch)
        with IDTape() as tape:
            state = layer.init_state(1, tape=tape)
            layer(torch.ones(1, 4, 1), state)
            leaf_map = layer.consolidate(state, budget=4)
            self.assertEqual(int((leaf_map > 0).sum()), 1)
            expected_extra = state.extra.clone()
            layer.consolidate(state, budget=0)
            self.assertEqual(int((state.leaf_slots > 0).sum()), 0)
            layer.consolidate(state, budget=4)
            torch.testing.assert_close(state.extra, expected_extra)
            self.assertEqual(int((state.leaf_slots > 0).sum()), 1)
            y, _ = layer.step(torch.ones(1, 1), state)
            self.assertTrue(torch.isfinite(y).all())

    @torch.no_grad()
    def test_budget_preserves_base_and_archive(self):
        torch.manual_seed(17)
        layer = SHRFLayer(SHRFConfig(d_model=8, depth=3, r_key=4, d_value=6))
        x = torch.randn(2, 19, 8)
        with IDTape() as tape:
            state = layer.init_state(2, tape=tape)
            _, state = layer(x, state)
            old = state.heap.clone()
            old_extra = state.extra.numel()
            routes = layer.routing(torch.einsum("btd,erd->bter", x, layer.W_k).tanh())[-1]
            leaf = int(routes.node[1, 0, 0, 0]) - layer.cfg.n_leaves
            slot_map = layer.consolidate(state, budget=2)
            self.assertTrue(torch.all((slot_map > 0).sum(-1) == 2))
            self.assertLess(state.extra.numel(), old_extra)
            torch.testing.assert_close(old, state.heap, atol=0, rtol=0)
            layer.consolidate(state, budget=0)
            recovered = tape.recover(0, leaf, 1, 2, batch=1)
            torch.testing.assert_close(recovered.value[0], layer.W_v(x)[1, 1], atol=0, rtol=0)
            refit = layer.recover_prediction(tape, recovered.key[0], 0, leaf, 1, 2, batch=1)
            torch.testing.assert_close(refit, recovered.value[0])
            next_out, _ = layer.step(x[:, 0], state)
            self.assertTrue(torch.isfinite(next_out).all())

    def test_archive_required_for_eviction(self):
        layer = SHRFLayer(SHRFConfig())
        with self.assertRaisesRegex(ValueError, "IDTape"):
            layer.consolidate(layer.init_state(1), 0)
