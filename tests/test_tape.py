import tempfile
import unittest
from io import BytesIO
from pathlib import Path
import torch
from shrf import SHRFConfig, SHRFLayer
from shrf.tape import IDTape, TapeConflictError


class TapeTest(unittest.TestCase):
    @torch.no_grad()
    def test_checkpoint_restores_archive_and_idempotent_replay(self):
        torch.manual_seed(14)
        layer = SHRFLayer(SHRFConfig(d_model=4, r_key=2, d_value=2, depth=2, decay=.95))
        x = torch.randn(1, 12, 4)
        with tempfile.TemporaryDirectory() as tmp:
            for path in (":memory:", Path(tmp) / "out.sqlite"):
                with IDTape(path, wal=True) as tape:
                    st = layer.init_state(1, tape=tape)
                    layer(x[:, :6], st)
                    buf = BytesIO()
                    torch.save(st, buf)
                    expected, _ = layer(x[:, 6:], st)
                    buf.seek(0)
                    restored = torch.load(buf, weights_only=False)
                    with restored.tape:
                        out, _ = layer(x[:, 6:], restored)
                        torch.testing.assert_close(expected, out)
                        self.assertEqual(restored.tape.recover_time(0, 12).time.tolist(), list(range(1, 12)))
                        if path != ":memory:":
                            self.assertEqual(restored.tape.db.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_retry_conflict_rolls_back_without_overwriting_history(self):
        k, v = torch.ones(1, 2, 1, 2), torch.zeros(1, 2, 3)
        gate = torch.ones(1, 2)
        leaves = torch.ones(1, 2, 1, 1, dtype=torch.long)
        mass = torch.ones_like(leaves, dtype=torch.float32)
        with IDTape() as tape:
            tape.record(torch.tensor([1]), k, v, gate, leaves, mass)
            tape.record(torch.tensor([1]), k, v, gate, leaves, mass)
            for changed_v, changed_leaf in ((v + 1, leaves), (v, leaves + 1)):
                with self.assertRaisesRegex(TapeConflictError, "t=1"):
                    tape.record(torch.tensor([0]), k, changed_v, gate, changed_leaf, mass)
                self.assertEqual(tape.recover_time(0, 3).time.tolist(), [1, 2])
            torch.testing.assert_close(tape.recover_time(0, 3).value, v[0])
            with self.assertRaisesRegex(TapeConflictError, "omitted"):
                tape.record(torch.tensor([1]), k, v, torch.zeros_like(gate), leaves, mass)
    def test_persist_exact_rows_by_batch_tree_and_time(self):
        k, v = torch.randn(2, 7, 2, 4), torch.randn(2, 7, 5)
        route = torch.ones(2, 7, 2, 1, dtype=torch.long)
        gate = torch.linspace(0,1,7).expand(2,-1)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work)/"archive.sqlite"
            with IDTape(path) as tape:
                tape.record(torch.tensor([0,2]), k, v, gate, route, torch.ones_like(route)*.25)
            with IDTape(path) as tape:
                out = tape.recover(1, 1, 4, 8, batch=1, limit=3)
                self.assertEqual(out.time.tolist(), [4,5,6])
                torch.testing.assert_close(out.key, k[1,2:5,1], atol=0, rtol=0)
                torch.testing.assert_close(out.value, v[1,2:5], atol=0, rtol=0)
                torch.testing.assert_close(out.weight, gate[1,2:5]*.25, atol=0, rtol=0)
                self.assertIsNone(tape.recover(0,0,0,8))
