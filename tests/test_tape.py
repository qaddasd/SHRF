import tempfile
import unittest
from pathlib import Path
import torch
from shrf.tape import IDTape


class TapeTest(unittest.TestCase):
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
