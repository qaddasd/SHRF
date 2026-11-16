import unittest
import torch
from shrf.scans import causal_prefix, exclusive_segmented_cumsum


class ScanTest(unittest.TestCase):
    def test_scan_against_brute_force(self):
        group = torch.tensor([1,1,1,3,3,4,4,4])
        x = torch.randn(8, 5).requires_grad_()
        actual = exclusive_segmented_cumsum(x, group)
        expected = torch.stack([x[(group == group[i]) & (torch.arange(8) < i)].sum(0) for i in range(8)])
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(x.grad[:, 0], torch.tensor([2.,1.,0.,1.,0.,2.,1.,0.]))

    def test_distinct_read_write_groups(self):
        reads = torch.tensor([[[[2]], [[2]], [[0]], [[1]]]])
        writes = torch.tensor([[[[2]], [[0]], [[1]], [[1]]]])
        vals = torch.tensor([2.,3.,5.,7.]).reshape(1,4,1,1,1)
        actual = causal_prefix(reads, writes, vals).flatten()
        torch.testing.assert_close(actual, torch.tensor([0.,2.,3.,5.]))
