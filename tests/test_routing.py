import itertools
import unittest

import torch

from shrf import SHRFConfig, SHRFLayer
from shrf.routing import Routing


class RoutingTest(unittest.TestCase):
    def test_beam_mass_and_stable_ties(self):
        route = Routing(2, 3, 4, 0.2, 0.3, 2, 15)
        levels = route(torch.zeros(5, 2, 4))
        self.assertEqual(levels[-1].node[0, 0].tolist(), [8, 9])
        self.assertEqual([float(level.mass[0, 0].sum()) for level in levels], [1., 1., .5, .25])
        for previous, curr in itertools.pairwise(levels):
            torch.testing.assert_close(curr.node//2, previous.node.gather(-1, curr.parent))

    def test_calibration_invalidates_stale_state(self):
        torch.manual_seed(5)
        layer = SHRFLayer(SHRFConfig(d_model=4, n_trees=1, depth=1, r_key=4, d_value=1, dir_mix=1.0))
        old = layer.init_state(1)
        k = torch.randn(600, 1, 4)
        v = k[:, 0, :1].square()
        layer.refine_directions(k, v)
        self.assertGreater(layer.routing.dirs[0, 1, 0].abs().item(), .85)
        with self.assertRaisesRegex(ValueError, "Routing changed"):
            layer(torch.randn(1, 3, 4), old)
