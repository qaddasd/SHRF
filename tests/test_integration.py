import unittest

import torch

from experiments.language import GQAMixer
from shrf import SHRFConfig
from shrf.integration import LlamaSHRFBlock


class IntegrationTest(unittest.TestCase):
    @torch.no_grad()
    def test_full_and_cached_gqa(self):
        mixer = GQAMixer(32).eval()
        x = torch.randn(2, 11, 32)
        expected, _ = mixer(x)
        _, state = mixer(x[:, :3])
        suffix, state = mixer(x[:, 3:10], state)
        final, _ = mixer(x[:, 10:], state)
        torch.testing.assert_close(torch.cat((suffix, final), 1), expected[:, 3:])

    @torch.no_grad()
    def test_contextual_local_block_causality_and_stream(self):
        cfg = SHRFConfig(d_model=16, depth=2, chunk_size=4, context_lags=2)
        block = LlamaSHRFBlock(16, cfg, local_lags=3, local_rank=8, memory_gate_bias=-3.).eval()
        block.local.taps.normal_(std=.2)
        block.mem.context_mix.normal_(std=.2)
        x = torch.randn(2, 11, 16)
        full, _ = block(x)
        _, state = block(x[:, :3])
        suffix, _ = block(x[:, 3:], state)
        torch.testing.assert_close(suffix, full[:, 3:], atol=4e-6, rtol=1e-5)
        x[:, 6:] += 1.5
        changed, _ = block(x)
        torch.testing.assert_close(changed[:, :6], full[:, :6], atol=4e-6, rtol=1e-5)
