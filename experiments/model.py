"""Small attention-free decoder used by the synthetic benchmarks."""
from __future__ import annotations
import torch
from torch import nn, Tensor
from shrf import SHRFConfig, SHRFLayer


class SHRFBlock(nn.Module):
    def __init__(self, cfg: SHRFConfig, mlp_mult: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mem = SHRFLayer(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, mlp_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, state=None, write_weight=None):
        h, state = self.mem(self.ln1(x), state, write_weight=write_weight)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, state


class SHRFModel(nn.Module):
    def __init__(self, vocab: int, d: int, n_layers: int, cfg: SHRFConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([SHRFBlock(cfg) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: Tensor, states=None, *, write_weight=None):
        h = self.embed(ids)
        new_states = []
        aux = 0.0
        for i, blk in enumerate(self.blocks):
            st = states[i] if states is not None else None
            h, st = blk(h, st, write_weight)
            new_states.append(st)
            aux = aux + blk.mem.last_aux
        return self.head(self.ln_f(h)), new_states, aux
