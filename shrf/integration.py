"""RMSNorm/SwiGLU decoder block with SHRF in the attention slot."""
from __future__ import annotations
import torch
from torch import nn, Tensor
from .config import SHRFConfig
from .layer import SHRFLayer


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scale = torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * (norm_input * scale).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class CausalLocalMixer(nn.Module):
    """Learned low-rank FIR for the most recent inputs."""
    def __init__(self, width, lags=3, rank=32):
        super().__init__()
        if lags < 1 or rank < 1:
            raise ValueError("Local lag and rank must be positive")
        self.lags = lags
        self.reduce = nn.Linear(width, rank, bias=False)
        self.expand = nn.Linear(rank, width, bias=False)
        self.taps = nn.Parameter(torch.zeros(lags, rank))

    def forward(self, x, state=None):
        raw = self.reduce(x)
        if state is None:
            state = raw.new_zeros((x.shape[0], self.lags, raw.shape[-1]))
        past = torch.cat((state, raw), dim=1)
        mixed = torch.zeros_like(raw)
        for lag, tap in enumerate(self.taps.to(raw.dtype), start=1):
            mixed = mixed + past[:, self.lags-lag:self.lags-lag+x.shape[1]] * tap
        return self.expand(mixed), past[:, -self.lags:].clone()


def memory_output_gate(width, bias):
    gate = nn.Linear(width, 1)
    nn.init.zeros_(gate.weight)
    nn.init.constant_(gate.bias, bias)
    return gate


class LlamaSHRFBlock(nn.Module):
    """Llama-style block with SHRF memory instead of self-attention."""
    def __init__(self, d: int, cfg: SHRFConfig, mlp_hidden: int | None = None,
                 *, local_lags: int = 0, local_rank: int = 32,
                 memory_gate_bias: float | None = None):
        super().__init__()
        if cfg.d_model != d:
            raise ValueError("Decoder width must equal the SHRF projection input width")
        self.ln1 = RMSNorm(d)
        self.mem = SHRFLayer(cfg)
        self.local = CausalLocalMixer(d, local_lags, local_rank) if local_lags else None
        self.memory_gate = (memory_output_gate(d, memory_gate_bias)
                            if memory_gate_bias is not None else None)
        self.ln2 = RMSNorm(d)
        self.mlp = SwiGLU(d, mlp_hidden or int(2.7 * d))

    def forward(self, x, state=None, *, write_weight=None):
        normalized = self.ln1(x)
        if self.local is None or state is None:
            mem_state, local_state = state, None
        else:
            mem_state, local_state = state
        h, state = self.mem(normalized, mem_state, write_weight=write_weight)
        if self.memory_gate is not None:
            h = h * self.memory_gate(normalized).sigmoid()
        if self.local is not None:
            local, local_state = self.local(normalized, local_state)
            h = h + local
            state = (state, local_state)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, state


def shrf_llama_config(d: int = 512, depth: int = 8, n_trees: int = 4,
                      r: int = 8, q: int = 64) -> SHRFConfig:
    return SHRFConfig(d_model=d, n_trees=n_trees, depth=depth, r_key=r,
                      d_value=q, sketch_rank=8, beam=2, chunk_size=32, tau=0.2)


if __name__ == "__main__":
    cfg = shrf_llama_config(d=128, depth=6)
    blk = LlamaSHRFBlock(128, cfg)
    x = torch.randn(2, 32, 128)
    y, st = blk(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    print("LlamaSHRFBlock OK:", tuple(y.shape), "params", sum(p.numel() for p in blk.parameters()))
