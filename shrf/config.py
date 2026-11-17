"""SHRF hyperparameters shared by training and recurrent inference."""
import math
from dataclasses import dataclass


@dataclass
class SHRFConfig:
    d_model: int = 128
    n_trees: int = 2
    depth: int = 5
    r_key: int = 8
    d_value: int = 32
    sketch_rank: int = 8
    beam: int = 2
    ridge: float = 0.05
    sketch_ridge: float = 0.1
    sketch_bandwidth: float = 2.0
    tau: float = 0.2
    dir_mix: float = 0.35
    kappa_min: float = 0.25
    kappa_max: float = 256.0
    chunk_size: int = 64
    write_lag: int = 1
    sketch_budget: int | None = None
    use_sketch: bool = True
    use_shrinkage: bool = True
    detach_chunks: bool = False
    load_balance_weight: float = 0.01
    seed: int = 0
    context_lags: int = 0
    gate_power: float = 1.0
    coalesce_writes: bool = True
    decay: float = 1.0

    def __post_init__(self):
        if min(self.d_model, self.n_trees, self.r_key, self.d_value, self.beam, self.chunk_size) < 1:
            raise ValueError("Model dimensions, beam and chunk size must be positive")
        if not 0 <= self.depth <= 20 or self.write_lag not in (0, 1):
            raise ValueError("Depth must be 0–20 and write_lag must be 0 or 1")
        scales = (self.ridge, self.sketch_ridge, self.sketch_bandwidth, self.tau,
                  self.kappa_min, self.kappa_max)
        if any(not math.isfinite(s) or s <= 0 for s in scales) or self.kappa_max < self.kappa_min:
            raise ValueError("Invalid ridge, routing, or shrinkage scale")
        if not 0 <= self.dir_mix <= 1 or self.sketch_rank < 0:
            raise ValueError("Invalid direction mixture or sketch size")
        if not isinstance(self.context_lags, int) or not 0 <= self.context_lags <= 16:
            raise ValueError("context_lags must be an integer in [0, 16]")
        if not math.isfinite(self.gate_power) or not 1 <= self.gate_power <= 16:
            raise ValueError("gate_power must be finite and in [1, 16]")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if self.sketch_budget is not None and not 0 <= self.sketch_budget <= self.n_leaves:
            raise ValueError("Sketch budget must be a per-tree leaf count")

    @property
    def n_nodes(self):
        return 2 ** (self.depth + 1)

    @property
    def n_leaves(self):
        return 2 ** self.depth

    @property
    def sketch_width(self):
        return self.sketch_rank if self.use_sketch else 0
