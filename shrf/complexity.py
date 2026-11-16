"""Calculate resident tensor bytes and leading arithmetic FLOPs."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    d: int = 4096
    layers: int = 32
    trees: int = 4
    depth: int = 11
    r: int = 8
    q: int = 64
    m: int = 8
    beam: int = 2
    chunk: int = 64
    context_lags: int = 0
    local_lags: int = 0
    local_rank: int = 32
    memory_gated: bool = False


def memory_and_flops(c):
    d, E, D, r, q, m = c.d, c.trees, c.depth, c.r, c.q, c.m
    J, N = 2**D, 2**(D+1)
    H = 1 + sum(min(c.beam, 2**j) for j in range(1, D+1))
    W = min(c.beam, J) if m else 0
    S, X = 2 + r + q + r*r + r*q, m + r*m + m*m + m*q

    def solve(p):
        return 2*p**3/3 + 2*p*p*q

    def ridge(p):
        return solve(p) + 4*p*p + 5*p*q + 3*(p+q) + 2*q

    parts = {
        "projections": 2*d*(E*r + 2*q + 1),
        "read_and_write_routing": 2*E*((H-min(c.beam, J))*(2*r+6)),
        "base_moments": E*(r*r + r*q + 2*q),
        "base_weight_and_write": 2*E*H*S,
        "base_read": E*H*ridge(r),
        "shrinkage_and_prefix_mixture": 9*E*(H-1)*q,
        "sketch_features_and_moments": E*(4*r*m + r*m + m*m + m*q),
        "sketch_weight_and_write": 2*E*W*X,
        "sketch_joint_read": E*W*ridge(r+m),
        "contextual_keys": 2*c.context_lags*E*r,
        "local_value_mixer": ((4*d+2*c.local_lags)*c.local_rank+d if c.local_lags else 0),
        "memory_output_gate": 3*d if c.memory_gated else 0,
    }
    decode = sum(parts.values())
    # Successor read/write routes and feature evaluations share C-1 positions.
    shared_prefill = (1-1/c.chunk)*(parts["read_and_write_routing"]/2 + 2*E*r*m)
    prefill = decode + 4*E*(H*S + W*X)
    prefill -= shared_prefill
    prefill += E*(H*S + W*X)  # Upper bound for the coalesced detached update reduction.

    def state(b, slots):
        # Sketch slot zero is a shared sentinel within each tree and batch.
        sketch = (slots+1)*X if m else 0
        carry = E*r*c.context_lags + c.local_lags*c.local_rank
        return c.layers*(b*(E*(N*S + sketch + r) + carry) + (8*E*J if m else 0) + 8)

    # Packed idealization for per-layer cold archive, before indexing or compression.
    # A token stores Er keys, q values, one gate and <= E*W leaf IDs and weights.
    memberships = E*min(c.beam, J)
    cold_per_token = c.layers*(2*(E*r + q + 1) + 4*memberships)
    cold_fp32 = c.layers*(4*(E*r + q + 1) + 6*memberships)
    return dict(shape=dict(S=S, X=X, active_nodes=H, active_leaves=W),
                hot_fp32_bytes=state(4, J), hot_fp16_storage_bytes=state(2, J),
                hot_fp32_half_slots_bytes=state(4, J//2),
                decode_flops_per_layer=decode, prefill_flops_per_layer=prefill,
                decode_breakdown=parts, prefill_shared_flops=shared_prefill,
                ideal_packed_cold_bytes_per_token=cold_per_token,
                ideal_packed_fp32_cold_bytes_per_token=cold_fp32)


def attention(d, dkv, t, layers):
    proj = 4*d*d + 4*d*dkv
    return dict(kv_bytes=4*layers*t*dkv, decode_flops_per_layer=proj+4*t*d,
                prefill_flops_per_layer=proj+2*(t+1)*d)


def build_report(c=Config()):
    shrf = memory_and_flops(c)
    rows = []
    for t in (128_000, 1_000_000, 10_000_000):
        mha, gqa = (attention(c.d, width, t, c.layers) for width in (c.d, c.d//4))
        rows.append(dict(tokens=t, mha=mha, gqa8=gqa,
                         mha_hot_ratio=mha["kv_bytes"]/shrf["hot_fp32_bytes"],
                         gqa8_hot_ratio=gqa["kv_bytes"]/shrf["hot_fp32_bytes"],
                         ideal_packed_cold_bytes=t*shrf["ideal_packed_cold_bytes_per_token"]))
    compact = Config(**(asdict(c) | {"depth": min(c.depth, 9)}))
    local = Config(**(asdict(compact) | {"context_lags": 2, "local_lags": 3,
                                         "memory_gated": True}))
    return dict(config=asdict(c), shrf=shrf, contexts=rows,
                compact=dict(config=asdict(compact), shrf=memory_and_flops(compact)),
                contextual_local=dict(config=asdict(local), shrf=memory_and_flops(local)))


if __name__ == "__main__":
    result = build_report()
    Path("results").mkdir(exist_ok=True)
    Path("results/complexity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
