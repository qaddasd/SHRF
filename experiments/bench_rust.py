"""Compare recurrent SHRF CPU latency with identical trained equations and inputs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from shrf import SHRFConfig, SHRFLayer
from shrf.rust_backend import RustSHRF


@torch.no_grad()
def bench(name, cfg, tokens, repeats):
    torch.manual_seed(21)
    layer = SHRFLayer(cfg).eval()
    x = torch.randn(tokens, cfg.d_model)
    stream = list(x.unbind())
    with RustSHRF(layer) as rust:
        st = layer.init_state(1)
        limit = min(64, tokens)
        expected = torch.stack([layer.step(a[None], st)[0][0] for a in stream[:limit]])
        native = torch.stack([rust.step(a) for a in stream[:limit]])
        err = (expected-native).abs().max().item()
        torch.testing.assert_close(expected, native, atol=2e-4, rtol=2e-3)
        py, rs = [], []
        for _ in range(repeats):
            st = layer.init_state(1)
            t0 = time.perf_counter()
            for row in stream:
                layer.step(row[None], st)
            py.append(time.perf_counter()-t0)
            rust.reset()
            t0 = time.perf_counter()
            for row in stream:
                rust.step(row)
            rs.append(time.perf_counter()-t0)
    p, r = statistics.median(py), statistics.median(rs)
    return dict(name=name, d=cfg.d_model, E=cfg.n_trees, D=cfg.depth, r=cfg.r_key,
                q=cfg.d_value, m=cfg.sketch_width, beam=cfg.beam, tokens=tokens, repeats=repeats,
                state_python="float32", state_rust="float64", max_abs_diff=err,
                torch_ms_per_token=p/tokens*1e3, rust_ms_per_token=r/tokens*1e3, speedup=p/r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(1)
    rows = [bench("small", SHRFConfig(d_model=64, n_trees=4, depth=6, r_key=8,
                                     d_value=32, sketch_rank=8), args.tokens, args.repeats),
            bench("d4096", SHRFConfig(d_model=4096, n_trees=4, depth=11, r_key=8,
                                     d_value=64, sketch_rank=8), args.tokens, args.repeats)]
    result = dict(torch=torch.__version__, cpu=os.environ.get("PROCESSOR_IDENTIFIER"), runs=rows)
    Path("results").mkdir(exist_ok=True)
    Path("results/rust_bench.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
