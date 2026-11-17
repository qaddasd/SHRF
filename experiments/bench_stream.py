"""Measure one-layer prefill against a selectable SHRF source tree."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--label", default="current")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("results/prefill.json"))
    args = p.parse_args()
    if args.source is not None:
        sys.path.insert(0, str(args.source.resolve()))
    from shrf import SHRFConfig, SHRFLayer

    torch.set_num_threads(1)
    torch.manual_seed(118)
    c = SHRFConfig(d_model=512, n_trees=4, depth=8, r_key=8, d_value=32, sketch_rank=8,
                   chunk_size=64, beam=2)
    layer = SHRFLayer(c).to(args.device).eval()
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    rows = []
    for length in (256, 1024, 4096):
        x = torch.randn(1, length, c.d_model, device=args.device)
        for _ in range(2):
            layer(x)
        synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            layer(x)
            synchronize()
            times.append(time.perf_counter()-start)
        seconds = statistics.median(times)
        row = {"length": length, "seconds": seconds, "tokens_s": length/seconds, "repeats": args.repeats}
        rows.append(row)
        print(args.label, row, flush=True)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps({
        "label": args.label, "torch": torch.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "shape": {"d": 512, "trees": 4, "depth": 8, "r": 8, "q": 32, "m": 8, "beam": 2, "chunk": 64}, "rows": rows},
        indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
