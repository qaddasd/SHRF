"""Paired length extrapolation of the learned write gate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.harness import evaluate, train_eval
from experiments.tasks import niah
from shrf import SHRFConfig


@torch.no_grad()
def gate_mass(model, device, seed):
    torch.manual_seed(seed)
    ids, _, score, _, meta = niah(32, ctx_len=512, device=device)
    block = model.blocks[0]
    gate = block.mem.write_gate(block.ln1(model.embed(ids))).sigmoid().squeeze(-1)
    gate = gate.pow(block.mem.cfg.gate_power)
    pos = meta["write_weight"].bool()
    filler = ~pos & ~score & (ids >= 33)
    def summary(x):
        return dict(mean=x.mean().item(), median=x.median().item())
    return dict(payload=summary(gate[pos]), filler=summary(gate[filler]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 32768, 128000])
    p.add_argument("--power", type=float, nargs="+", default=[1, 4])
    p.add_argument("--output", type=Path, default=Path("results/length.json"))
    args = p.parse_args()
    torch.set_num_threads(2)
    report = dict(device=args.device, torch=torch.__version__, runs={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for power in args.power:
        cfg = SHRFConfig(d_model=48, n_trees=2, depth=3, r_key=8, d_value=16,
                         sketch_rank=6, beam=2, tau=.14, chunk_size=192, gate_power=power)
        train, model = train_eval(niah, {"ctx_len": 80}, cfg, steps=args.steps,
                                  batch=args.batch, device=args.device, seed=args.seed,
                                  return_model=True)
        row = dict(config=asdict(cfg), training=train, gates=gate_mass(model, args.device, args.seed+1),
                   sweep={})
        report["runs"][str(power)] = row
        for length in args.lengths:
            for fraction in (0., .05) if length <= 2048 else (0.,):
                spec = dict(ctx_len=length, needle_fraction=.05, decoy_fraction=fraction)
                val = evaluate(model, niah, spec, batch=4, batches=4, seed=args.seed+1805)
                label = f"{length}/{fraction:g}"
                row["sweep"][label] = val
                args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
                print(f"gate={power:g}, length={label}: {val}", flush=True)


if __name__ == "__main__":
    main()
