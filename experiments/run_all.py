"""Run trained mechanism probes on held-out episodes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch

from shrf import SHRFConfig
from experiments import tasks
from experiments.harness import evaluate, train_eval


BASE = dict(d_model=48, n_trees=2, depth=3, r_key=8, d_value=16,
            sketch_rank=6, beam=2, tau=0.14, chunk_size=64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--long", type=int, nargs="*", default=[512, 2048, 8192])
    parser.add_argument("--oracle-writes", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/experiments.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    settings = [
        ("mqar", tasks.mqar, {}, {}),
        ("semantic_alias", tasks.semantic_recall, {}, {"held_out": True}),
        ("induction_span", tasks.copy_span, {}, {}),
        ("dense_cluster", tasks.dense_clusters, {}, {}),
        ("needle", tasks.niah, {"ctx_len": 80}, {"ctx_len": 80}),
    ]
    cfg = SHRFConfig(**BASE)
    report = dict(device=args.device, torch=torch.__version__, config=asdict(cfg), runs={})
    args.output.parent.mkdir(exist_ok=True, parents=True)
    for name, task, kw, val in settings:
        result, model = train_eval(task, kw, cfg, steps=args.steps, batch=args.batch,
                                   device=args.device, seed=args.seed, val_kwargs=val,
                                   oracle=args.oracle_writes, return_model=True)
        if name == "induction_span":
            result["rollout"] = evaluate(model, task, val, rollout=True, oracle=args.oracle_writes)
        if name == "needle":
            result["distance"] = {}
            for length in args.long:
                fractions = (0.05, 0.5, 0.95) if length < 32_000 else (0.05,)
                for fraction in fractions:
                    result["distance"][f"{length}@{fraction}"] = evaluate(
                        model, task, dict(ctx_len=length, needle_fraction=fraction),
                        batch=8 if length < 32_000 else 4, batches=2, oracle=args.oracle_writes)
            result["decoys"] = {
                str(f): evaluate(model, task, dict(ctx_len=2048, needle_fraction=0.05, decoy_fraction=f),
                                 batch=8, batches=4)
                for f in (0.0, 0.01, 0.05, 0.25)
            }
        if name == "dense_cluster":
            result["stress"] = {str(n): evaluate(model, task, dict(n_items=n), oracle=args.oracle_writes)
                                for n in (4, 12, 20)}
        report["runs"][name] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(name, result["eval"], "query-only", result["query_only"], flush=True)


if __name__ == "__main__":
    main()
