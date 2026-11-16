"""Train component ablations; corrupt representations of the same full model."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch

from experiments import tasks
from experiments.harness import evaluate, train_eval
from experiments.run_all import BASE
from shrf import SHRFConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/ablations_v2.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    options = dict(full={}, no_sketch={"use_sketch": False}, no_shrinkage={"use_shrinkage": False},
                   E1={"n_trees": 1}, E4={"n_trees": 4}, r4={"r_key": 4}, r12={"r_key": 12},
                   m2={"sketch_rank": 2}, m12={"sketch_rank": 12}, no_binding={"write_lag": 0},
                   beam1={"beam": 1})
    report = dict(device=args.device, steps=args.steps, seed=args.seed, runs={}, robustness={})
    path = args.output
    path.parent.mkdir(exist_ok=True, parents=True)
    for name, opt in options.items():
        cfg = SHRFConfig(**(BASE | opt))
        result, model = train_eval(tasks.semantic_recall, {}, cfg, steps=args.steps,
                                   batch=args.batch, device=args.device, seed=args.seed,
                                   val_kwargs={"held_out": True}, return_model=True)
        result["overload"] = evaluate(model, tasks.semantic_recall, {"held_out": True, "n_items": 16})
        result["noisy_overload"] = evaluate(model, tasks.semantic_recall,
                                            {"held_out": True, "n_items": 16}, noise=0.75)
        report["runs"][name] = dict(result, config=asdict(cfg))
        if name == "full":
            kwargs = {"held_out": True}
            report["robustness"] = {
                "noise": {str(n): evaluate(model, tasks.semantic_recall, kwargs, noise=n)
                          for n in (0., .2, .5, 1., 2.)},
                "rank": {str(n): evaluate(model, tasks.semantic_recall, kwargs, collapse_rank=n)
                         for n in (16, 8, 4, 2, 0)},
            }
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(name, result["eval"], flush=True)


if __name__ == "__main__":
    main()
