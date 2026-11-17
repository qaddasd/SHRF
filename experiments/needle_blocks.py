"""Isolate the forest's long-range contribution beside the local value mixer."""
import argparse
import json
from pathlib import Path

import torch

from experiments.harness import evaluate, train_eval
from experiments.language import TinyDecoder
from experiments.tasks import niah
from shrf import SHRFConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--output", type=Path, default=Path("results/needle_blocks.json"))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = SHRFConfig(d_model=48)
    report = {"device": args.device, "width": 48, "layers": 2, "steps": args.steps, "runs": {}}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    for kind in ("local_only", "shrf_gated"):
        def factory(vocab, width, layers, config, kind=kind):
            return TinyDecoder(kind, width, 2, 128)
        train, model = train_eval(niah, {"ctx_len": 96, "needle_fraction": .05}, cfg,
                                  steps=args.steps, batch=8, device=args.device, seed=33,
                                  eval_batches=4, eval_batch=32, return_model=True,
                                  model_factory=factory)
        novel = evaluate(model, niah, {"ctx_len": 512, "needle_fraction": .05},
                         batch=16, batches=8, seed=41)
        report["runs"][kind] = {"train": train, "accuracy_512": novel}
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(kind, train["eval"], novel, flush=True)


if __name__ == "__main__":
    main()
