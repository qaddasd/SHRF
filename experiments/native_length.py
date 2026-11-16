"""Run a fitted needle model on million-token streams through the Rust decoder."""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.harness import train_eval
from experiments.tasks import niah
from shrf import SHRFConfig
from shrf.rust_backend import RustSHRF


@torch.no_grad()
def predict(model, decoder, ids, block_size):
    decoder.reset()
    block = model.blocks[0]
    for piece in ids.split(block_size):
        x = model.embed(piece)
        memory = decoder.prefill(block.ln1(x))
    x = x[-1:] + memory[-1:]
    x = x + block.mlp(block.ln2(x))
    return model.head(model.ln_f(x))[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=1_000_000)
    p.add_argument("--fractions", type=float, nargs="+", default=[.05, .5])
    p.add_argument("--training-device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--output", type=Path, default=Path("results/native_length.json"))
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = SHRFConfig(d_model=48, n_trees=2, depth=3, r_key=8, d_value=16,
                     sketch_rank=6, tau=.14, chunk_size=192)
    train, model = train_eval(niah, {"ctx_len":80}, cfg, steps=160, batch=8,
                              device=args.training_device, seed=73, return_model=True)
    model = model.cpu().eval()
    torch.set_num_threads(1)
    report = dict(length=args.length, training=train, episodes=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with RustSHRF(model.blocks[0].mem) as decoder, torch.no_grad():
        torch.manual_seed(161)
        ids, _, _, _, _ = niah(1, ctx_len=180)
        expected = model(ids)[0][0,-1]
        torch.testing.assert_close(predict(model, decoder, ids[0], 64), expected,
                                   atol=2e-4, rtol=2e-4)
        for i, fraction in enumerate(args.fractions):
            torch.manual_seed(1591+i)
            ids, target, _, _, _ = niah(1, ctx_len=args.length, needle_fraction=fraction)
            start = time.perf_counter()
            actual = int(predict(model, decoder, ids[0], 4096).argmax())
            expected = int(target[0,-1])
            row = dict(fraction=fraction, expected=expected, predicted=actual, correct=actual==expected,
                       seconds=time.perf_counter()-start)
            report["episodes"].append(row)
            args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            print(row, flush=True)


if __name__ == "__main__":
    main()
