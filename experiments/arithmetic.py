"""Train and verify the optional SHRF decimal-arithmetic specialist."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from shrf.arithmetic import (
    DecimalArithmetic,
    DigitCell,
    structured_answer,
    transition_examples,
)

from .curriculum import greedy, numeric_equal, save_json
from .fable import atomic_write
from .language import TinyDecoder


def fit(device, width, steps, seed):
    torch.manual_seed(seed)
    cell = DigitCell(width).to(device)
    optimizer = torch.optim.AdamW(cell.parameters(), lr=.002)
    rows, labels = (x.to(device) for x in transition_examples())
    history = []
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        prediction = cell(rows)
        loss = F.cross_entropy(prediction.flatten(0, 1), labels.flatten())
        loss.backward()
        optimizer.step()
        if step not in (1, steps) and step % 50:
            continue
        with torch.no_grad():
            correct = (cell(rows).argmax(-1) == labels).all(-1).sum().item()
        history.append({"step": step, "loss": loss.item(), "correct": correct, "total": len(rows)})
        print(json.dumps(history[-1]), flush=True)
        if correct == len(rows):
            break
    return cell.eval(), history


def long_cases(engine, seed):
    rng = random.Random(seed)
    results = {}
    for length in (2, 8, 24):
        for name, apply, truth in (
            ("addition", engine.add, lambda a, b: a+b),
            ("subtraction", engine.subtract, lambda a, b: a-b),
            ("multiplication", engine.multiply, lambda a, b: a*b),
            ("division", engine.divmod, divmod),
        ):
            correct = 0
            for _ in range(250):
                divisor_digits = rng.randrange(1, length+1) if name == "division" else length
                raw = [rng.randrange(10**(n-1), 10**n) for n in (length, divisor_digits)]
                a, b = (value*rng.choice((-1, 1)) for value in raw)
                correct += apply(a, b) == truth(a, b)
            results[f"{name}_{length}_digits"] = {"correct": correct, "total": 250}
    return results


def prompt_cases():
    rng = random.Random(483)
    for i in range(16):
        a, b, x = rng.sample(range(37, 100), 3)
        sign = rng.choice((-1, 1))
        b *= sign
        yield "arithmetic", f"Compute ({a} + {x}) * {b}.", (a+x)*b
        yield "algebra", f"Solve {a}x {'+' if b > 0 else '-'} {abs(b)} = {a*x+b} for x.", x
        if i % 2 == 0:
            yield "physics", f"A {a} kg body accelerates at {x} m/s^2. Find the net force in N.", a*x
        else:
            yield "physics", f"A cyclist covers {a*x} m in {a} seconds. Find the average speed in m/s.", x
        yield "python_expressions", f"What does Python print?\nprint(({a} * {b}) % {x})", (a*b)%x


def compare(engine, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arch = saved["metadata"]["architecture"]
    parent = TinyDecoder(**arch).eval()
    parent.load_state_dict(saved["model"])
    report = {}
    for domain, prompt, answer in prompt_cases():
        row = report.setdefault(domain, {"parent_correct": 0, "combined_correct": 0, "total": 0})
        prompt += " Reply with only the number."
        text = greedy(parent, prompt, "cpu", arch["context"], 24, numeric=True)
        result = structured_answer(engine, prompt)
        row["parent_correct"] += numeric_equal(text, str(answer))
        row["combined_correct"] += result == str(answer)
        row["total"] += 1
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=843)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cell, history = fit(args.device, args.width, args.steps, args.seed)
    engine = DecimalArithmetic(cell)
    verification = long_cases(engine, args.seed+1)
    assert all(v["correct"] == v["total"] for v in verification.values())
    with atomic_write(args.output/"digit_cell.pt") as path:
        torch.save({"width": args.width, "model": cell.state_dict()}, path)
    report = {
        "training": history, "composed_integer_tests": verification,
        "parameters": sum(p.numel() for p in cell.parameters()),
    }
    save_json(args.output/"arithmetic_metrics.json", report)
    if args.compare:
        report["structured_prompts"] = compare(engine, args.compare)
        save_json(args.output/"arithmetic_metrics.json", report)
        print(json.dumps(report["structured_prompts"]), flush=True)


if __name__ == "__main__":
    main()
