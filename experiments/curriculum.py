"""Continue a SHRF checkpoint on balanced short instruction tasks."""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .curriculum_data import (
    generated,
    gsm_cases,
    openassistant_examples,
    public_examples,
    teacher_examples,
)
from .fable import ASSISTANT_END, ASSISTANT_START, IGNORE, atomic_write, evaluate
from .language import TinyDecoder


def prompt_bytes(prompt):
    return b"<user>\n" + prompt.encode() + b"\n</user>\n" + ASSISTANT_START


def encode(case, context):
    prefix = prompt_bytes(case.prompt)
    answer = case.answer.encode()
    if len(prefix) + len(answer) + len(ASSISTANT_END) > context + 1 or not answer:
        return None
    data = list(prefix + answer + ASSISTANT_END)
    labels = [IGNORE] * (len(prefix) - 1) + data[len(prefix):]
    emphasized = len(case.number.encode()) if case.number and case.answer.endswith(case.number) else 0
    importance = ([0.] * (len(prefix)-1) + [1.] * (len(answer)-emphasized)
                  + [4.] * emphasized + [.2] * len(ASSISTANT_END))
    return data[:-1], labels, importance


def batch_cases(cases, device, *, evaluation=False):
    length = max(len(ids) for ids, _, _ in cases)
    inputs, targets, valid, weights = [], [], [], []
    for ids, labels, importance in cases:
        pad = length - len(ids)
        inputs.append(ids + [0]*pad)
        if evaluation:
            labels = [label if weight >= 1 else IGNORE
                      for label, weight in zip(labels, importance, strict=True)]
        targets.append(labels + [IGNORE]*pad)
        valid.append([1.] * len(ids) + [0.]*pad)
        weights.append(importance + [0.]*pad)
    return tuple(torch.tensor(value, device=device) for value in (inputs, targets, valid, weights))


def prompt_key(prompt):
    return " ".join(prompt.casefold().split())


def problem_key(prompt):
    return re.sub(r" (?:reply with only the number|explain briefly)\.$", "", prompt_key(prompt))


def prepare(args):
    rng = random.Random(args.seed)
    eval_cases = defaultdict(list)
    for row in generated(args.seed + 1, 240, heldout=True):
        eval_cases[row.domain].append(row)
    eval_cases["gsm8k"] = list(gsm_cases(args.sources/"gsm_test.jsonl", heldout=True))
    reserved = {problem_key(row.prompt) for group in eval_cases.values() for row in group}
    training = defaultdict(list)
    teacher_training = defaultdict(list)
    teacher_counts = Counter()
    counts = Counter()
    seen = set()
    seen_eval = set()
    digest = hashlib.sha256()
    public = list(public_examples(args.sources, args.dolly))
    if args.oasst:
        dialogues = list(openassistant_examples(args.oasst))
        rng.shuffle(dialogues)
        for language in ("en", "ru"):
            heldout = [row for row in dialogues if row.domain == f"writing_{language}"
                       and encode(row, args.context) is not None][:64]
            eval_cases[f"openassistant_{language}"] = heldout
            reserved.update(problem_key(row.prompt) for row in heldout)
        public.extend(dialogues)
    rng.shuffle(public)
    teacher = list(teacher_examples(args.teacher)) if args.teacher else []
    rng.shuffle(teacher)
    for row, is_teacher in [
        *((row, True) for row in teacher),
        *((row, False) for row in public),
        *((row, False) for row in generated(args.seed, 1400)),
    ]:
        key = prompt_key(row.prompt)
        sample = encode(row, args.context)
        original = problem_key(row.original_prompt or row.prompt)
        if sample is None or problem_key(row.prompt) in reserved or original in reserved or key in seen:
            continue
        limit = {"Dolly": 1800, "OpenAssistant-en": 1000, "OpenAssistant-ru": 1800}.get(row.source)
        if limit is not None and counts[row.source] >= limit:
            continue
        seen.add(key)
        target = teacher_training if is_teacher else training
        target[row.domain].append(sample)
        counts[row.source] += 1
        if is_teacher:
            teacher_counts[row.source] += 1
        digest.update((json.dumps(vars(row), ensure_ascii=False, sort_keys=True)+"\n").encode())
    evaluation = {}
    eval_rng = random.Random(args.seed + 3)
    for domain, rows in eval_cases.items():
        eval_rng.shuffle(rows)
        selected = []
        wanted = 32 if domain == "gsm8k" else 16
        for row in rows:
            key = prompt_key(row.prompt)
            if encode(row, args.context) is None or key in seen_eval:
                continue
            seen_eval.add(key)
            selected.append(row)
            if len(selected) == wanted:
                break
        if len(selected) != wanted:
            raise ValueError(f"{domain} has only {len(selected)} eligible evaluation examples")
        evaluation[domain] = selected
    return training, teacher_training, evaluation, dict(counts), dict(teacher_counts), digest.hexdigest()


def val_batches(evaluation, args):
    validation = {}
    for domain, cases in evaluation.items():
        encoded = [encode(example, args.context) for example in cases]
        validation[domain] = [
            batch_cases(encoded[i:i+args.batch], args.device, evaluation=True)[:2]
            for i in range(0, len(encoded), args.batch)
        ]
    return validation


def score(model, validation):
    return {domain: evaluate(model, batches)[0] for domain, batches in validation.items()}


@torch.inference_mode()
def greedy(model, prompt, device, context, max_bytes, numeric=False):
    prefix = torch.tensor([list(prompt_bytes(prompt)[-context:])], device=device)
    logits, state, _ = model(prefix)
    output = bytearray()
    for _ in range(max_bytes):
        token = logits[:, -1].argmax(-1, keepdim=True)
        output.append(token.item())
        if output.endswith(ASSISTANT_END):
            del output[-len(ASSISTANT_END):]
            break
        if numeric and b"\n" in output:
            break
        logits, state, _ = model(token, state)
    return output.decode("utf-8", errors="replace").strip()


def numeric_equal(text, reference):
    match = re.fullmatch(r"\s*(?:Answer:\s*)?([+-]?\d+(?:\.\d+)?)\s*\.?\s*", text, re.IGNORECASE)
    return bool(match and Decimal(match[1]) == Decimal(reference))


def exact_scores(model, evaluation, args):
    model.eval()
    report = {}
    for domain, cases in evaluation.items():
        if not cases[0].number:
            continue
        correct = sum(numeric_equal(
            greedy(model, row.prompt, args.device, args.context, 32, numeric=True), row.number,
        ) for row in cases)
        report[domain] = {"correct": correct, "total": len(cases)}
    return report


def save_json(path, data):
    with atomic_write(path) as temp:
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def train(args):
    if min(args.steps, args.batch, args.context, args.eval_every) < 1 or args.lr <= 0:
        raise ValueError("Training dimensions and learning rate must be positive")
    if not 0 <= args.teacher_fraction <= 1:
        raise ValueError("Teacher fraction must be in [0, 1]")
    parent_bytes = args.checkpoint.read_bytes()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    arch = {**parent["metadata"]["architecture"], "context": args.context}
    model = TinyDecoder(**arch).to(args.device)
    model.load_state_dict(parent["model"])
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    training, teachers, evaluation, counts, teacher_counts, fingerprint = prepare(args)
    validation = val_batches(evaluation, args)
    metadata = {
        "architecture": arch, "source": "balanced STEM and writing curriculum",
        "parent_steps": parent["step"], "parent_sha256": parent_hash,
        "examples_per_source": counts, "examples_per_domain": {k: len(v) for k, v in training.items()},
        "teacher_examples_per_source": teacher_counts, "teacher_fraction": args.teacher_fraction,
        "teacher_examples_per_domain": {k: len(v) for k, v in teachers.items()},
        "training_sha256": fingerprint, "seed": args.seed, "batch": args.batch, "lr": args.lr,
        "objective": "assistant text with 4x weight on numeric answers and 0.2x on end tags",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"latest.pt").exists():
        raise ValueError("Choose a new output directory for a new training stage")
    print(json.dumps(metadata), flush=True)
    report = {"metadata": metadata, "steps_requested": args.steps, "curve": []}
    start = perf_counter()
    domains = sorted(training)
    heldout_bpb = score(model, validation)
    report["curve"].append({"stage_step": 0, "bpb": heldout_bpb})
    save_json(args.output/"metrics.json", report)
    print(json.dumps(report["curve"][-1]), flush=True)
    best = math.inf
    assistant_bytes = 0
    teacher_examples_seen = 0
    for step in range(1, args.steps+1):
        if (step-1) % len(domains) == 0:
            rng.shuffle(domains)
        domain = domains[(step-1) % len(domains)]
        regular, distilled = training[domain], teachers.get(domain)
        sampled = []
        for _ in range(args.batch):
            use_teacher = bool(distilled) and rng.random() < args.teacher_fraction
            sampled.append(rng.choice(distilled if use_teacher else regular))
            teacher_examples_seen += int(use_teacher)
        ids, targets, write_weight, importance = batch_cases(sampled, args.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, states, aux = model(ids, write_weight=write_weight)
        del states
        byte_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                    ignore_index=IGNORE, reduction="none")
        loss = (byte_loss * importance.flatten()).sum() / importance.sum()
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        assistant_bytes += (targets != IGNORE).sum().item()
        if step != 1 and step % args.eval_every and step != args.steps:
            continue
        bpb = score(model, validation)
        macro = sum(bpb.values())/len(bpb)
        report["curve"].append({
            "stage_step": step, "bpb": bpb, "macro_bpb": macro,
            "samples": step*args.batch, "assistant_bytes": assistant_bytes,
            "teacher_examples_seen": teacher_examples_seen,
            "seconds": perf_counter()-start,
        })
        saved = {
            "model": model.state_dict(), "step": parent["step"]+step,
            "stage_step": step, "metadata": metadata,
        }
        with atomic_write(args.output/"latest.pt") as temp:
            torch.save({**saved, "optimizer": optimizer.state_dict(), "rng": rng.getstate()}, temp)
        if macro < best:
            best = macro
            with atomic_write(args.output/"best.pt") as temp:
                torch.save(saved, temp)
        save_json(args.output/"metrics.json", report)
        print(json.dumps(report["curve"][-1]), flush=True)
    report["exact_final"] = exact_scores(model, evaluation, args)
    model.load_state_dict(parent["model"])
    report["exact_parent"] = exact_scores(model, evaluation, args)
    save_json(args.output/"metrics.json", report)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("exact_")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "sources", "dolly", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--oasst", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-fraction", type=float, default=.50)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=320)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--lr", type=float, default=.0007)
    parser.add_argument("--seed", type=int, default=56)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    torch.set_num_threads(2)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
