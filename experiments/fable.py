"""Train a byte-level SHRF decoder on a local Fable conversation JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from shrf.arithmetic import DecimalArithmetic, structured_answer

from .language import TinyDecoder

IGNORE = -100
ASSISTANT_START = b"<assistant>\n"
ASSISTANT_END = b"\n</assistant>\n"


def encode_text(value):
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def render_trace(messages):
    """Render roles and return their UTF-8 bytes and supervised assistant spans."""
    data = bytearray()
    spans = []
    for message in messages:
        role = message["role"]
        data.extend(f"<{role}>\n".encode("utf-8"))
        start = len(data)
        reasoning = message.get("reasoning_content")
        if reasoning:
            data.extend(b"<think>\n" + encode_text(reasoning) + b"\n</think>\n")
        content = message.get("content")
        if content is not None:
            data.extend(encode_text(content))
        text_end = len(data)
        calls = message.get("tool_calls")
        if calls:
            data.extend(b"\n<tool_calls>\n" + encode_text(calls) + b"\n</tool_calls>")
        data.extend(f"\n</{role}>\n".encode("utf-8"))
        if role == "assistant" and text_end > start:
            spans.append((start, text_end if calls else len(data)))
    return data, spans


class Corpus:
    def __init__(self, path, spans, rows, context):
        if not spans:
            raise ValueError("Both splits need assistant targets with a full context window")
        index = np.asarray(spans, dtype=np.int64)
        self.path = path
        self.data = np.memmap(path, mode="r", dtype=np.uint8)
        self.starts, self.stops, eligible = index.T
        self.eligible = eligible
        self.cumulative = np.cumsum(self.stops - eligible)
        self.relative = np.arange(-context, 1)
        self.metadata = {
            "traces": rows, "bytes": int(self.data.size), "windows": int(self.cumulative[-1]),
        }

    def sample(self, rng, size):
        sampled = torch.randint(self.cumulative[-1], (size,), generator=rng).numpy()
        span = np.searchsorted(self.cumulative, sampled, side="right")
        previous = self.cumulative[span] - (self.stops[span] - self.eligible[span])
        ends = self.eligible[span] + sampled - previous
        positions = ends[:, None] + self.relative
        ids = self.data[positions].astype(np.int64)
        target_positions = positions[:, 1:]
        prev_span = np.maximum(np.searchsorted(self.starts, target_positions, side="right") - 1, 0)
        supervised = ((self.starts[prev_span] <= target_positions)
                      & (target_positions < self.stops[prev_span]))
        targets = np.where(supervised, ids[:, 1:], IGNORE)
        return ids[:, :-1], targets

    def batch(self, rng, size, device):
        return tuple(torch.from_numpy(array).to(device) for array in self.sample(rng, size))


def prepare_corpora(source, directory, context):
    """Hash-split whole traces before selecting any windows and index disk-backed bytes."""
    paths = [directory / name for name in ("train.bytes", "validation.bytes")]
    spans = [[], []]
    rows = [0, 0]
    digest = hashlib.sha256()
    seen = set()
    with source.open("rb") as raw_file, ExitStack() as stack:
        streams = [stack.enter_context(path.open("wb")) for path in paths]
        for raw in raw_file:
            digest.update(raw)
            raw = raw.strip()
            if not raw:
                continue
            row_hash = hashlib.sha256(raw).digest()
            if row_hash in seen:
                continue
            seen.add(row_hash)
            split = int(int.from_bytes(row_hash[:8], "big") % 100 < 5)
            data, assistant = render_trace(json.loads(raw)["messages"])
            if not assistant or assistant[-1][1] <= context:
                continue
            offset = streams[split].tell()
            streams[split].write(data)
            spans[split].extend((offset + a, offset + b, offset + min(b, max(a, context)))
                                for a, b in assistant)
            rows[split] += 1
    corpora = [Corpus(path, index, count, context)
               for path, index, count in zip(paths, spans, rows, strict=True)]
    return corpora, digest.hexdigest()


def unseen_continuations(train, candidates, length):
    # Search via bytes' substring matcher without keeping the entire corpus in RAM.
    stride = 16 * 1024 * 1024
    remaining = set(candidates) - {None}
    for offset in range(0, len(train), stride):
        if not remaining:
            break
        block = train[offset:offset + stride + length - 1]
        remaining = {candidate for candidate in remaining if candidate not in block}
    return remaining


def validation_batches(train_corpus, heldout, args):
    """Keep held-out continuations with no verbatim occurrence in the train traces."""
    rng = torch.Generator().manual_seed(args.seed + 2)
    length = min(64, max(1, args.context // 4))
    wanted = args.batch * args.eval_batches
    inputs, targets, selected = [], [], set()
    attempts = 0
    with train_corpus.path.open("rb") as raw, mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as train:
        for _ in range(max(8, args.eval_batches * 2)):
            candidate_ids, candidate_targets = heldout.sample(rng, args.batch * 16)
            suffixes = candidate_targets[:, -length:]
            continuations = [None if (suffix == IGNORE).any() else suffix.astype(np.uint8).tobytes()
                             for suffix in suffixes]
            unseen = unseen_continuations(train, continuations, length)
            for ids, labels, continuation in zip(
                candidate_ids, candidate_targets, continuations, strict=True,
            ):
                attempts += 1
                if continuation not in unseen or continuation in selected:
                    continue
                selected.add(continuation)
                labels[:-length] = IGNORE
                inputs.append(ids)
                targets.append(labels)
                if len(inputs) == wanted:
                    break
            if len(inputs) == wanted:
                break
    if len(inputs) != wanted:
        raise ValueError(f"Found {len(inputs)} of {wanted} distinct, unseen validation continuations")
    tensors = [torch.from_numpy(np.stack(rows)).to(args.device) for rows in (inputs, targets)]
    batches = list(zip(*(tensor.split(args.batch) for tensor in tensors), strict=True))
    return batches, {"windows": wanted, "continuation_bytes": length, "candidates": attempts}


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    total_loss = total_bytes = 0
    for ids, targets in batches:
        logits, _, _ = model(ids)
        total_loss += F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="sum", ignore_index=IGNORE,
        ).item()
        total_bytes += (targets != IGNORE).sum().item()
    return total_loss / total_bytes / math.log(2), total_bytes


@contextmanager
def atomic_write(path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    yield temporary
    temporary.replace(path)


def train(args):
    if min(args.width, args.layers, args.context, args.batch, args.steps,
           args.eval_every, args.eval_batches) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Training dimensions, schedule and learning rate must be positive")
    if (args.output / "latest.pt").exists() and not args.resume:
        raise ValueError("Output already contains training state. Use --resume or another --output")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    print("Indexing Fable conversations for training and validation", flush=True)
    with TemporaryDirectory(prefix="corpus-", dir=args.output) as directory:
        corpora, source_hash = prepare_corpora(args.data, Path(directory), args.context)
        try:
            optimize(args, *corpora, source_hash)
        finally:
            # Windows requires the disk-backed views to be closed before removing their files.
            for corpus in corpora:
                del corpus.data


def optimize(args, train_corpus, val_corpus, source_hash):
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 1)
    architecture = {
        "kind": "shrf_gated", "width": args.width, "layers": args.layers, "context": args.context,
    }
    model = TinyDecoder(**architecture).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print("Selecting held-out continuations absent verbatim from the training traces", flush=True)
    val_batches, val_protocol = validation_batches(train_corpus, val_corpus, args)
    metadata = {
        "source": args.source, "sha256": source_hash, "architecture": architecture,
        "seed": args.seed, "batch": args.batch, "lr": args.lr, "eval_batches": args.eval_batches,
        "train": train_corpus.metadata, "validation": val_corpus.metadata,
        "validation_protocol": val_protocol,
        "parameters": sum(p.numel() for p in model.parameters()),
        "objective": "assistant text next UTF-8 byte", "format_version": 2,
    }
    checkpoint = args.output / "latest.pt"
    completed, best, curve = 0, float("inf"), []
    seen_bytes = 0
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("Resume requires the same data, model and sampling configuration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        completed, best, curve = saved["step"], saved["best_val_bpb"], saved["curve"]
        seen_bytes = saved["seen_bytes"]
        if args.steps <= completed:
            raise ValueError(f"--steps must exceed the checkpoint step ({completed})")
    print(json.dumps(metadata), flush=True)
    started = time.perf_counter()
    for step in range(completed + 1, args.steps + 1):
        model.train()
        ids, targets = train_corpus.batch(rng, args.batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        logits, states, aux = model(ids)
        del states
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        seen_bytes += (targets != IGNORE).sum().item()
        if step != completed + 1 and step % args.eval_every and step != args.steps:
            continue
        validation, val_bytes = evaluate(model, val_batches)
        if not math.isfinite(validation):
            raise FloatingPointError(f"Nonfinite validation loss at step {step}")
        row = {
            "step": step, "train_bpb": loss.item() / math.log(2), "val_bpb": validation,
            "validation_targets": val_bytes, "seen_assistant_bytes": seen_bytes,
            "seconds_this_run": time.perf_counter() - started,
        }
        curve.append(row)
        improved = validation < best
        best = min(best, validation)
        saved = {
            "metadata": metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": rng.get_state(), "step": step, "best_val_bpb": best, "curve": curve,
            "seen_bytes": seen_bytes,
        }
        with atomic_write(checkpoint) as path:
            torch.save(saved, path)
        if improved:
            weights = {"metadata": metadata, "model": saved["model"], "step": step, "val_bpb": best}
            with atomic_write(args.output / "best.pt") as path:
                torch.save(weights, path)
        report = {"metadata": metadata, "steps_requested": args.steps, "curve": curve}
        with atomic_write(args.output / "metrics.json") as path:
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


@torch.inference_mode()
def generate(args):
    if arithmetic := getattr(args, "arithmetic", None):
        answer = structured_answer(DecimalArithmetic.load(arithmetic), args.prompt)
        if answer is not None:
            print(answer)
            return
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    arch = saved["metadata"]["architecture"]
    model = TinyDecoder(**arch).to(args.device).eval()
    model.load_state_dict(saved["model"])
    prompt = b"<user>\n" + args.prompt.encode("utf-8") + b"\n</user>\n" + ASSISTANT_START
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    ids = torch.tensor([list(prompt[-arch["context"]:])], device=args.device)
    logits, states, _ = model(ids)
    output = bytearray()
    for i in range(args.max_bytes):
        probabilities = (logits[:, -1] / args.temperature).softmax(-1)
        ids = torch.multinomial(probabilities, 1, generator=rng)
        output.append(ids.item())
        if output.endswith(ASSISTANT_END):
            del output[-len(ASSISTANT_END):]
            break
        if i + 1 < args.max_bytes:
            logits, states, _ = model(ids, states)
    sys.stdout.buffer.write(output.decode("utf-8", errors="replace").encode("utf-8") + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--source", default="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--width", type=int, default=96)
    fit.add_argument("--layers", type=int, default=4)
    fit.add_argument("--context", type=int, default=256)
    fit.add_argument("--batch", type=int, default=4)
    fit.add_argument("--steps", type=int, default=300)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--eval-every", type=int, default=25)
    fit.add_argument("--eval-batches", type=int, default=12)
    sample = commands.add_parser("generate")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--arithmetic", type=Path, help="Optional trained decimal arithmetic checkpoint")
    sample.add_argument("--prompt", required=True)
    sample.add_argument("--max-bytes", type=int, default=128)
    sample.add_argument("--temperature", type=float, default=0.8)
    for command in (fit, sample):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        command.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        train(args)
    else:
        if args.max_bytes < 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
            parser.error("Generation length and temperature must be positive")
        generate(args)


if __name__ == "__main__":
    main()
