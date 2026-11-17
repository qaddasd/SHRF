"""Train and run a bilingual SHRF decoder with explicit solution and answer spans."""

import argparse
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from shrf.arithmetic import DecimalArithmetic, structured_answer

from .code_checks import check
from .curriculum import greedy as byte_greedy
from .curriculum import numeric_equal, prompt_key, save_json
from .curriculum_data import gsm_cases, openassistant_examples
from .fable import ASSISTANT_START, IGNORE, atomic_write
from .language import TinyDecoder
from .reasoning_data import (
    ReasoningCase,
    arithmetic_cases,
    tensorov_cases,
    verified_cases,
)
from .reasoning_tokenizer import ASSISTANT, END, FINAL, USER, WORK, ReasoningTokenizer

DOMAINS = ("en", "ru", "math", "code", "logic")
MATH_SCHEDULE = ("arithmetic", "math", "arithmetic", "en", "arithmetic", "math", "ru",
                 "arithmetic", "math", "code", "arithmetic", "logic")


def encode(case, tokenizer, context, *, allow_prefix=False):
    prefix = tokenizer.prompt(case.prompt)
    work = [WORK, *tokenizer.encode(case.work)] if case.work else []
    answer = tokenizer.encode(case.answer)
    available = context+1-len(prefix)-len(work)-1
    if not answer or available <= 0:
        return None
    ending = [END]
    if len(answer)+1 > available:
        if not allow_prefix or available < 24:
            return None
        answer, ending = answer[:available], []
    tokens = [*prefix, *work, FINAL, *answer, *ending]
    labels = [IGNORE]*(len(prefix)-1) + tokens[len(prefix):]
    answer_weight = 3. if case.domain in {"math", "logic"} and case.source.startswith("verified") else 1.
    weights = [0.]*(len(prefix)-1) + [.6]*len(work) + [1.] + [answer_weight]*len(answer) + [.2]*len(ending)
    return tokens[:-1], labels, weights


def make_batch(cases, device):
    length = max(len(ids) for ids, _, _ in cases)
    ids, targets, weights, valid = [], [], [], []
    for x, y, w in cases:
        pad = length-len(x)
        ids.append(x + [0]*pad)
        targets.append(y + [IGNORE]*pad)
        weights.append(w + [0.]*pad)
        valid.append([1.]*len(x) + [0.]*pad)
    return tuple(torch.tensor(x, device=device) for x in (ids, targets, weights, valid))


def dialogue_cases(path):
    for row in openassistant_examples(path):
        if len(row.prompt) <= 340 and len(row.answer) <= 720:
            yield ReasoningCase(row.domain.removeprefix("writing_"), row.prompt, row.answer,
                                source=row.source)


def unique_sample(rows, count, predicate=lambda _: True):
    seen = set()
    result = []
    for row in rows:
        key = prompt_key(row.prompt)
        if key in seen or not predicate(row):
            continue
        seen.add(key)
        result.append(row)
        if len(result) == count:
            break
    return result


def prepare(args, parent_context, tokenizer=None):
    rng = random.Random(args.seed)
    tensorov = tensorov_cases(args.tensorov, args.seed)
    dialogues = list(dialogue_cases(args.oasst))
    rng.shuffle(dialogues)
    reference = list(verified_cases(args.seed+701, 180))
    gsm = [ReasoningCase("math", row.prompt, row.answer, source="GSM8K holdout")
           for row in gsm_cases(args.gsm_test, heldout=True)]
    reserved = {prompt_key(row.prompt) for row in (*reference, *gsm)}
    math_focus = args.gsm_train is not None
    arithmetic_train = list(arithmetic_cases(args.seed)) if math_focus else []
    arithmetic_test = list(arithmetic_cases(args.seed, heldout=True)) if math_focus else []
    rng_for_arithmetic = random.Random(args.seed+83)
    rng_for_arithmetic.shuffle(arithmetic_test)
    reserved.update(prompt_key(row.prompt) for row in arithmetic_test)
    heldout = {}
    for language in ("en", "ru"):
        heldout[language] = unique_sample(
            (row for row in dialogues if row.domain == language), 20,
            lambda row: len(row.prompt.encode()) + len(row.answer.encode()) + 33 <= parent_context,
        )
        if len(heldout[language]) < 12:
            raise ValueError(f"Need more short {language} dialogue evaluations")
        reserved.update(prompt_key(row.prompt) for row in heldout[language])

    math_word_problems = []
    if math_focus:
        math_word_problems = [ReasoningCase("math", row.prompt, row.answer, source="GSM8K train")
                              for row in gsm_cases(args.gsm_train)
                              if not row.prompt.endswith(" Explain briefly.")]
    data = [*tensorov, *dialogues, *verified_cases(args.seed, 3500), *math_word_problems]
    rng.shuffle(data)
    data = [row for row in unique_sample(data, len(data))
            if prompt_key(row.prompt) not in reserved
            and prompt_key(row.prompt+" Reply with only the number.") not in reserved]
    if tokenizer is None:
        tokenizer = ReasoningTokenizer.fit(
            (text for row in (*data, *arithmetic_train) for text in (row.prompt, row.work, row.answer[:900])),
            args.vocab_size,
        )
    pools = {domain: defaultdict(list) for domain in DOMAINS}
    if math_focus:
        pools["arithmetic"] = defaultdict(list)
    counts = Counter()
    domains_by_source = defaultdict(Counter)
    partial_tensorov = 0
    for row in (*data, *arithmetic_train):
        is_tensorov = row.source.startswith("tensorov/")
        encoded = encode(row, tokenizer, args.context, allow_prefix=is_tensorov)
        if encoded is None:
            continue
        domain = "arithmetic" if row.source == "verified-arithmetic" else row.domain
        kind = "tensorov" if is_tensorov else "other"
        if row.source == "GSM8K train":
            kind = "gsm8k"
        pools[domain][kind].append(encoded)
        counts[row.source] += 1
        domains_by_source[row.source][row.domain] += 1
        partial_tensorov += int(is_tensorov and encoded[1][-1] != END)
    if not any(pools[domain].get("tensorov") for domain in DOMAINS):
        raise ValueError("No complete Tensorov examples fit this training context")
    if any(not pools[domain] for domain in DOMAINS):
        raise ValueError("Every requested domain needs training examples")
    exam = {}
    for domain in ("math", "logic", "code"):
        exam[domain] = unique_sample(
            (replace(row, work="") for row in reference if row.domain == domain), 18,
            lambda row: encode(row, tokenizer, args.context) is not None,
        )
    exam["gsm8k"] = unique_sample(gsm, 24, lambda row: encode(row, tokenizer, args.context) is not None
                                 and len(row.prompt.encode()) + 24 <= parent_context)
    if math_focus:
        arithmetic_by_op = {
            op: [row for row in arithmetic_test if f" {op} " in row.prompt]
            for op in ("+", "-", "*", "/")
        }
        exam["arithmetic"] = [row for rows in arithmetic_by_op.values() for row in rows[:8]]
    composition = {k: dict(v) for k, v in domains_by_source.items()}
    return tokenizer, pools, heldout, exam, dict(counts), composition, partial_tensorov


def warm_start(parent, tokenizer, context):
    architecture = {**parent["metadata"]["architecture"], "context": context,
                    "vocab_size": len(tokenizer.bytes)}
    model = TinyDecoder(**architecture)
    old_state = parent["model"]
    if "words" in parent:
        if tuple(parent["words"]) != tokenizer.words:
            raise ValueError("Continuing a reasoning checkpoint requires its original vocabulary")
        model.load_state_dict(old_state)
        return model, architecture
    byte_embedding = old_state["embed.weight"]
    if byte_embedding.shape[0] != 256:
        raise ValueError("This initialization requires a 256-byte parent checkpoint")
    embedding = torch.stack([
        byte_embedding[list(token)].sum(dim=0)/math.sqrt(len(token))
        for token in tokenizer.bytes
    ])
    model.load_state_dict({**old_state, "embed.weight": embedding})
    return model, architecture


@torch.inference_mode()
def decode(model, tokenizer, prompt, device, context, limit=96, *, show_work=False):
    prefix = torch.tensor([tokenizer.prompt(prompt)[-context:]], device=device)
    logits, state, _ = model(prefix)
    tokens = []
    for _ in range(limit):
        logits[:, -1, [USER, ASSISTANT]] = -torch.inf
        token = logits[:, -1].argmax(-1, keepdim=True)
        value = int(token.item())
        if value == END:
            break
        tokens.append(value)
        logits, state, _ = model(token, state)
    if not show_work and FINAL in tokens:
        tokens = tokens[tokens.index(FINAL)+1:]
    if not show_work:
        tokens = [token for token in tokens if token not in (WORK, FINAL)]
    return tokenizer.decode(tokens).strip()


@torch.inference_mode()
def text_scores(model, heldout, device, tokenizer=None):
    model.eval()
    scores = {}
    for language, cases in heldout.items():
        loss, nbytes, ntokens = 0., 0, 0
        for case in cases:
            prefix = (tokenizer.prompt(case.prompt)+[FINAL] if tokenizer
                      else list(b"<user>\n" + case.prompt.encode() + b"\n</user>\n" + ASSISTANT_START))
            answer = tokenizer.encode(case.answer) if tokenizer else list(case.answer.encode())
            sequence = torch.tensor([prefix + answer[:-1]], device=device)
            logits, _, _ = model(sequence)
            desired = torch.tensor(answer, device=device)
            loss += F.cross_entropy(logits[0, len(prefix)-1:], desired, reduction="sum").item()
            nbytes += len(case.answer.encode())
            ntokens += len(answer)
        scores[language] = {"bpb": loss/nbytes/math.log(2), "answer_bytes": nbytes,
                            "answer_tokens": ntokens, "examples": len(cases)}
    return scores


def passed(case, response):
    if case.domain == "math":
        return numeric_equal(response, case.answer)
    if case.domain == "logic":
        return response.strip().strip(".").casefold() == case.answer.casefold()
    a, b = map(int, re.search(r"return (\d+) \* x - (\d+)", case.answer).groups())
    return check(response, ((-4,), (0,), (7,)), lambda x: a*x-b)[1]


def exam_scores(model, exam, device, context, tokenizer=None):
    model.eval()
    result = {}
    for domain, cases in exam.items():
        correct = 0
        by_language = defaultdict(Counter)
        for case in cases:
            limit = {"code": 64, "arithmetic": 12}.get(domain, 40)
            if tokenizer:
                response = decode(model, tokenizer, case.prompt, device, context, limit)
            else:
                response = byte_greedy(model, case.prompt, device, context, limit,
                                       numeric=case.domain == "math")
            is_correct = passed(case, response)
            correct += is_correct
            language = "ru" if re.search("[А-Яа-я]", case.prompt) else "en"
            by_language[language]["correct"] += is_correct
            by_language[language]["total"] += 1
        result[domain] = {"correct": correct, "total": len(cases),
                          "languages": {k: dict(v) for k, v in by_language.items()}}
    return result


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed+1)
    parent = torch.load(args.parent, map_location="cpu", weights_only=True)
    parent_context = parent["metadata"]["architecture"]["context"]
    parent_tokenizer = ReasoningTokenizer(parent["words"]) if "words" in parent else None
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"model.pt").exists():
        raise ValueError("Choose a new output directory for the next training run")
    print("Preparing Tensorov, bilingual dialogues, and verified tasks", flush=True)
    byte_limit = 384 if parent_tokenizer else parent_context
    tokenizer, pools, heldout, exam, counts, composition, partial_tensorov = prepare(
        args, byte_limit, parent_tokenizer,
    )
    model, architecture = warm_start(parent, tokenizer, args.context)
    model = model.to(args.device)
    embed = model.embed.weight
    optimizer = torch.optim.AdamW([
        {"params": [embed], "lr": args.lr*2},
        {"params": [p for p in model.parameters() if p is not embed]},
    ], lr=args.lr, weight_decay=.01)
    metadata = {
        "architecture": architecture, "parent_step": parent["step"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "examples": counts, "composition": composition,
        "tensorov_file": args.tensorov.name, "tensorov_bytes": args.tensorov.stat().st_size,
        "tensorov_answer_prefixes": partial_tensorov,
        "seed": args.seed, "batch": args.batch, "vocabulary": len(tokenizer.bytes),
        "math_focus": args.gsm_train is not None,
    }
    report = {"metadata": metadata, "curve": [], "training_samples": {}}
    print(metadata, flush=True)
    parent_model = TinyDecoder(**parent["metadata"]["architecture"])
    parent_model.load_state_dict(parent["model"])
    parent_model.to(args.device)
    report["before_text"] = text_scores(parent_model, heldout, args.device, parent_tokenizer)
    save_json(args.output/"metrics.json", report)
    started = time.perf_counter()
    samples = Counter()
    schedule = MATH_SCHEDULE if args.gsm_train is not None else DOMAINS
    for step in range(1, args.steps+1):
        model.train()
        domain = schedule[(step-1) % len(schedule)]
        domain_pool = pools[domain]
        collected = []
        for _ in range(args.batch):
            if "gsm8k" in domain_pool and (len(domain_pool) == 1 or rng.random() < .35):
                kind = "gsm8k"
            else:
                kind = ("tensorov" if "tensorov" in domain_pool and
                        ("other" not in domain_pool or rng.random() < .48) else "other")
            collected.append(rng.choice(domain_pool[kind]))
            samples[f"{domain}/{kind}"] += 1
        ids, targets, weights, valid = make_batch(collected, args.device)
        optimizer.zero_grad(set_to_none=True)
        logits, states, aux = model(ids, write_weight=valid)
        del states
        nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                              reduction="none", ignore_index=IGNORE)
        loss = (nll*weights.flatten()).sum()/weights.sum()
        (loss+aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step != 1 and step % args.eval_every and step != args.steps:
            continue
        scores = text_scores(model, heldout, args.device, tokenizer)
        result = {"step": step, "loss": loss.item(), "text": scores,
                  "seconds": time.perf_counter()-started}
        report["curve"].append(result)
        report["training_samples"] = dict(samples)
        save_json(args.output/"metrics.json", report)
        with atomic_write(args.output/"model.pt") as path:
            torch.save({"model": model.state_dict(), "metadata": metadata, "words": list(tokenizer.words),
                        "step": step, "format": "shrf-reasoning-1"}, path)
        print(result, flush=True)
    report["after_text"] = report["curve"][-1]["text"]
    report["before_exam"] = exam_scores(parent_model, exam, args.device, parent_context, parent_tokenizer)
    report["after_exam"] = exam_scores(model, exam, args.device, args.context, tokenizer)
    save_json(args.output/"metrics.json", report)
    print({"before_exam": report["before_exam"], "after_exam": report["after_exam"]}, flush=True)


def generate(args):
    if args.arithmetic is not None and not args.show_work:
        answer = structured_answer(DecimalArithmetic.load(args.arithmetic), args.prompt)
        if answer is not None:
            print(answer)
            return
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ReasoningTokenizer(saved["words"])
    model = TinyDecoder(**saved["metadata"]["architecture"]).to(args.device)
    model.load_state_dict(saved["model"])
    model.eval()
    result = decode(model, tokenizer, args.prompt, args.device,
                    saved["metadata"]["architecture"]["context"], args.max_tokens,
                    show_work=args.show_work)
    sys.stdout.buffer.write((result+"\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    training = commands.add_parser("train")
    for name in ("parent", "tensorov", "oasst", "gsm-test", "output"):
        training.add_argument("--"+name, type=Path, required=True)
    training.add_argument("--gsm-train", type=Path, help="Enable mathematics-focused training with GSM8K train")
    training.add_argument("--steps", type=int, default=360)
    training.add_argument("--batch", type=int, default=6)
    training.add_argument("--context", type=int, default=144)
    training.add_argument("--vocab-size", type=int, default=2048)
    training.add_argument("--eval-every", type=int, default=90)
    training.add_argument("--lr", type=float, default=.00065)
    training.add_argument("--seed", type=int, default=62)
    generation = commands.add_parser("generate")
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--arithmetic", type=Path)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--show-work", action="store_true")
    generation.add_argument("--max-tokens", type=int, default=96)
    for command in (training, generation):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if command is training else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "train":
        if min(args.steps, args.batch, args.context, args.eval_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            parser.error("Training dimensions and learning rate must be positive")
        train(args)
    else:
        if args.max_tokens < 1:
            parser.error("Generation length must be positive")
        generate(args)


if __name__ == "__main__":
    main()
