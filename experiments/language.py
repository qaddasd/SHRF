"""Token-matched SHRF/GQA comparison on held-out Python documentation topics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from pydoc_data.topics import topics

import torch
from torch import nn
from torch.nn import functional as F

from shrf import SHRFConfig, SHRFLayer
from shrf.integration import CausalLocalMixer, RMSNorm, SwiGLU, memory_output_gate


class TopicCorpus:
    def __init__(self, context, split):
        # Keep topics whole. Group identical topic excerpts before splitting.
        unique = {}
        for name, doc in sorted(topics.items()):
            payload = doc.encode("utf-8")
            unique.setdefault(hashlib.sha256(payload).hexdigest(), (name, payload))
        eligible = sorted((digest, name, raw) for digest, (name, raw) in unique.items()
                          if len(raw) > context)
        if context < 1 or len(eligible) < 2:
            raise ValueError("Context must be positive and fit at least two documentation topics")
        n_val = max(1, len(eligible)//5)
        selected = eligible[:n_val] if split == "validation" else eligible[n_val:]
        offsets, counts, data = [], [], bytearray()
        for _, _, raw in selected:
            offsets.append(len(data))
            counts.append(len(raw)-context)
            data.extend(raw)
        self.context = context
        self.offsets = torch.tensor(offsets)
        self.ends = torch.tensor(counts).cumsum(0)
        self.total = int(self.ends[-1])
        self.data = torch.tensor(list(data), dtype=torch.uint8)
        self.metadata = {"topics": [name for _, name, _ in selected], "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()}

    def batch(self, rng, size, device):
        sample = torch.randint(self.total, (size,), generator=rng)
        topic = torch.searchsorted(self.ends, sample, right=True)
        prior = F.pad(self.ends, (1, 0))[topic]
        start = self.offsets[topic] + sample-prior
        ids = self.data[start[:, None] + torch.arange(self.context+1)].to(device, dtype=torch.long)
        return ids[:, :-1], ids[:, 1:]


class GQAMixer(nn.Module):
    """Four query heads, two KV heads by default, with an explicit KV cache."""
    def __init__(self, d, heads=4, kv_heads=2):
        super().__init__()
        self.dh = d//heads
        if d % heads or heads % kv_heads or self.dh % 2:
            raise ValueError("Grouped heads and rotary pairs must divide the model shape")
        self.heads, self.kv_heads = heads, kv_heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kv_heads*self.dh, bias=False)
        self.v = nn.Linear(d, kv_heads*self.dh, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.register_buffer("frequency", 10_000.0**(-torch.arange(0, self.dh, 2)/self.dh))

    def _heads(self, x, heads):
        return x.unflatten(-1, (heads, self.dh)).transpose(1, 2)

    def _rotary(self, x, cos, sin):
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a*cos-b*sin, a*sin+b*cos), -1).flatten(-2)

    def forward(self, x, state=None):
        prefix = 0 if state is None else state[0].shape[-2]
        positions = torch.arange(prefix, prefix+x.shape[1], device=x.device)
        phase = positions[:, None] * self.frequency
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        q = self._rotary(self._heads(self.q(x), self.heads), cos, sin)
        k = self._rotary(self._heads(self.k(x), self.kv_heads), cos, sin)
        v = self._heads(self.v(x), self.kv_heads)
        if state is not None:
            k = torch.cat((state[0], k), dim=-2)
            v = torch.cat((state[1], v), dim=-2)
        # Explicit repeat is portable across PyTorch CPU/CUDA SDPA backends.
        groups = self.heads//self.kv_heads
        mask = (torch.arange(k.shape[-2], device=x.device) <= positions[:, None]
                if prefix and x.shape[1] > 1 else None)
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(groups, dim=1),
                                          v.repeat_interleave(groups, dim=1),
                                          attn_mask=mask, is_causal=not prefix)
        return self.o(y.transpose(1, 2).flatten(-2)), (k, v)


class TinyDecoder(nn.Module):
    def __init__(self, kind, width, layers, context, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embed.weight, std=.08)
        self.before = nn.ModuleList(RMSNorm(width) for _ in range(layers))
        self.after = nn.ModuleList(RMSNorm(width) for _ in range(layers))
        self.mlp = nn.ModuleList(SwiGLU(width, 2*width) for _ in range(layers))
        self.final = RMSNorm(width)
        local_names = ("shrf_local", "shrf_gated", "local_only")
        local = (nn.ModuleList(CausalLocalMixer(width) for _ in range(layers))
                 if kind in local_names else None)
        if kind == "gqa":
            self.mix = nn.ModuleList(GQAMixer(width) for _ in range(layers))
        elif kind == "local_only":
            self.mix = local
        else:
            cfg = SHRFConfig(d_model=width, n_trees=2, depth=3, r_key=8, d_value=24,
                             sketch_rank=6, beam=2, chunk_size=context,
                             context_lags=2 if kind != "shrf" else 0)
            if kind == "hybrid":
                if layers < 2:
                    raise ValueError("The hybrid baseline needs at least one SHRF and one GQA block")
                self.mix = nn.ModuleList(
                    GQAMixer(width) if (i+1) % 4 == 0 or i+1 == layers else SHRFLayer(cfg)
                    for i in range(layers)
                )
            else:
                self.mix = nn.ModuleList(SHRFLayer(cfg) for _ in range(layers))
        self.local = local if kind in ("shrf_local", "shrf_gated") else None
        self.gates = (nn.ModuleList(memory_output_gate(width, -3.) for _ in range(layers))
                      if kind == "shrf_gated" else None)

    def forward(self, ids, states=None, *, write_weight=None):
        x = self.embed(ids)
        aux, new = x.new_zeros(()), []
        for i, mix in enumerate(self.mix):
            state = None if states is None else states[i]
            norm = self.before[i](x)
            if self.local is None or state is None:
                memory, history = state, None
            else:
                memory, history = state
            if isinstance(mix, SHRFLayer):
                h, state = mix(norm, memory, write_weight=write_weight)
                aux = aux + mix.last_aux
            else:
                h, state = mix(norm, memory)
            if self.gates is not None:
                h = h * self.gates[i](norm).sigmoid()
            if self.local is not None:
                local, history = self.local[i](norm, history)
                h, state = h+local, (state, history)
            x = x+h
            x = x+self.mlp[i](self.after[i](x))
            new.append(state)
        return F.linear(self.final(x), self.embed.weight), new, aux


@torch.no_grad()
def validate(model, corpus, args):
    model.eval()
    rng = torch.Generator().manual_seed(17)
    losses = []
    for _ in range(args.eval_batches):
        ids, target = corpus.batch(rng, args.batch, args.device)
        y, _, _ = model(ids)
        losses.append(F.cross_entropy(y.flatten(0,1), target.flatten()).item())
    return sum(losses)/len(losses)


def run(kind, train, validation, args):
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 1)
    model = TinyDecoder(kind, args.width, args.layers, args.context).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    curve = []
    total_seconds = 0.0
    synchronize()
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        ids, target = train.batch(rng, args.batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        y, _, aux = model(ids)
        loss = F.cross_entropy(y.flatten(0,1), target.flatten())
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            synchronize()
            total_seconds += time.perf_counter()-start
            nll = validate(model, validation, args)
            curve.append({"step": step, "train_nll": loss.item(), "val_bpb": nll/math.log(2)})
            print(kind, curve[-1], flush=True)
            model.train()
            synchronize()
            start = time.perf_counter()
    return {"parameters": sum(p.numel() for p in model.parameters()),
                "total_seconds": total_seconds, "tokens": args.steps*args.batch*args.context,
                "tokens_per_second": args.steps*args.batch*args.context/total_seconds,
                "val_bpb": curve[-1]["val_bpb"], "curve": curve,
                "validation_targets": args.eval_batches*args.batch*args.context}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--context", type=int, default=96)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--eval-every", type=int, default=60)
    p.add_argument("--eval-batches", type=int, default=12)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--output", type=Path, default=Path("results/language.json"))
    names = ("gqa", "shrf", "shrf_context", "shrf_local", "shrf_gated", "local_only")
    p.add_argument("--models", nargs="+", choices=(*names, "hybrid"), default=list(names))
    args = p.parse_args()
    torch.set_num_threads(2)
    train = TopicCorpus(args.context, "train")
    val = TopicCorpus(args.context, "validation")
    assert not set(train.metadata["topics"]).intersection(val.metadata["topics"])
    report = {"config": {k: v for k, v in vars(args).items() if k != "output"},
                  "source": "Python standard library pydoc_data.topics", "runtime": sys.version,
                  "torch": torch.__version__, "train": train.metadata, "validation": val.metadata, "runs": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for kind in args.models:
        report["runs"][kind] = run(kind, train, val, args)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
