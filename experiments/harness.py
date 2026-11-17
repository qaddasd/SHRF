"""Synthetic training and blind/query-only/recurrent evaluation."""
import time

import torch
from torch.nn import functional as F

from experiments.model import SHRFModel
from experiments.tasks import init_cluster_embeddings


def masked_loss_acc(logits, targets, mask):
    scores, labels = logits[mask], targets[mask]
    return F.cross_entropy(scores, labels), (scores.argmax(-1) == labels).float().mean().item()


def train_eval(task, kwargs, cfg, *, steps, batch, device, seed, val_kwargs=None,
               oracle=False, eval_batches=4, eval_batch=32, return_model=False,
               model_factory=SHRFModel):
    torch.manual_seed(seed)
    _, _, _, vocab, meta = task(2, device=device, **kwargs)
    model = model_factory(vocab, cfg.d_model, 1, cfg).to(device)
    if "clusters" in meta:
        init_cluster_embeddings(model, meta)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    sync()
    start = time.perf_counter()
    curve = []
    for i in range(steps):
        ids, tgt, score, _, info = task(batch, device=device, **kwargs)
        logits, _, aux = model(ids, write_weight=info["write_weight"] if oracle else None)
        loss, acc = masked_loss_acc(logits, tgt, score)
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        if i == 0 or (i+1) % max(1, steps//4) == 0 or i == steps-1:
            curve.append({"step": i+1, "acc": acc, "loss": loss.item()})
    sync()
    result = {"seed": seed, "steps": steps, "batch": batch, "seconds": round(time.perf_counter()-start, 3),
                  "chance": meta["chance"], "oracle_writes": oracle, "curve": curve}
    val = kwargs if val_kwargs is None else val_kwargs
    result["eval"] = evaluate(model, task, val, eval_batch, eval_batches, seed+1234, oracle=oracle)
    result["query_only"] = evaluate(model, task, val, eval_batch, eval_batches, seed+1234, no_memory=True)
    return (result, model) if return_model else result


@torch.no_grad()
def evaluate(model, task, kwargs, batch=32, batches=4, seed=1234, *,
             no_memory=False, oracle=False, noise=0.0, collapse_rank=None, rollout=False):
    model.eval()
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    projection = None
    if collapse_rank:
        p = torch.randn(model.cfg.d_model, collapse_rank, device=device)
        p = torch.linalg.qr(p, mode="reduced").Q
        projection = (p @ p.T) * (model.cfg.d_model / collapse_rank)**0.5
    hook = None
    if noise or collapse_rank is not None:
        def corrupt(_, inp, x):
            if collapse_rank == 0:
                x = torch.zeros_like(x)
            elif projection is not None:
                x = x @ projection
            return x + noise*x.std()*torch.randn_like(x) if noise else x
        hook = model.embed.register_forward_hook(corrupt)
    hits, scores, losses, exact, total = 0, 0, 0., 0, 0
    try:
        for _ in range(batches):
            ids, tgt, mask, _, meta = task(batch, device=device, **kwargs)
            w = torch.zeros_like(ids) if no_memory else meta["write_weight"] if oracle else None
            if rollout:
                start = meta["prefix"]
                _, states, _ = model(ids[:, :start], write_weight=None if w is None else w[:, :start])
                cue = ids[:, start:start+1]
                predictions = []
                for t in range(start, ids.shape[1]):
                    logit, states, _ = model(cue, states, write_weight=None if w is None else w[:, t:t+1])
                    cue = logit.argmax(-1)
                    predictions.append(cue)
                correct = torch.cat(predictions, dim=1) == tgt[:, start:]
                hits += int(correct.sum())
                scores += correct.numel()
                exact += int(correct.all(dim=1).sum())
                total += correct.shape[0]
            else:
                logits, _, _ = model(ids, write_weight=w)
                correct = logits[mask].argmax(-1) == tgt[mask]
                hits += int(correct.sum())
                scores += correct.numel()
                losses += F.cross_entropy(logits[mask], tgt[mask], reduction="sum").item()
    finally:
        if hook is not None:
            hook.remove()
    result = {"accuracy": round(hits/scores, 5), "targets": scores}
    if rollout:
        result["exact_span"] = round(exact/total, 5)
    else:
        result["loss"] = round(losses/scores, 5)
    return result
