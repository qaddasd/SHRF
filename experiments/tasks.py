"""Vectorized benchmarks with payloads sampled independently of query IDs."""
import torch


def choices(batch, pool, count, device):
    if count > pool:
        raise ValueError("The task requests more distinct keys than the vocabulary provides")
    return torch.rand(batch, pool, device=device).argsort(-1)[:, :count]


def _bindings(keys, payload, query_keys, idx, vocab, meta):
    B, items = keys.shape
    ids = torch.cat((torch.stack((keys, payload), -1).reshape(B, -1),
                     torch.stack((query_keys, torch.zeros_like(query_keys)), -1).reshape(B, -1)), -1)
    targets = torch.zeros_like(ids)
    targets[:, 2*items::2] = payload.gather(1, idx)
    mask = targets != 0
    weight = torch.zeros_like(ids, dtype=torch.float32)
    weight[:, 1:2*items:2] = 1
    return ids, targets, mask, vocab, dict(meta, write_weight=weight)


def mqar(B, n_pairs=6, n_queries=4, n_keys=24, n_values=16, device="cpu"):
    """Episode-specific random bindings, multiple queries, and unique addresses."""
    keys = choices(B, n_keys, n_pairs, device) + 1
    values = torch.randint(n_values, (B, n_pairs), device=device) + n_keys + 1
    idx = torch.randint(n_pairs, (B, n_queries), device=device)
    return _bindings(keys, values, keys.gather(1, idx), idx, 1+n_keys+n_values,
                     {"chance": 1/n_values})


def semantic_recall(B, n_items=6, n_queries=4, n_facts=16, n_para=5,
                    n_values=16, held_out=False, device="cpu"):
    """TRAIN reads aliases 1..n_para-2. Held-out evaluation reads only the last alias."""
    if n_para < 3:
        raise ValueError("Need a canonical key, at least one training alias and a held-out alias")
    keys = 1 + choices(B, n_facts, n_items, device)*n_para
    payload = torch.randint(n_values, (B, n_items), device=device) + 1+n_facts*n_para
    idx = torch.randint(n_items, (B, n_queries), device=device)
    alias = n_para-1 if held_out else torch.randint(1, n_para-1, idx.shape, device=device)
    meta = {"chance": 1/n_values, "clusters": n_facts, "per_cluster": n_para}
    return _bindings(keys, payload, keys.gather(1, idx)+alias, idx, 1+n_facts*n_para+n_values, meta)


def dense_clusters(B, n_items=8, n_queries=4, n_keys=24, n_values=16,
                   spread=0.12, device="cpu"):
    keys = choices(B, n_keys, n_items, device) + 1
    payload = torch.randint(n_values, (B, n_items), device=device) + 1+n_keys
    idx = torch.randint(n_items, (B, n_queries), device=device)
    meta = {"chance": 1/n_values, "clusters": 1, "per_cluster": n_keys, "spread": spread}
    return _bindings(keys, payload, keys.gather(1, idx), idx, 1+n_keys+n_values, meta)


def copy_span(B, n_spans=4, width=3, n_tokens=32, device="cpu"):
    """Prefix contains four random spans. Repeat a cue and predict its continuation."""
    cues = choices(B, 2*n_spans, n_spans, device) + 1
    payload = choices(B, n_tokens, n_spans*width, device).reshape(B, n_spans, width) + 1+2*n_spans
    episode = torch.cat((cues.unsqueeze(-1), payload), dim=-1)
    row, idx = torch.arange(B, device=device), torch.randint(n_spans, (B,), device=device)
    suffix = episode[row, idx]
    prefix = episode.flatten(1)
    ids = torch.cat((prefix, suffix[:, :-1]), dim=1)
    targets = torch.zeros_like(ids)
    targets[:, -width:] = suffix[:, 1:]
    weight = torch.ones_like(ids, dtype=torch.float32)
    weight[:, :prefix.shape[1]:width+1] = 0
    weight[:, prefix.shape[1]:] = 0
    return ids, targets, targets != 0, 1+2*n_spans+n_tokens, {
        "chance": 1/n_tokens, "write_weight": weight, "prefix": prefix.shape[1]}


def niah(B, ctx_len=96, n_needles=3, n_keys=16, n_values=16, n_filler=32,
         needle_fraction=None, decoy_fraction=0., device="cpu"):
    """Distinct key-value pairs embedded in distractors. Query reveals only the key."""
    vocab = 1+n_keys+n_values+n_filler
    if ctx_len < 2*n_needles:
        raise ValueError("Context too short for the requested needles")
    keys = choices(B, n_keys, n_needles, device) + 1
    vals = torch.randint(n_values, (B, n_needles), device=device) + 1+n_keys
    ids = torch.randint(n_filler, (B, ctx_len+1), device=device) + 1+n_keys+n_values
    if decoy_fraction:
        width = ctx_len//2
        mask = torch.rand(B, width, device=device) < decoy_fraction
        decoy = torch.randint(n_keys-1, (B, width), device=device) + 1
        decoy += (decoy >= keys[:, :1]).long()
        payloads = torch.randint(n_values, (B, width), device=device) + 1+n_keys
        ids[:, :2*width:2] = torch.where(mask, decoy, ids[:, :2*width:2])
        ids[:, 1:2*width:2] = torch.where(mask, payloads, ids[:, 1:2*width:2])
    slots = choices(B, ctx_len//2, n_needles, device)
    if needle_fraction is not None:
        fixed = min(max(int(needle_fraction*(ctx_len//2)), 0), ctx_len//2-1)
        # Force the selected needle depth while moving a conflicting distractor into the vacated slot.
        slots = torch.where(slots == fixed, slots[:, :1], slots)
        slots[:, 0] = fixed
    pos = 2 * slots
    ids.scatter_(1, pos, keys)
    ids.scatter_(1, pos+1, vals)
    ids[:, -1] = keys[:, 0]
    targets = torch.zeros_like(ids)
    targets[:, -1] = vals[:, 0]
    weight = torch.zeros_like(ids, dtype=torch.float32).scatter_(1, pos+1, 1)
    return ids, targets, targets != 0, vocab, {"chance": 1/n_values, "write_weight": weight}


@torch.no_grad()
def init_cluster_embeddings(model, meta, spread=0.2):
    """Use a fixed synthetic embedding geometry; held-out aliases never get gradients."""
    groups, per_group = meta["clusters"], meta["per_cluster"]
    weight = model.embed.weight
    root = torch.randn(groups, 1, weight.shape[-1], device=weight.device)
    offsets = torch.randn(groups, per_group, weight.shape[-1], device=weight.device)
    weight[1:1+groups*per_group] = (root + meta.get("spread", spread)*offsets).flatten(0, 1)
    weight.requires_grad_(False)
