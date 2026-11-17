"""Budgeted sketch maintenance with explicit cold-path admission."""
import torch

from .stats import ridge


def _restore_sketch(layer, state, batch, tree, leaf, end):
    sketch = layer.sketch
    rows = state.tape.recover(tree, leaf, 0, end, batch=batch)
    if rows is None:
        raise ValueError("The tape is missing observations for a leaf selected for admission")
    key = rows.key.to(state.heap)
    value = rows.value.to(state.heap)
    weight = rows.weight.to(state.heap)
    phi = sketch.tree_features(tree, key)
    if state.key_origin is not None:
        key = key - state.key_origin[batch, tree]
        value = value - state.value_origin[batch]
        phi = phi - state.feature_origin[batch, tree]
    if state.extra_time is not None:
        age = end - 1 - rows.time.to(state.count.device)
        weight = weight * layer._discount(age, weight)
    return (sketch.contribution(key, phi, value) * weight[:, None]).sum(0, dtype=torch.float64).to(state.heap)


def consolidate(layer, state, budget):
    cfg = layer.cfg
    if state.route_version != layer._route_version:
        raise ValueError("Routing changed. Reinitialize the state before consolidation.")
    budget = cfg.sketch_budget if budget is None else budget
    if budget is None:
        raise ValueError("Provide a per-tree sketch slot budget")
    if not isinstance(budget, int) or not 0 <= budget <= cfg.n_leaves:
        raise ValueError("Invalid per-tree sketch slot budget")
    if layer.sketch is None or state.heap.shape[0] == 0:
        return state.leaf_slots
    slots = state.leaf_slots
    if state.tape is None:
        if budget >= state.extra.shape[2] - 1:
            return slots
        raise ValueError("Attach an IDTape at state initialization to preserve evicted observations")

    base = state.heap[:, :, cfg.n_leaves:, :]
    query = base.new_zeros((*base.shape[:-1], cfg.r_key))
    if state.heap_time is not None:
        age = state.count[:, None, None] - state.heap_time[:, :, cfg.n_leaves:]
        base = base * layer._discount(age, base).unsqueeze(-1)
    # Affine residual mass is available with or without a resident sketch.
    score = ridge(base, query, layer.lay, cfg.ridge)["rss"]
    eligible = base[..., 0] > 1e-8
    rank = score.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    rank = rank[..., :budget]
    active = eligible.gather(2, rank)
    capacity = int(active.sum(-1).max().item())
    packed = state.extra.new_zeros((*slots.shape[:2], capacity + 1, layer.sketch.dim))
    new_slots = torch.zeros_like(slots)
    times = None if state.extra_time is None else state.extra_time.new_zeros(packed.shape[:-1])
    batches = rank.cpu().tolist()
    n_active = active.sum(-1).cpu().tolist()
    old_slots = slots.cpu().tolist()
    for batch, trees in enumerate(batches):
        end = int(state.count[batch].item())
        for tree, leaves in enumerate(trees):
            for new, leaf in enumerate(leaves[:n_active[batch][tree]], 1):
                old = old_slots[batch][tree][leaf]
                if old:
                    packed[batch, tree, new] = state.extra[batch, tree, old]
                    if times is not None:
                        times[batch, tree, new] = state.extra_time[batch, tree, old]
                else:
                    packed[batch, tree, new] = _restore_sketch(layer, state, batch, tree, leaf, end)
                    if times is not None:
                        times[batch, tree, new] = end
                new_slots[batch, tree, leaf] = new
    state.extra = packed
    state.extra_time = times
    state.leaf_slots = new_slots
    return new_slots
