"""Rate-distortion eviction of leaf sketches, with the base heap preserved."""
import torch
from .stats import ridge


def consolidate(layer, state, budget):
    cfg = layer.cfg
    if state.route_version != layer._route_version:
        raise ValueError("Routing changed. Reinitialize the state before consolidation.")
    budget = cfg.sketch_budget if budget is None else budget
    if budget is None:
        raise ValueError("Provide a per-tree sketch slot budget")
    if not isinstance(budget, int) or not 0 <= budget <= cfg.n_leaves:
        raise ValueError("Invalid per-tree sketch slot budget")
    if layer.sketch is None:
        return state.leaf_slots
    slots = state.leaf_slots
    current = state.extra.shape[2] - 1
    if budget >= current:
        return slots
    if state.tape is None:
        raise ValueError("Attach an IDTape at state initialization to preserve evicted observations")
    sketch = layer.sketch
    base = state.heap[:, :, cfg.n_leaves:, :]
    extra = state.extra.gather(2, slots.unsqueeze(-1).expand(*slots.shape, sketch.dim))
    k = base.new_zeros((*base.shape[:-1], cfg.r_key))
    p = base.new_zeros((*base.shape[:-1], sketch.m))
    affine = ridge(base, k, layer.lay, cfg.ridge)
    joint = sketch.joint(base, extra, k, p, layer.lay, cfg.ridge, cfg.sketch_ridge)
    gain = (joint["gain"] - affine["gain"]).clamp_min(0).masked_fill(slots == 0, -torch.inf)
    keep = gain.argsort(dim=-1, descending=True, stable=True)[..., :budget]
    old = slots.gather(2, keep)
    new = torch.arange(1, budget+1, device=slots.device).expand_as(keep)
    new_slots = torch.zeros_like(slots).scatter_(2, keep, new)
    selected = state.extra.gather(2, old.unsqueeze(-1).expand(*old.shape, sketch.dim))
    state.extra = torch.cat((torch.zeros_like(state.extra[:, :, :1]), selected), dim=2)
    state.leaf_slots = new_slots
    return new_slots
