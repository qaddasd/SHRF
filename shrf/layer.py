"""Decoder memory with the same soft causal update in scan and decode modes."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import SHRFConfig
from .routing import RouteLevel, Routing
from .scans import causal_prefix, coalesce_rows
from .sketch import ResidualSketch
from .stats import StatLayout, contribution, ridge, shrinkage_lambda
from .tape import IDTape


@dataclass
class SHRFState:
    heap: Tensor
    extra: Tensor
    leaf_slots: Tensor
    previous_key: Tensor
    count: Tensor
    route_version: int
    tape: IDTape | None = None
    key_history: Tensor | None = None
    heap_time: Tensor | None = None
    extra_time: Tensor | None = None
    decay: float = 1.0
    key_origin: Tensor | None = None
    value_origin: Tensor | None = None
    feature_origin: Tensor | None = None


class SHRFLayer(nn.Module):
    def __init__(self, cfg: SHRFConfig):
        super().__init__()
        self.cfg = cfg
        self.lay = StatLayout(cfg.r_key, cfg.d_value)
        self.W_k = nn.Parameter(torch.randn(cfg.n_trees, cfg.r_key, cfg.d_model) / cfg.d_model**0.5)
        self.context_mix = (nn.Parameter(torch.zeros(cfg.n_trees, cfg.context_lags))
                            if cfg.context_lags else None)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_value, bias=False)
        self.W_o = nn.Linear(cfg.d_value, cfg.d_model, bias=False)
        self.write_gate = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.write_gate.weight)
        initial_gate = (1 / (1 + math.exp(-1))) ** (1 / cfg.gate_power)
        nn.init.constant_(self.write_gate.bias, math.log(initial_gate / (1-initial_gate)))
        self.routing = Routing(cfg.n_trees, cfg.depth, cfg.r_key, cfg.tau,
                               cfg.dir_mix, cfg.beam, cfg.seed)
        self.sketch = (ResidualSketch(cfg.n_trees, cfg.r_key, cfg.d_value, cfg.sketch_width,
                                     cfg.sketch_bandwidth, cfg.seed + 1)
                       if cfg.sketch_width else None)
        self._route_version = 0
        self.last_aux = 0.0

    def init_state(self, B, device=None, dtype=None, tape=None):
        c = self.cfg
        device = self.W_k.device if device is None else device
        dtype = (torch.float64 if self.W_k.dtype == torch.float64 else torch.float32) if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("SHRF accumulators require float32 or float64")
        heap = torch.zeros(B, c.n_trees, c.n_nodes, self.lay.dim, device=device, dtype=dtype)
        extra = heap.new_empty((B, c.n_trees, 0, 0))
        slots = torch.empty(0, device=device, dtype=torch.long)
        if self.sketch is not None:
            extra = heap.new_zeros((B, c.n_trees, c.n_leaves + 1, self.sketch.dim))
            slots = torch.arange(1, c.n_leaves + 1, device=device).expand(B, c.n_trees, -1).clone()
        history = (heap.new_zeros((B, c.context_lags, c.n_trees, c.r_key))
                   if c.context_lags else None)
        count = torch.zeros(B, device=device, dtype=torch.long)
        clocks = tuple(count.new_zeros(tensor.shape[:-1]) for tensor in (heap, extra)) if c.decay < 1 else (None, None)
        return SHRFState(heap, extra, slots, heap.new_zeros((B, c.n_trees, c.r_key)),
                         count, self._route_version, tape, history, *clocks, c.decay)

    @staticmethod
    def _group(nodes, size):
        b, _, e, _ = nodes.shape
        offset = torch.arange(b*e, device=nodes.device).reshape(b, 1, e, 1)
        return offset * size + nodes

    def _discount(self, ages, ref):
        return (ages.to(torch.float64) * math.log(self.cfg.decay)).exp().to(ref.dtype)

    def _read_pass(self, heap, read_group, write_group, rows, clock, time, scale, *, root=False):
        flat = heap.flatten(0, 2)
        prefix = flat[read_group]
        if clock is not None:
            ages = time - clock.reshape(-1)[read_group]
            prefix = prefix * self._discount(ages, prefix).unsqueeze(-1)
        if read_group.shape[1] == 1:
            return prefix
        if root:
            delta = torch.cat((torch.zeros_like(rows[:, :1]), rows[:, :-1]), 1)
            delta = delta.cumsum(1, dtype=torch.float64).to(rows.dtype)
        else:
            delta = causal_prefix(read_group, write_group, rows)
        return prefix + (delta if scale is None else delta * scale)

    @staticmethod
    def _commit(heap, groups, rows, backpropagate, coalesce=False):
        flat = heap.flatten(0, 2)
        if backpropagate:
            index = torch.cat([group.reshape(-1) for group in groups])
            values = torch.cat([row.flatten(0, -2) for row in rows])
            return flat.index_add(0, index, values).reshape_as(heap)
        with torch.no_grad():
            if coalesce:
                index = torch.cat([group.reshape(-1) for group in groups])
                values = torch.cat([row.flatten(0, -2) for row in rows])
                index, values = coalesce_rows(index, values)
                flat.index_add_(0, index, values)
            else:
                for group, row in zip(groups, rows, strict=False):
                    flat.index_add_(0, group.reshape(-1), row.flatten(0, -2))
        return heap

    def _commit_decayed(self, heap, clock, groups, rows, end, scale, keep_graph):
        index = torch.cat([group.reshape(-1) for group in groups])
        values = torch.cat([row.flatten(0, -2) for row in rows])
        flat, times = heap.flatten(0, 2), clock.flatten()
        with torch.set_grad_enabled(keep_graph):
            index, values = coalesce_rows(index, values)
            end_time = end[index // (heap.shape[1] * heap.shape[2])]
            ages = end_time - times[index]
            values = flat[index] * self._discount(ages, flat).unsqueeze(-1) + values * scale
            if keep_graph:
                flat = flat.index_copy(0, index, values)
            else:
                flat.index_copy_(0, index, values)
        times.index_copy_(0, index, end_time)
        return flat.reshape_as(heap)

    def _contextual_keys(self, raw, history, keep_graph):
        if self.context_mix is None:
            return raw.tanh(), history
        lags = self.cfg.context_lags
        padded = torch.cat((history, raw), dim=1)
        contextual = raw
        for lag in range(1, lags+1):
            weight = self.context_mix[:, lag-1].to(raw.dtype)[None, None, :, None]
            contextual = contextual + padded[:, lags-lag:lags-lag+raw.shape[1]] * weight
        history = padded[:, -lags:]
        return contextual.tanh(), history if keep_graph else history.detach().clone()

    def _node_balance(self, route, probability):
        group = self._group(route.node, self.cfg.n_nodes).reshape(-1)
        mass = route.mass.detach()
        _, totals = coalesce_rows(group, torch.stack((mass, mass * probability), -1).reshape(-1, 2))
        visits, right = totals.unbind(-1)
        midpoint_error = right / visits.clamp_min(1e-8) - 0.5
        return (visits * midpoint_error.square()).sum() / visits.sum().clamp_min(1e-8)

    def _chunk(self, k, v, gate, state, balance):
        c = self.cfg
        kw = k
        trajectory = k
        if c.write_lag:
            trajectory = torch.cat((state.previous_key.unsqueeze(1), k), dim=1)
            kw = trajectory[:, :-1]
            first_valid = (state.count > 0).to(gate.dtype)
            gate = torch.cat((gate[:, :1] * first_valid.unsqueeze(-1), gate[:, 1:]), dim=1)

        route = self.routing(trajectory)
        if c.write_lag:
            read, write = [], []
            for level in route:
                parent = None if level.parent is None else level.parent[:, 1:]
                probability = None if level.right_probability is None else level.right_probability[:, 1:]
                read.append(RouteLevel(level.node[:, 1:], level.mass[:, 1:], parent, probability))
                write.append(RouteLevel(level.node[:, :-1], level.mass[:, :-1], None))
        else:
            read = write = route
        key_origin, value_origin, feature_origin = state.key_origin, state.value_origin, state.feature_origin
        if key_origin is None:
            fresh = state.count == 0
            key_origin = torch.where(fresh[:, None, None], k[:, 0].detach(), 0.)
            value_origin = torch.where(fresh[:, None], v[:, 0].detach(), 0.)
            if self.sketch is not None:
                feature_origin = torch.where(fresh[:, None, None], self.sketch.features(key_origin), 0.)
        query = k - key_origin[:, None]
        write_key = kw - key_origin[:, None]
        values = (v - value_origin[:, None]).unsqueeze(-2).expand(-1, -1, c.n_trees, -1)
        raw = contribution(write_key, values, self.lay)
        use_sketch = self.sketch is not None and state.extra.shape[2] > 1
        if use_sketch:
            phi = self.sketch.features(trajectory) - feature_origin[:, None]
            phi_r, phi_w = (phi[:, 1:], phi[:, :-1]) if c.write_lag else (phi, phi)
            raw_extra = self.sketch.contribution(write_key, phi_w, values)
        elapsed = k.shape[1]
        clock = state.heap_time
        time = prefix_scale = end_scale = None
        scaled_gate = gate
        if clock is not None:
            offsets = torch.arange(elapsed, device=k.device)
            time = (state.count[:, None] + offsets)[:, :, None, None]
            scaled_gate = gate * self._discount(-offsets, gate)
            prefix_scale = self._discount((offsets-1).clamp_min(0), k)[None, :, None, None, None]
            end_scale = self._discount(offsets[-1], k)
        groups, rows, prefixes = [], [], []
        for depth, (rd, wr) in enumerate(zip(read, write, strict=False)):
            rg, wg = self._group(rd.node, c.n_nodes), self._group(wr.node, c.n_nodes)
            weighted = raw.unsqueeze(-2) * (scaled_gate[..., None, None] * wr.mass).unsqueeze(-1)
            groups.append(wg)
            rows.append(weighted)
            stats = self._read_pass(state.heap, rg, wg, weighted, clock, time, prefix_scale, root=depth == 0)
            prefixes.append(stats)

        # Every affine fit is independent of the hierarchical read. Factorize
        # all the path nodes in one batched operation, then apply the recursion.
        widths = [rd.node.shape[-1] for rd in read]
        stats = torch.cat(prefixes, dim=3)
        keys = query.unsqueeze(-2).expand(*stats.shape[:-1], c.r_key)
        fits = ridge(stats, keys, self.lay, c.ridge)
        predictions = list(fits["pred"].split(widths, dim=3))
        means = fits["mu_v"].split(widths, dim=3)
        variances = fits["sigma2"].split(widths, dim=3)
        counts = fits["n"].split(widths, dim=3)
        write_slots = extra_rows = None
        if use_sketch:
            rd, wr = read[-1], write[-1]
            leaves_r, leaves_w = rd.node - c.n_leaves, wr.node - c.n_leaves
            offset = self._group(torch.zeros_like(leaves_r), c.n_leaves)
            flat_slots = state.leaf_slots.reshape(-1)
            rs, ws = flat_slots[offset + leaves_r], flat_slots[offset + leaves_w]
            read_slots = self._group(rs, state.extra.shape[2])
            write_slots = self._group(ws, state.extra.shape[2])
            mass = (scaled_gate[..., None, None] * wr.mass) * (ws != 0)
            extra_rows = raw_extra.unsqueeze(-2) * mass.unsqueeze(-1)
            extra = self._read_pass(state.extra, read_slots, write_slots, extra_rows,
                                    state.extra_time, time, prefix_scale)
            phi = phi_r.unsqueeze(-2).expand(*rd.node.shape, self.sketch.m)
            leaf_key = query.unsqueeze(-2).expand(*rd.node.shape, c.r_key)
            nonlinear = self.sketch.joint(prefixes[-1], extra, leaf_key, phi, self.lay,
                                          c.ridge, c.sketch_ridge)["pred"]
            predictions[-1] = torch.where((rs != 0).unsqueeze(-1), nonlinear, predictions[-1])

        previous_pred = predictions[0]
        result = previous_pred.squeeze(-2)
        aux = []
        for depth, rd in enumerate(read[1:], start=1):
            pred = predictions[depth]
            parent_idx = rd.parent.unsqueeze(-1).expand_as(pred)
            parent_pred = previous_pred.gather(-2, parent_idx)
            parent_mean = means[depth-1].gather(-2, parent_idx)
            parent_var = variances[depth-1].gather(-1, rd.parent)
            between = (means[depth] - parent_mean).square().mean(-1) + 0.001*parent_var + 1e-5
            n, sigma2 = counts[depth], variances[depth]
            lam = (shrinkage_lambda(n, sigma2, between, c.kappa_min, c.kappa_max)
                   if c.use_shrinkage else (n > 1e-8).to(k.dtype))
            correction = lam.unsqueeze(-1) * (pred - parent_pred)
            previous_pred = parent_pred + correction
            result = result + (rd.mass.unsqueeze(-1) * correction).sum(-2)
            if balance:
                aux.append(self._node_balance(read[depth-1], rd.right_probability))

        if state.tape is not None:
            state.tape.record(state.count, kw, v, gate, write[-1].node - c.n_leaves, write[-1].mass)

        keep_graph = torch.is_grad_enabled() and not c.detach_chunks
        coalesce = c.coalesce_writes and k.shape[1] > 1
        end = state.count + elapsed
        if clock is None:
            state.heap = self._commit(state.heap, groups, rows, keep_graph, coalesce)
            if use_sketch:
                state.extra = self._commit(state.extra, [write_slots], [extra_rows], keep_graph, coalesce)
        else:
            state.heap = self._commit_decayed(state.heap, clock, groups, rows, end, end_scale, keep_graph)
            if use_sketch:
                state.extra = self._commit_decayed(state.extra, state.extra_time, [write_slots],
                                                   [extra_rows], end, end_scale, keep_graph)
        state.previous_key = k[:, -1] if keep_graph else k[:, -1].detach().clone()
        state.count = end
        state.key_origin, state.value_origin, state.feature_origin = key_origin, value_origin, feature_origin
        result = result + (counts[0] > 1e-8) * value_origin[:, None, None]
        return result.mean(-2), torch.stack(aux).mean() if aux else k.new_zeros(())

    def forward(self, x, state=None, return_aux=False, *, write_weight=None):
        if x.ndim != 3 or x.shape[-1] != self.cfg.d_model:
            raise ValueError("Expected x with shape [batch, time, d_model]")
        B, T, _ = x.shape
        if state is None:
            state = self.init_state(B, device=x.device)
        if state.route_version != self._route_version:
            raise ValueError("Routing changed. Initialize an empty state or replay the archive.")
        if state.decay != self.cfg.decay:
            raise ValueError("State decay differs from the layer configuration")
        if state.heap.shape[0] != B or state.heap.device != x.device:
            raise ValueError("State batch or device does not match the input")
        if write_weight is not None and write_weight.shape != (B, T):
            raise ValueError("Expected write_weight with shape [batch, time]")
        if not T or not B:
            self.last_aux = x.new_zeros(())
            return (x, state, self.last_aux) if return_aux else (x, state)

        c = self.cfg
        if not bool(torch.isfinite(x).all()):
            raise ValueError("SHRF inputs must be finite")
        # Project in model precision. Keep float32/64 for moments and solves.
        raw_keys = torch.einsum("btd,erd->bter", x, self.W_k).to(state.heap.dtype)
        values = self.W_v(x).to(state.heap.dtype)
        logits = self.write_gate(x).squeeze(-1).to(state.heap.dtype)
        # Reserve headroom for value-energy sums over any int64 token count.
        value_cap = math.sqrt(torch.finfo(values.dtype).max / (c.d_value * 2**66))
        safe_projections = ((raw_keys.abs() <= value_cap).all() & torch.isfinite(logits).all()
                            & (values.abs() <= value_cap).all())
        if self.context_mix is not None:
            safe_projections = safe_projections & (self.context_mix.abs() <= value_cap).all()
        if not bool(safe_projections):
            raise ValueError("SHRF projections are non-finite or exceed the moment accumulation range")
        gate = logits.sigmoid()
        if c.gate_power != 1:
            gate = gate.pow(c.gate_power)
        if write_weight is not None:
            weight = write_weight.to(device=x.device, dtype=gate.dtype)
            if not bool(((weight >= 0) & (weight <= 1)).all()):
                raise ValueError("write_weight must be finite and in [0, 1]")
            gate = gate * weight
        output, aux_total = [], raw_keys.new_zeros(())
        keep_graph = torch.is_grad_enabled() and not c.detach_chunks
        balance = c.load_balance_weight != 0 and (return_aux or torch.is_grad_enabled())
        chunk = c.chunk_size
        if c.decay < 1:
            # Keep chunk-local inverse powers bounded even for very short half-lives.
            chunk = min(chunk, max(1, int(16 / -math.log(c.decay))))
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                keys, history = self._contextual_keys(raw_keys[:, start:end], state.key_history, keep_graph)
                y, aux = self._chunk(keys, values[:, start:end], gate[:, start:end], state, balance)
                state.key_history = history
                output.append(y)
                aux_total = aux_total + (end-start) * aux
        self.last_aux = c.load_balance_weight * aux_total / T
        y = self.W_o(torch.cat(output, 1).to(self.W_o.weight.dtype))
        return (y, state, self.last_aux) if return_aux else (y, state)

    @torch.no_grad()
    def step(self, x_t, state, *, write_weight=None):
        weight = None if write_weight is None else write_weight.unsqueeze(1)
        y, state = self(x_t.unsqueeze(1), state, write_weight=weight)
        return y[:, 0], state

    @torch.no_grad()
    def refine_directions(self, keys, values, min_support=8):
        self.routing.calibrate(keys, values, self.cfg.ridge, min_support)
        self._route_version += 1

    @torch.no_grad()
    def consolidate(self, state, budget=None):
        from .consolidate import consolidate
        return consolidate(self, state, budget)

    @torch.no_grad()
    def recover_prediction(self, tape, query, tree, leaf, t_start, t_end, *, batch=0, limit=None):
        """Explicit cold-path ridge refit on archived projected observations."""
        rows = tape.recover(tree, leaf, t_start, t_end, batch=batch, limit=limit)
        if rows is None:
            return None
        c = self.cfg
        if query.shape != (c.r_key,):
            raise ValueError("Recovery query must be a single projected key")
        key, val, weight = rows.key.double(), rows.value.double(), rows.weight.double().unsqueeze(-1)
        query = query.to(key)
        key_origin, value_origin = key[0], val[0]
        base = (contribution(key - key_origin, val - value_origin, self.lay) * weight).sum(0)
        if self.sketch is None:
            pred = ridge(base, query - key_origin, self.lay, c.ridge)["pred"]
        else:
            phi = self.sketch.tree_features(tree, key)
            phi_query = self.sketch.tree_features(tree, query)
            extra = (self.sketch.contribution(key - key_origin, phi - phi[0],
                                              val - value_origin) * weight).sum(0)
            pred = self.sketch.joint(base, extra, query - key_origin, phi_query - phi[0],
                                    self.lay, c.ridge, c.sketch_ridge)["pred"]
        return (pred + value_origin).to(rows.value.dtype) if base[0] > 1e-8 else pred.to(rows.value.dtype)
