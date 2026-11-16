"""Differentiable segmented prefixes with query events before same-time writes."""
import torch


def sort_by_group_time(group, time):
    order = time.argsort(stable=True)
    return order[group[order].argsort(stable=True)]


def exclusive_segmented_cumsum(values, group_sorted):
    if values.shape[0] == 0:
        return values.clone()
    positions = torch.arange(values.shape[0], device=values.device)
    first = torch.cat((torch.ones(1, dtype=torch.bool, device=values.device),
                       group_sorted[1:] != group_sorted[:-1]))
    starts = torch.where(first, positions, 0).cummax(0).values
    accum = values.to(torch.float64).cumsum(0)
    exclusive = torch.cat((torch.zeros_like(accum[:1]), accum[:-1]), dim=0)
    return (exclusive - exclusive[starts]).to(values.dtype)


def causal_prefix(read_group, write_group, write_rows):
    """Groups [B,C,E,K]; writes [B,C,E,K,S], independent read/write beam widths."""
    C = read_group.shape[1]
    time = torch.arange(C, device=read_group.device).view(1, C, 1, 1)
    read_event = (read_group*(2*C) + 2*time).reshape(-1)
    write_event = (write_group*(2*C) + 2*time + 1).reshape(-1)
    event = torch.cat((read_event, write_event))
    order = event.argsort(stable=True)
    rows = torch.cat((write_rows.new_zeros((read_event.numel(), write_rows.shape[-1])),
                      write_rows.flatten(0, -2)), dim=0)
    prefix = exclusive_segmented_cumsum(rows[order], event[order] // (2*C))
    out = torch.empty_like(prefix).index_copy(0, order, prefix)
    return out[:read_event.numel()].reshape(*read_group.shape, write_rows.shape[-1])


def coalesce_rows(group, values):
    """Return one FP64-accumulated update per distinct node."""
    order = group.argsort(stable=True)
    nodes, lengths = group[order].unique_consecutive(return_counts=True)
    sums = torch.segment_reduce(values[order].to(torch.float64), "sum", lengths=lengths)
    return nodes, sums.to(values.dtype)
