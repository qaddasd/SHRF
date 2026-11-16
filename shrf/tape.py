"""Optional exact cold archive, indexed by batch, tree, leaf and token position."""
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
import torch


@dataclass
class Recovery:
    time: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    weight: torch.Tensor


class IDTape:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS obs(
              batch INTEGER, t INTEGER, key BLOB, value BLOB, gate REAL,
              PRIMARY KEY(batch,t)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS route(
              batch INTEGER, tree INTEGER, leaf INTEGER, t INTEGER, mass REAL,
              PRIMARY KEY(batch,tree,leaf,t),
              FOREIGN KEY(batch,t) REFERENCES obs(batch,t)) WITHOUT ROWID;
        """)

    def record(self, starts, keys, values, gate, leaves, mass):
        starts, keys, values, gate, leaves, mass = (
            x.detach().cpu().numpy() for x in (starts, keys, values, gate, leaves, mass))
        B, C, E, _ = keys.shape
        payload, routes = [], []
        # np.save headers are unnecessary: encode dimensionality and dtype with each blob.
        for b in range(B):
            for s in range(C):
                t = int(starts[b]) + s
                g = float(gate[b, s])
                if g <= 0:
                    continue
                payload.append((b, t, self._encode(keys[b, s]), self._encode(values[b, s]), g))
                for e in range(E):
                    for l, w in zip(leaves[b, s, e], mass[b, s, e]):
                        if w > 0:
                            routes.append((b, e, int(l), t, float(w)))
        with self.db:
            self.db.executemany("INSERT INTO obs VALUES(?,?,?,?,?)", payload)
            self.db.executemany("INSERT INTO route VALUES(?,?,?,?,?)", routes)

    @staticmethod
    def _encode(array):
        shape = np.asarray(array.shape, dtype="<u4")
        width = array.dtype.itemsize
        return bytes((width, len(shape))) + shape.tobytes() + array.astype(f"<f{width}", copy=False).tobytes()

    @staticmethod
    def _decode(blob):
        width, ndim = blob[:2]
        shape = np.frombuffer(blob, dtype="<u4", count=ndim, offset=2)
        return np.frombuffer(blob, dtype=f"<f{width}", offset=2+4*ndim).reshape(tuple(shape))

    def recover(self, tree, leaf, t_start, t_end, *, batch=0, limit=None):
        """Recover projected key/value/weight for a half-open interval, ordered by time."""
        if t_end < t_start or (limit is not None and limit < 1):
            raise ValueError("Invalid half-open interval or recovery limit")
        rows = self.db.execute("""
            SELECT r.t, o.key, o.value, o.gate*r.mass FROM route AS r
            JOIN obs AS o ON r.batch=o.batch AND r.t=o.t
            WHERE r.batch=? AND r.tree=? AND r.leaf=? AND r.t>=? AND r.t<?
            ORDER BY r.t LIMIT ?
        """, (batch, tree, leaf, t_start, t_end, -1 if limit is None else limit)).fetchall()
        return self._recovery(rows, tree)

    def recover_time(self, t_start, t_end, *, batch=0, limit=None):
        """Fetch full projected keys for all trees and the original gate."""
        if t_end < t_start or (limit is not None and limit < 1):
            raise ValueError("Invalid half-open interval or recovery limit")
        rows = self.db.execute("""
            SELECT t,key,value,gate FROM obs WHERE batch=? AND t>=? AND t<? ORDER BY t LIMIT ?
        """, (batch, t_start, t_end, -1 if limit is None else limit)).fetchall()
        return self._recovery(rows)

    def _recovery(self, rows, tree=None):
        if not rows:
            return None
        time, keys, values, weights = zip(*rows)
        key = np.stack([self._decode(k) for k in keys])
        value = np.stack([self._decode(v) for v in values])
        return Recovery(torch.tensor(time), torch.from_numpy(key if tree is None else key[:, tree]),
                        torch.from_numpy(value), torch.tensor(weights, dtype=torch.from_numpy(key).dtype))

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
