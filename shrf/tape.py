"""Optional exact cold archive, indexed by batch, tree, leaf and token position."""
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch


@dataclass
class Recovery:
    time: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    weight: torch.Tensor


class TapeConflictError(sqlite3.IntegrityError):
    """An archived position was reused for a different observation."""


class IDTape:
    def __init__(self, path: str | Path = ":memory:", *, wal=False, checkpoint_mode="snapshot"):
        if checkpoint_mode not in ("snapshot", "reference"):
            raise ValueError("checkpoint_mode must be 'snapshot' or 'reference'")
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).resolve())
        self.wal = bool(wal)
        self.checkpoint_mode = checkpoint_mode
        self.db = sqlite3.connect(self.path)
        self._configure()
        self._create_tables()
        with self._transaction():
            self.db.execute("INSERT OR IGNORE INTO tape_identity VALUES(1,?)", (uuid4().hex,))
            self.identity = self._identity()

    def _configure(self):
        self.db.execute("PRAGMA foreign_keys = ON")
        if self.wal and self.path != ":memory:":
            mode = self.db.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if mode != "wal":
                self.db.close()
                raise RuntimeError("SQLite could not enable WAL for this archive")
            self.db.execute("PRAGMA synchronous = NORMAL")

    def _create_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS obs(
              batch INTEGER, t INTEGER, key BLOB, value BLOB, gate REAL,
              PRIMARY KEY(batch,t)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS route(
              batch INTEGER, tree INTEGER, leaf INTEGER, t INTEGER, mass REAL,
              PRIMARY KEY(batch,tree,leaf,t),
              FOREIGN KEY(batch,t) REFERENCES obs(batch,t)) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS route_position ON route(batch,t,tree,leaf);
            CREATE TABLE IF NOT EXISTS tape_identity(
              singleton INTEGER PRIMARY KEY CHECK(singleton = 1), token TEXT NOT NULL);
        """)

    def _identity(self):
        return self.db.execute("SELECT token FROM tape_identity WHERE singleton=1").fetchone()[0]

    def __getstate__(self):
        if self.checkpoint_mode == "reference" and self.path != ":memory:":
            return self.path, self.wal, self.identity, None
        with self._transaction():
            snapshot = "\n".join(self.db.iterdump())
        return ":memory:", False, self.identity, snapshot

    def __setstate__(self, state):
        self.path, self.wal, self.identity, snapshot = state
        self.checkpoint_mode = "snapshot" if snapshot is not None else "reference"
        if snapshot is not None:
            self.db = sqlite3.connect(":memory:")
            self.db.executescript(snapshot)
            self._configure()
        else:
            uri = Path(self.path).as_uri() + "?mode=rw"
            self.db = sqlite3.connect(uri, uri=True)
            self._configure()
            try:
                self.db.execute("SELECT batch,t FROM obs LIMIT 0")
                self.db.execute("SELECT batch,tree,leaf,t FROM route LIMIT 0")
                if self._identity() != self.identity:
                    raise TapeConflictError("Checkpoint points to a different SQLite archive")
            except (sqlite3.Error, TypeError):
                self.db.close()
                raise

    @contextmanager
    def _transaction(self):
        # An outer transaction, if any, stays owned by the caller.
        self.db.execute("SAVEPOINT shrf_tape")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK TO shrf_tape")
            self.db.execute("RELEASE shrf_tape")
            raise
        else:
            self.db.execute("RELEASE shrf_tape")

    def record(self, starts, keys, values, gate, leaves, mass):
        starts, keys, values, gate, leaves, mass = (
            x.detach().cpu().numpy() for x in (starts, keys, values, gate, leaves, mass))
        if not all(np.isfinite(array).all() for array in (keys, values, gate, mass)):
            raise ValueError("Archive observations must be finite")
        if not ((gate >= 0) & (gate <= 1)).all() or not ((mass >= 0) & (mass <= 1)).all():
            raise ValueError("Archive gate and route mass must be in [0, 1]")
        B, C, E, _ = keys.shape
        payload, routes = [], []
        skipped = {}
        # np.save headers are unnecessary: encode dimensionality and dtype with each blob.
        for b in range(B):
            for s in range(C):
                t = int(starts[b]) + s
                g = float(gate[b, s])
                if g <= 0:
                    skipped.setdefault(b, set()).add(t)
                    continue
                payload.append((b, t, self._encode(keys[b, s]), self._encode(values[b, s]), g))
                for e in range(E):
                    for leaf, w in zip(leaves[b, s, e], mass[b, s, e], strict=False):
                        if w > 0:
                            routes.append((b, e, int(leaf), t, float(w)))
        try:
            with self._transaction():
                self.db.executemany("INSERT INTO obs VALUES(?,?,?,?,?)", payload)
                self.db.executemany("INSERT INTO route VALUES(?,?,?,?,?)", routes)
                self._check_skipped(skipped)
        except sqlite3.IntegrityError:
            self._record_retry(payload, routes, skipped)

    def _check_skipped(self, skipped):
        for batch, times in skipped.items():
            for (t,) in self.db.execute("SELECT t FROM obs WHERE batch=? AND t>=? AND t<=?",
                                        (batch, min(times), max(times))):
                if t in times:
                    raise TapeConflictError(f"Replay omitted archived data at batch={batch}, t={t}")

    def _record_retry(self, payload, routes, skipped):
        by_position = {}
        for batch, tree, leaf, t, mass in routes:
            by_position.setdefault((batch, t), []).append((tree, leaf, mass))
        with self._transaction():
            for batch, t, key, value, gate in payload:
                inserted = self.db.execute(
                    "INSERT INTO obs VALUES(?,?,?,?,?) ON CONFLICT(batch,t) DO NOTHING",
                    (batch, t, key, value, gate),
                ).rowcount
                expected_routes = sorted(by_position.get((batch, t), ()))
                if inserted:
                    self.db.executemany("INSERT INTO route VALUES(?,?,?,?,?)",
                                        ((batch, e, leaf, t, w) for e, leaf, w in expected_routes))
                    continue
                existing = self.db.execute("SELECT key,value,gate FROM obs WHERE batch=? AND t=?",
                                           (batch, t)).fetchone()
                existing_routes = self.db.execute(
                    "SELECT tree,leaf,mass FROM route WHERE batch=? AND t=? ORDER BY tree,leaf",
                    (batch, t),
                ).fetchall()
                if existing != (key, value, gate) or existing_routes != expected_routes:
                    raise TapeConflictError(f"Archive already contains different data at batch={batch}, t={t}")
            self._check_skipped(skipped)

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
        time, keys, values, weights = zip(*rows, strict=False)
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
