# Inference and archive hardening

## Checkpoints and retries

`torch.save(state, path)` can serialize a state with an `IDTape`.
Load a checkpoint you created with `torch.load(path, weights_only=False)`
and restore compatible weights, routing and decay settings before resuming.
The checkpoint includes an in-memory tape as a SQLite dump. A file-backed tape
stores its absolute path and database identity. Keep that database available
at the recorded path, and use SQLite's backup API for a portable disk snapshot.

Replaying identical observations and route weights is idempotent in the
archive. A conflicting payload, route, or omitted prior observation raises
`TapeConflictError`. Inserts from the conflicting chunk are rolled back.
Restore a matching state or use a separate archive for a divergent stream.
Positions with no archived write are omitted from the tape.

`IDTape(path, wal=True)` enables SQLite WAL with `synchronous=NORMAL`
for file-backed tapes. It can improve read/write concurrency. A system
crash or power loss may roll back recent commits in this mode.
[SQLite durability modes](https://www.sqlite.org/pragma.html#pragma_synchronous)

## Numerical behavior

PyTorch validates each input call, projection range and user write weights
before updating any chunk. The native backend rejects a nonfinite token and
leaves its history and moments untouched. Native prefill validates all raw
inputs before decoding. A projection range failure during native prefill
reports how many earlier tokens were accepted.

Rust checks Cholesky pivots before division or square root. If the solve
fails, the node uses its finite intercept and centered value variance for
that read. The moment buffers contain only weighted observations. The
allocation test instruments the decoder with routing, sketches, contextual
history and decay enabled. The Python wrapper can still allocate conversion
and output buffers.

Balancing averages the right-branch probability **per visited node** before
squaring its deviation from one half. The reduction weights node penalties
by detached visit masses. Opposite imbalances at different nodes therefore
contribute separately.

## Optional forgetting

Set `SHRFConfig(decay=0.999)` for a roughly 693-token half-life. The
default 1.0 retains the cumulative behavior. At position `t`, the
read uses `S[t-1]` and the write applies
`S[t] = decay * S[t-1] + delta[t]`. Untouched nodes are aged lazily.
Within a chunk, factoring out chunk-local powers preserves the causal
segmented prefix operation. In PyTorch the added timestamp memory is
eight bytes per node or sketch slot, per tree and batch element. Rust
shares timestamps between a leaf and its sketch.

A controlled switch from 256 positive observations to 32 negative ones
produced a final prediction of +0.778 with cumulative memory and −0.931
with decay 0.9. This is a controlled adaptation probe. Decay also reduces
the influence of genuinely useful old records. Archived observations keep
the original write weights.

## Follow-up benchmarks

With the same width-4096 CPU setup and 300 tokens per repeat, the audit
run measured **0.961 ms/token** for the Rust core through its Python wrapper
and **9.350 ms/token** for PyTorch. The maximum absolute output difference
was 1.54e-6.
[Audit timing data](../results/rust_bench_audit.json)

The opt-in language benchmark `--models hybrid` uses a GQA layer after every
three SHRF layers in each complete four-layer group, with GQA at the final
layer. On a single width-64, four-layer, 100-step run on held-out Python
documentation, hybrid scored 4.037 bits per byte. GQA scored 3.794, gated SHRF
with a local filter scored 3.011, and the local-only control scored 3.001.
The run measures short-context language modeling and uses different parameter
counts across these configurations.
[Hybrid audit data](../results/language_hybrid_audit.json)

Delta-rule memory changes the learning objective. Recursive inverse updates
require treating centering, changing ridge regularization and sketch updates
explicitly to preserve SHRF's regression semantics. Both remain separate
research experiments.
