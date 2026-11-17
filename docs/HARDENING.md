# Inference and archive hardening

## Checkpoints and retries

`torch.save(state, path)` can serialize a state with an `IDTape`.
Load a checkpoint you created with `torch.load(path, weights_only=False)`
and restore compatible weights, routing and decay settings before resuming.
The default checkpoint embeds a consistent SQLite dump for memory and disk
tapes and restores it as an independent in-memory archive. Snapshot size
grows with the archive. For a small checkpoint that reopens an existing
archive, set `IDTape(path, checkpoint_mode="reference")`. Reference mode
records its host path and database identity.

Replaying identical observations and route weights is idempotent in the
archive. A conflicting payload, route, or omitted prior observation raises
`TapeConflictError`. Inserts from the conflicting chunk are rolled back.
Restore a matching state or use a separate archive for a divergent stream.
Positions with no archived write are omitted from the tape.
Record operations use SQLite savepoints. A caller's outer transaction
can still commit or roll back the entire operation.

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
The native ABI rejects null or misaligned buffer pointers and overflowing
prefill sizes. Callers must still supply live pointers to appropriately sized
disjoint buffers. Rebuild the library for ABI version 5.

## Ridge numerics and sketch re-admission

The configured ridge diagonal stays fixed as node mass increases. PyTorch
subtracts per-stream key, value, and feature origins before forming moments.
Rust subtracts a value origin and retains FP64 moments. Adding the value
origin back preserves the affine prediction with an unpenalized intercept.
PyTorch centers and solves in FP64. The default compact resident tensors
can still store moments in FP32. For streams with substantially changing
offsets, FP64 resident state is available through `init_state(dtype=torch.float64)`.
A first-observation origin reduces cancellation from a persistent offset.
Nearly exact fits and substantial drift can still make FP32 raw moments
relative to that origin less accurate.

In a reproducible regression with 100,000 FP32 values, an offset of 20,000,
and small injected noise, the revised RSS was 160.221832 versus 160.222350
with FP64 moments, a relative discrepancy of 0.000323%. Test
`test_anchored_fp32_rss_matches_fp64_with_large_offset` encodes that case.

Call `consolidate(state, budget)` at a maintenance boundary to rank
observed leaves by affine residual mass. Empty leaves stay unassigned.
A selected leaf can regain a sketch by replaying its archived observations
with their original route weights, current decay and saved origins. Rebuild
work is bounded by the selected leaves' archived records and sits outside
token decoding.

Full cross-chunk gradients are enabled by default. The resulting graph
grows with context. Set `detach_chunks=True` for truncated BPTT when its
memory tradeoff is appropriate. The regression test backpropagates from
token 131 to token 1 with the default chunk size of 64.

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

## Shared-vocabulary long-stream probe

All non-needle pair positions carried competing key/value pairs from the
same vocabulary during training and evaluation. The queried key was
excluded from distractor keys to keep the correct answer unambiguous.
With the revised decoder, one held-out stream at 1,000,000 tokens
recovered payload 29 correctly at 5% depth in approximately 43 seconds.
This single synthetic trial establishes that a separate filler vocabulary
is unnecessary for that example. It gives no estimate of average recall
on long natural documents.
[Raw shared-vocabulary probe](../results/native_length_shared_vocab.json)

## Follow-up benchmarks

After the ridge changes, the same width-4096 timing harness measured
**0.889 ms/token** through the Rust wrapper and **9.360 ms/token** in
PyTorch. Maximum absolute output difference was 1.54e-6. GPU prefill at
4,096 tokens measured about **2,247 tokens/s** versus **2,342 tokens/s**
in the earlier run with the same settings. These are separate benchmark
runs on the same device.
[Current CPU and GPU timing](../results/rust_bench_numerics.json) ·
[Prefill measurements](../results/prefill_numerics.json)

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
