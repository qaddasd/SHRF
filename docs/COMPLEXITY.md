# SHRF complexity model

One multiply-add is two FLOPs. All rows are per sequence at batch one.
Let r be key width, q value width, m sketch width, L model layers, E trees,
D depth, N=2^(D+1), J=2^D, B beam width, H=1+Σ min(B,2^j), and
W=min(B,J). One base row has S=2+r+q+r²+rq numbers.
One sketch row has X=m+rm+m²+mq numbers (full matrices).
Hot state bytes across L layers with K sketch slots per tree is

    L [b E (N S + (K+1) X + r) + 8 E J + 8].

Here b is accumulator bytes, the map uses int64, and the last terms
account for the previous projected key and per-sequence int64 count. The
sketch includes a shared sentinel slot per tree. With sketches disabled,
omit the entire sketch term and the 8 E J map. Model
weights, allocator overhead, training activations and the external tape are
excluded. PyTorch uses b=4 by default for FP16/BF16 model weights.
The hypothetical b=2 row is a storage bound, with no claim that half
precision is safe for accumulation or supported by torch.linalg.solve.

For MHA/GQA (32 query heads, head dimension 128, d=4096), FP16 KV bytes
are 4 L T d_kv, with d_kv=4096 MHA and 1024 GQA. The leading matmul FLOPs
per layer per generated token with T cached tokens are

    decode = 4 d² + 4 d d_kv + 4 T d
    average causal prefill token = 4 d² + 4 d d_kv + 2 (T+1) d .

The two attention matmuls use d query channels for both MHA and GQA.
These omit softmax, normalization and memory movement.

For SHRF, use F(p)=2p³/3+2p²q as the leading LU factor/solve count and
include moment formation, accumulation, centering, read, shrinkage, and
projection terms in the executable calculator. Prefill performs the same
fits, plus segmented scan/gather/scatter and integer sorting at each depth.
Per chunk C, the number of events at a depth is at most 2 C E B per
sequence, and the event vector has length S or X.
Comparison sorting is O(P log P) for P events; PyTorch backend details
may differ. Recurrent decode does not sort. Sigmoids, cosine, cache
traffic, kernel launch costs, LU pivoting, index arithmetic and archive I/O
are separate from the FLOP estimates. See results/complexity.json for
the concrete configuration, component counts and context sweep.

In the reference training implementation, one chunk's node event arrays
and ridge inputs scale as O(batch C E [H S + W X]) elements, plus solver
intermediates. FP64 segmented accumulation uses temporary buffers. Autograd
retains per-chunk work until backward, and full cross-chunk gradients add
functional heap snapshots. A deployment using chunk-wise loss/backward or
activation checkpointing needs a separately measured training peak.

The optional contextual and gated local paths add only projected lag
state. The formulas and concrete example are in REVISION.md. The
calculator also accounts for shared read/write routes, sketch-feature
reuse and the FP64 chunk commit reduction in the current prefill path.
