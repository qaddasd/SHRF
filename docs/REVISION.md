# Long streams and contextual addressing

For a projected vector \(a_{t,e}=W_e x_t\), the optional local key encoder uses

$$
z_{t,e}=\tanh\left(a_{t,e}+\sum_{i=1}^{K}\alpha_{e,i}a_{t-i,e}\right).
$$

Missing positions at sequence start have zero projection. The learned
coefficients start at zero. Read \(t\) uses this contextual key. With successor
binding, write \(t\) uses the previous contextual key and the current value.
\(K=0\) recovers the original model and its checkpoint layout. A single-layer
model can distinguish the same token under different immediate histories.
If queries and writes have different histories, the model can learn a small
context coefficient.

The optional write gate is \(g_t=\sigma(w_g^\top x_t+b_g)^p\), for \(p\ge1\).
Its tail approaches \(\exp(p\,\mathrm{logit})\) as the logit decreases. Its bias
initializes to preserve the original write gate's initial mass \(\sigma(1)\).
Very negative logits may get negligible gradients as well as negligible mass.
Keep \(p=1\) as the default and evaluate \(p=2\)–\(4\) on length extrapolation
before adoption. The gate remains continuous and differentiable.

A decoder block can also add a low-rank causal local value path. Let
\(u_t=U x_t\in\mathbb R^h\). The block adds the following to the forest output
before the residual.

$$
V\sum_{i=1}^{J} c_i\odot u_{t-i}
$$

It stores \(J\) projected vectors, begins with zero filter taps, and exposes
short-range order to the MLP. The local-only baseline trains the same path
as a standalone sequence mixer.

An optional learned output gate multiplies the forest output by
\(\sigma(a^\top x_t+b)\) before adding the local value path. Use \(b=-3\) and
zero \(a\) at initialization for roughly 5% forest gain. The forest still
reads and writes normally, so the gate adds no resident state. It costs about
\(3d\) FLOPs per token per layer plus a scalar sigmoid. The output gate and
local value path are in the Python decoder block. Native inference covers
the forest, including its contextual keys and powered write gate.

The node sufficient statistics, deterministic beam semantics, empirical
Bayes weights, and recoverable consolidation in FORMALIZATION.md apply.
For a detached multi-token chunk, sort write events by node and sum each
node's batch in FP64 before one addition to its resident row. In a numerical
test, adding 5,120 writes of mass 0.01 directly to FP32 mass 1,000,000
lost every update, while coalescing 64 writes at a time retained 50.0 out
of 51.2 units. Choose FP64 state for recurrent streams where very small
updates must survive large resident mass. Rust uses FP64 state. Full-gradient
PyTorch training keeps its functional state update.

## Complexity deltas

At batch one the key filter adds \(b K E r L\) resident bytes, where \(b\)
is the accumulator width in bytes. With \(K=1\), \(E=4\), \(r=8\), \(L=32\) and FP32,
this is 4,096 bytes independent of context length. It costs \(2 K E r\) FLOPs
per token per layer, or 64 FLOPs in that configuration. The local value path
adds \(b J h L\) resident bytes and approximately \(4 d h + 2 J h + d\) FLOPs
per layer per token. With \(J=3\), \(h=32\), \(d=4096\), it adds 12,288 bytes
of FP32 state across 32 layers and 528,576 FLOPs per layer per token. The
write gate adds one scalar power for \(p\ne1\).

Coalescing writes adds an integer sort over at most batch × C × E × H
events per chunk, with one FP64 segmented reduction for the base rows
and one for leaf sketch rows when enabled. Additional work is linear
in the number of scalar contributions after sorting. Additional peak
work tensors scale with chunk size and visited nodes. Single-token decode
bypasses this sort.

Under successor binding, a length-\(C\) prefill chunk routes \(C+1\) distinct
keys for its reads and writes jointly. Their overlapping sketch features
are computed once as well. Ridge systems from all visited depths are
factored in one batched PyTorch launch before the hierarchical recursion.
This changes dispatch and workspace size. The individual systems have
the same asymptotic arithmetic.
