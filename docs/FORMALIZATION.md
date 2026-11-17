# SHRF — Shrinkage Heap Regression Forest

## 1. Decoder interface and causal binding

For input hidden state x_t in R^d, each tree e uses z_{te}=tanh(W_e x_t) in R^r.
The read key is z_{te}. The write key is z_{t-1,e} and the write value is
v_t=W_v x_t. At sequence start the first write has mass zero. With lag zero,
the write key is z_{te}. A learned gate g_t=sigmoid(w_g^T x_t+b_g) controls
write mass, optionally multiplied by a caller-provided weight in [0,1]. Read t
uses writes with index s<t. Next-token language modeling can therefore recover
a past successor without letting the current target enter the memory. The
output is W_o times the arithmetic mean of E tree predictions.

REVISION.md defines the optional projected key history, powered write gate,
local value path and memory output gate. The core below describes their
default settings and the invariant shared by all configurations.

## 2. Heap and routing

Each tree is a fixed 1-indexed heap of depth D with 2^(D+1) slots including
sentinel 0. Split u has unit initialization a_u and threshold b_u (both can be
trained). For input z the right probability is p=sigmoid((a_u^T z-b_u)/temperature).
A root of mass 1 spawns children with parent mass times (1-p,p). At each depth
keep the B highest-mass nodes per tree with a left-first stable tie break.
Use original masses, so dropped mass is absorbed at its deepest retained
ancestor. The same traversal serves read and write. Gradients flow through
retained sigmoid weights. Beam membership changes discontinuously at selection
boundaries; there are no learned codebooks. Bounded depth and a root always
present make routing errors degrade to a broader model.
The auxiliary loss averages branch probabilities per visited node before
squaring deviations from one half. Node penalties are weighted by
detached visit mass.

Initially a_u is a seeded independent random direction. The optional
calibration uses labeled projected write keys and values independently of a
live sequence. At each visited internal node fit the current affine model,
compute residual error e_i=||v_i-f_u(z_i)||², weighted key center μ_u, and
H_u=Σ_i w_i e_i (z_i-μ_u)(z_i-μ_u)^T. For sufficient nonzero support, take its
leading eigenvector h_u sign-aligned to the random direction a_u^(0), set
a_u=normalize((1-α)a_u^(0)+αh_u), and set b_u to the weighted median of its scores.
Process levels in order so deeper calibration uses the new shallower routing.
Reinitialize or replay the memory after changing a projection or split.
Calibration uses temporary rows and is outside recurrent decoding.

## 3. Additive update and affine read

A node stores mass n, key sum A, value sum b, uncentered Gram G, cross
moment C, and value squared norm sum F. In PyTorch the stored coordinates
are k = z - z_origin and y = v - v_origin, with origins set by the first
token. Rust uses the value origin and FP64 moments. For retained write
node u, let w = g_t times its path mass. Add

    (n,A,b,G,C,F) += w (1,k,y,kk^T,ky^T,y^T y).

The means are mu = A/n and nu = b/n. Empty fits return zero. Form the
centered moments S = G - AA^T/n, U = C - Ab^T/n and V = F - b^T b/n.
Compute M = (S + rho I)^-1 U with the configured, mass-independent rho.
The affine prediction is v_origin + nu + (k-mu)^T M for nonempty fits.
The residual variance is max(0,V-tr(M^T U)-rho||M||^2)/(n q).
Fit and centering use FP64. The origin cancels out of parent/child mean
differences when both are occupied. See HARDENING.md for precision limits.

## 4. Nonlinear residual sketch and empirical Bayes

A retained leaf optionally stores the additive random feature moments for
φ_j(z)=sqrt(2/m) cos(ω_j^T z+β_j), ω_j Gaussian at configured bandwidth,
β_j uniform on [0,2π). Store Σwφ, Σw zφ^T, Σw φφ^T, Σw φv^T in addition
to the base moments. For the augmented centered Gram, let A0=S+ρ I,
J=Σw(z-μ)(φ-μ_φ)^T, K=Σw(φ-μ_φ)(φ-μ_φ)^T, Uφ=Σw(φ-μ_φ)(v-ν)^T.
The joint ridge uses the same key regularizer and a separate positive
ρ_φ=ρ_sketch. Its Schur-complement correction is

    H=(K+ρ_φ I-J^T A0^-1 J)^-1 (Uφ-J^T M)
    c_u(z)=[φ(z)-μ_φ-J^T A0^-1(z-μ)]^T H.

The implementation solves the small joint system directly and subtracts
the base prediction to recover the equivalent residual correction. This sketch
allocates capacity to nonlinear key-conditioned residuals, is mergeable by
addition and needs no per-prefix SVD. It has random-feature approximation
behavior rather than the Frequent Directions covariance error bound, and it
cannot retain arbitrary random bindings as occupancy grows.

Root P_root=f_root, plus its sketch correction when D=0 and a root sketch
is active. For a visited child u with parent p, let
s²=||ν_u-ν_p||²/q+10^-3 σ²_p+10^-5,
κ=clip(σ²_u/s²,κ_min,κ_max), λ=n/(n+κ).
Set P_u=P_p+λ(f_u+c_u-P_p). Here c_u=0 for internal nodes and inactive
sketch slots. σ²_u is from the affine model even at leaves. With shrinkage
disabled, λ is 1 for nonempty nodes and 0 for empty nodes. For tree e the
returned read is

    P_root + Σ_(retained nonroot u) π_u (P_u-P_parent(u)).

This telescopes to the expected prediction of retained prefixes, including
prefixes where mass fell out of the beam. The κ estimator is a
local empirical-Bayes heuristic. Its same-prefix parent and child estimates
are correlated. No posterior calibration is claimed.

## 5. Parallel training, decode, and consolidation

For each depth (and the leaf sketch), merge node-keyed read and write events
in (batch, tree, node, time, kind) order, with READ before WRITE at equal
time. An exclusive segmented sum over weighted write contributions yields
the exact prefix increment for each query. Add the starting heap row and fit
the node. Chunk endpoints scatter-add writes into the mutable inference
state. Full cross-chunk backpropagation uses functional state updates;
the detached-chunk option provides truncated BPTT. Neither changes the
forward equations. Single-token decode bypasses sorting and directly
gathers/scatters visited nodes.

For consolidation, rank observed leaves by their affine residual sums of
squares after decay. Allocate up to the configured budget per tree.
Retain existing moments for chosen leaves and reconstruct newly admitted
sketches from the tape. Unobserved leaves do not consume slots. Budget zero
is valid. The tape stores original projected observations and route weights
in SQLite, indexed by batch, tree, leaf and time. Maintenance can replay
a leaf's history, with work proportional to the recovered records.

## 6. Complexity

See the generated results and formulas in COMPLEXITY.md. Per-token active
nodes per tree are H=1+Σ_(j=1..D)min(B,2^j). Heap capacity is fixed at
construction. With D chosen proportional to log(max context), the bounded
beam visits O(log(max context)) nodes; for one allocated heap, recurrent
compute is independent of the number of tokens already consumed. The hot
state is sublinear in historical payload size; the cold archive grows
linearly. The fit includes r-by-r solves per visited node and one augmented
fit per visited leaf. The implementation does not assume cached ridge
coefficients or estimate GQA arithmetic from the number of KV heads.

At fixed resident capacity, arbitrary independent facts make at least
linear information requirements. Visited keys that have collapsed to the
same representation cannot be distinguished by the sketch, split choice,
or shrinkage. Layer widths, number of trained layers, and long-context
accuracy have to be evaluated jointly.

## References

- Rahimi & Recht, Random Features for Large-Scale Kernel Machines (2007).
  https://proceedings.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf
- Ghashami et al., Frequent Directions (2016).
  https://doi.org/10.1137/15M1009718
- Ainslie et al., Grouped-Query Attention (2023).
  https://aclanthology.org/2023.emnlp-main.298/
- Arora et al., Zoology (ICLR 2024).
  https://arxiv.org/abs/2312.04927
