# SHRF experiments and accounting

## Calculated resident state

Decimal GB, batch one, d=4,096, 32 layers. MHA/GQA KV uses FP16. SHRF stores FP32 moments in PyTorch. The FP16 column is a storage-only model.

| Context | MHA FP16 | GQA8 FP16 | SHRF D11 FP32 | D11 FP16 modeled | SHRF D9 FP32 | D9 GQA8 ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 128,000 | 67.109 | 16.777 | 2.045 | 1.024 | 0.512 | 32.8× |
| 1,000,000 | 524.288 | 131.072 | 2.045 | 1.024 | 0.512 | 256.2× |
| 10,000,000 | 5242.880 | 1310.720 | 2.045 | 1.024 | 0.512 | 2562.4× |

The ideal packed FP32 cold archive, with 16-bit leaf IDs and 32-bit weights, adds 13,952 B/token across 32 layers (1.786 GB at 128k, 13.952 GB at 1M, 139.520 GB at 10M). SQLite stores exact arrays and has additional BLOB and index overhead. State and tape exclude model weights.

## Calculated arithmetic per layer

| Context | MHA prefill / decode (GFLOPs) | GQA8 prefill / decode (GFLOPs) | SHRF D11 prefill / decode (MFLOPs) |
|---:|---:|---:|---:|
| 128,000 | 1.183 / 2.231 | 1.132 / 2.181 | 3.244 / 2.922 |
| 1,000,000 | 8.326 / 16.518 | 8.276 / 16.468 | 3.244 / 2.922 |
| 10,000,000 | 82.054 / 163.974 | 82.004 / 163.924 | 3.244 / 2.922 |

Prefill is averaged over causal prefixes. The estimates omit sorting and transcendentals, so see the CPU measurement below for latency.

## Synthetic recall

PyTorch 2.6.0+cu124, device cuda (GeForce GTX 1650 in the recorded CUDA runs). One decoder block, d=48, E=2, D=3, r=8, q=16, m=6, beam 2. The recorded runs use 160 AdamW steps, batch 12, and fresh random evaluation episodes. The model learned its write gate from loss without an oracle mask. The query-only control disables every write.

| Task | Seed | Accuracy | Query-only | Chance | Scored tokens | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| mqar | 37 | 100.0% | 2.3% | 6.25% | 512 | 6.24 |
| semantic_alias | 37 | 100.0% | 7.6% | 6.25% | 512 | 5.59 |
| induction_span | 37 | 100.0% | 3.6% | 3.12% | 384 | 5.60 |
| dense_cluster | 37 | 100.0% | 5.1% | 6.25% | 512 | 5.89 |
| needle | 37 | 100.0% | 7.0% | 6.25% | 128 | 14.69 |
| mqar | 73 | 100.0% | 5.7% | 6.25% | 512 | 13.49 |
| semantic_alias | 73 | 100.0% | 6.6% | 6.25% | 512 | 9.86 |
| induction_span | 73 | 100.0% | 2.6% | 3.12% | 384 | 9.94 |
| dense_cluster | 73 | 100.0% | 7.0% | 6.25% | 512 | 9.89 |
| needle | 73 | 100.0% | 9.4% | 6.25% | 128 | 19.19 |

| Task | What success measures | Failure signature |
|---|---|---|
| MQAR-style | Independent random payloads from repeated episode keys | Near query-only accuracy or confusion between payloads |
| Semantic alias | Random payload recovered with a held-out alias in frozen synthetic embedding geometry | Accuracy falls on novel renderings or noisy features |
| Induction | Three-token continuation in free-running decoding | Exact span falls even if individual teacher-forced tokens score well |
| Dense cluster | Bind nearby but distinct key vectors to separate payloads | Interference as distinct bindings per episode increase |
| Needle | Recover an independent payload far from the query | Accuracy declines as distractors start carrying competing facts |

Seed 37. Exact autoregressive span 100.0% (128 spans). Dense load 4 keys 100.0%, 12 keys 100.0%, 20 keys 98.6%. Needle accuracy by context length and relative position 512@0.05 100.0% (16 answers), 512@0.5 100.0% (16 answers), 512@0.95 100.0% (16 answers), 2048@0.05 100.0% (16 answers), 2048@0.5 100.0% (16 answers), 2048@0.95 100.0% (16 answers), 8192@0.05 100.0% (16 answers), 8192@0.5 100.0% (16 answers), 8192@0.95 100.0% (16 answers).

Needle at 2,048 tokens with competing key/value decoys drawn from the same vocabulary (queried key excluded). 0% decoy slots → 100.0%, 1% decoy slots → 100.0%, 5% decoy slots → 100.0%, 25% decoy slots → 100.0%. Each ratio scored on 32 queries.

Seed 73. Exact autoregressive span 100.0% (128 spans). Dense load 4 keys 100.0%, 12 keys 100.0%, 20 keys 99.6%. Needle accuracy by context length and relative position 512@0.05 100.0% (16 answers), 512@0.5 100.0% (16 answers), 512@0.95 100.0% (16 answers), 2048@0.05 100.0% (16 answers), 2048@0.5 100.0% (16 answers), 2048@0.95 100.0% (16 answers), 8192@0.05 100.0% (16 answers), 8192@0.5 100.0% (16 answers), 8192@0.95 100.0% (16 answers), 128000@0.05 87.5% (8 answers).

Needle at 2,048 tokens with competing key/value decoys drawn from the same vocabulary (queried key excluded). 0% decoy slots → 100.0%, 1% decoy slots → 100.0%, 5% decoy slots → 100.0%, 25% decoy slots → 100.0%. Each ratio scored on 32 queries.

The 128k probe uses few needles, short trained contexts, and few evaluation episodes. The result measures extrapolation in this generator. It provides no estimate of perplexity, unrestricted long-context recall, or 1M/10M quality.

## Ablations

Single seed, 120 training steps. The held-out alias evaluation uses six bindings per episode by default. The load test increases this to sixteen.

| Variant | 6 bindings | 16 bindings | 16 bindings, noise 0.75 |
|---|---:|---:|---:|
| full | 100.0% | 100.0% | 55.1% |
| no_sketch | 100.0% | 100.0% | 56.2% |
| no_shrinkage | 100.0% | 100.0% | 53.1% |
| E1 | 100.0% | 100.0% | 50.8% |
| E4 | 100.0% | 100.0% | 62.1% |
| r4 | 100.0% | 97.9% | 45.1% |
| r12 | 100.0% | 100.0% | 53.1% |
| m2 | 100.0% | 100.0% | 51.6% |
| m12 | 100.0% | 100.0% | 55.1% |
| no_binding | 7.0% | 6.1% | 6.6% |
| beam1 | 100.0% | 100.0% | 52.7% |

Both clean loads saturate for many variants. The noisy load and the controlled numeric probes below provide additional measures.

| Corruption | Setting | Accuracy |
|---|---:|---:|
| noise | 0.0 | 100.0% |
| noise | 0.2 | 100.0% |
| noise | 0.5 | 93.4% |
| noise | 1.0 | 44.1% |
| noise | 2.0 | 10.4% |
| rank | 16 | 71.5% |
| rank | 8 | 34.4% |
| rank | 4 | 23.6% |
| rank | 2 | 11.7% |
| rank | 0 | 5.1% |

Noise is scaled by embedding standard deviation. Rank corruption is an unseen random orthogonal projection; rank zero sets the input to a constant. This probe stresses distribution shift as well as address collisions.

### Controlled numeric probes

Nonlinear regression held-out MSE for cos(5x), averaged over 10 sketch seeds. Width zero uses only affine ridge.

| Sketch width | MSE |
|---:|---:|
| 0 | 0.43478 |
| 2 | 0.26345 |
| 4 | 0.13721 |
| 8 | 0.05544 |
| 16 | 0.00465 |

Shrinkage mean-estimation MSE on 8,192 independently sampled child nodes per row. The true mean is 0.1 and noise standard deviation is 1. The parent mean is given as zero to isolate shrinkage.

| Node observations | Raw child MSE | Shrunk MSE |
|---:|---:|---:|
| 2 | 0.50461 | 0.33045 |
| 4 | 0.25174 | 0.14909 |
| 16 | 0.06245 | 0.03457 |
| 64 | 0.01594 | 0.01110 |

## Measured CPU decode

Intel Core i5-10400F at 2.90 GHz (nominal).
Batch one, one Torch thread, median of three runs. Rust includes its Python FFI wrapper. PyTorch uses FP32 state, Rust FP64. These rows compare SHRF implementations on the same machine and inputs.

| Shape (d, E, D, r, q, m, B) | Tokens/run | PyTorch ms/token | Rust ms/token | Speedup | Max absolute difference |
|---|---:|---:|---:|---:|---:|
| 64, 4, 6, 8, 32, 8, 2 | 300 | 6.701 | 0.306 | 21.9× | 1.16e-06 |
| 4096, 4, 11, 8, 64, 8, 2 | 300 | 10.560 | 1.131 | 9.3× | 1.53e-06 |

The large row measures one SHRF layer. It supplies no end-to-end Transformer or wall-clock GPU comparison with fused MHA/GQA.

## Contextual decoder revision

The compact D9 profile with two key lags, a three-lag rank-32 local path and a memory output gate adds 20,480 FP32 resident bytes across 32 layers. It uses 0.511537 GB of resident state and approximately 3.516 / 3.245 MFLOPs per layer per token for prefill / decode. See docs/REVISION.md.

### Held-out Python documentation

The two-layer, width-64 byte models see identical batches for each seed and hold out complete topics. The run uses 180 steps of batch 8 at length 96 for 138,240 training targets per model and 9,216 sampled validation targets. Model size and wall-clock compute differ. GQA has 4 query heads and 2 KV heads. The three local variants start from identical local weights for each seed. The corpus contains 63 train and 15 validation topics.

| Model | Parameters | Seed 45 bpb | Seed 79 bpb |
|---|---:|---:|---:|
| gqa | 90,432 | 3.209 | 3.064 |
| shrf | 74,466 | 3.762 | 3.751 |
| shrf_context | 74,474 | 3.735 | 3.733 |
| shrf_local | 82,858 | 3.163 | 3.056 |
| shrf_gated | 82,988 | 2.720 | 2.687 |
| local_only | 74,240 | 2.710 | 2.689 |

Lower bits per byte is better. This small, short-context corpus is a language-model sanity check. The local-only control measures how much of the short-context gain is attributable to the local path.

### Long-range contribution in the combined decoder

Two-layer width-48 models train on 96-token synthetic contexts with a random payload and early needle. The 512-token evaluation contains 128 fresh episodes. Uniform chance is 6.25%.

| Model | 96-token validation | 512-token validation |
|---|---:|---:|
| local_only | 7.8% | 6.2% |
| shrf_gated | 100.0% | 100.0% |

### Write-gate length extrapolation

Each condition has 16 fresh needle queries. Training contexts contain 80 tokens and evaluation plants the needle at 5% relative depth. At 2,048 tokens one condition fills 5% of pair slots with competing facts.

| Gate power | Mean filler mass | 2,048 with decoys | 32,768 | 128,000 |
|---:|---:|---:|---:|---:|
| 1 | 0.773 | 100.0% | 100.0% | 100.0% |
| 4 | 0.756 | 100.0% | 100.0% | 100.0% |

### Rust needle probe at 1,000,000 tokens

One trained width-48 layer with three planted bindings and separate filler vocabulary. Torch performs embedding, normalization and the final classifier around the Rust streaming core. Wall time includes those CPU operations. This small probe has no competing distractor bindings.

| Needle position | Correct | Seconds |
|---:|---|---:|
| 5% | True | 58.2 |
| 50% | True | 58.5 |

### Measured GPU prefill

One SHRF layer on NVIDIA GeForce GTX 1650 with d=512, E=4, D=8, r=8, q=32, m=8, beam 2, chunk 64 and batch 1. Median of five synchronized runs.

| Prompt tokens | Previous SHRF tok/s | Current SHRF tok/s | Speedup |
|---:|---:|---:|---:|
| 256 | 2,007 | 2,334 | 1.16× |
| 1,024 | 1,896 | 2,346 | 1.24× |
| 4,096 | 1,844 | 2,342 | 1.27× |
