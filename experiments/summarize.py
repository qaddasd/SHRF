"""Generate the research results report from experiment and accounting JSON."""
import json
from pathlib import Path


def load(name):
    return json.loads(Path(f"results/{name}.json").read_text(encoding="utf-8"))


def pct(row):
    return f"{100*row['accuracy']:.1f}%"


def append_current_results(out, cx):
    local = cx["contextual_local"]["shrf"]
    out.extend(["## Contextual decoder revision", "",
                ("The compact D9 profile with two key lags, a three-lag rank-32 local path "
                "and a memory output gate adds 20,480 FP32 resident bytes across 32 layers. "
                f"It uses {local['hot_fp32_bytes']/1e9:.6f} GB of resident state and approximately "
                f"{local['prefill_flops_per_layer']/1e6:.3f} / {local['decode_flops_per_layer']/1e6:.3f} "
                "MFLOPs per layer per token for prefill / decode. See docs/REVISION.md."), ""])
    lang = Path("results/language.json")
    if lang.exists():
        first = load("language")
        comparisons = {}
        for name in ("language", "language_seed79", "language_control45", "language_control79"):
            if Path(f"results/{name}.json").exists():
                run = load(name)
                comparisons.setdefault(run["config"]["seed"], {}).update(run["runs"])
        out.extend(["### Held-out Python documentation", "",
                    ("The two-layer, width-64 byte models see identical batches for each seed "
                    "and hold out complete topics. The run uses 180 steps of batch 8 at length 96 "
                    "for 138,240 training targets per model and 9,216 sampled validation targets. "
                    "Model size and wall-clock compute differ. GQA has 4 query heads and 2 KV heads. "
                    "The three local variants start from identical local weights for each seed. "
                    f"The corpus contains {len(first['train']['topics'])} train and "
                    f"{len(first['validation']['topics'])} validation topics."), "",
                    "| Model | Parameters | Seed 45 bpb | Seed 79 bpb |",
                    "|---|---:|---:|---:|"])
        for name in ("gqa", "shrf", "shrf_context", "shrf_local", "shrf_gated", "local_only"):
            records = [comparisons.get(seed, {}).get(name) for seed in (45, 79)]
            valid = next((x for x in records if x), None)
            if valid:
                vals = [f"{v['val_bpb']:.3f}" if v else "—" for v in records]
                out.append(f"| {name} | {valid['parameters']:,} | {vals[0]} | {vals[1]} |")
        out.extend(["", ("Lower bits per byte is better. This small, short-context corpus is a "
                    "language-model sanity check. The local-only control measures how much of "
                    "the short-context gain is attributable to the local path."), ""])

    if Path("results/needle_blocks.json").exists():
        probe = load("needle_blocks")
        out.extend(["### Long-range contribution in the combined decoder", "",
                    ("Two-layer width-48 models train on 96-token synthetic contexts with a "
                    "random payload and early needle. The 512-token evaluation contains "
                    "128 fresh episodes. Uniform chance is 6.25%."),
                    "", "| Model | 96-token validation | 512-token validation |",
                    "|---|---:|---:|"])
        for name, r in probe["runs"].items():
            out.append(f"| {name} | {pct(r['train']['eval'])} | {pct(r['accuracy_512'])} |")
        out.append("")

    if Path("results/length.json").exists():
        out.extend(["### Write-gate length extrapolation", "",
                    ("Each condition has 16 fresh needle queries. Training contexts contain 80 "
                    "tokens and evaluation plants the needle at 5% relative depth. At 2,048 "
                    "tokens one condition fills 5% of pair slots with competing facts."),
                    "", "| Gate power | Mean filler mass | 2,048 with decoys | 32,768 | 128,000 |",
                    "|---:|---:|---:|---:|---:|"])
        for power, entry in load("length")["runs"].items():
            sweep = entry["sweep"]
            cells = [pct(sweep[k]) if k in sweep else "—"
                     for k in ("2048/0.05", "32768/0", "128000/0")]
            out.append(f"| {power} | {entry['gates']['filler']['mean']:.3f} | "
                       f"{cells[0]} | {cells[1]} | {cells[2]} |")
        out.append("")

    if Path("results/native_length.json").exists():
        native = load("native_length")
        out.extend([f"### Rust needle probe at {native['length']:,} tokens", "",
                    ("One trained width-48 layer with three planted bindings and separate "
                    "filler vocabulary. Torch performs embedding, normalization and the final "
                    "classifier around the Rust streaming core. Wall time includes those CPU "
                    "operations. This small probe has no competing distractor bindings."),
                    "", "| Needle position | Correct | Seconds |", "|---:|---|---:|"])
        for r in native["episodes"]:
            out.append(f"| {100*r['fraction']:g}% | {r['correct']} | {r['seconds']:.1f} |")
        out.append("")

    if all(Path(f"results/{k}.json").exists() for k in ("prefill_v1", "prefill")):
        old, new = load("prefill_v1"), load("prefill")
        out.extend(["### Measured GPU prefill", "",
                    (f"One SHRF layer on {new['gpu']} with d=512, E=4, D=8, r=8, q=32, m=8, "
                    "beam 2, chunk 64 and batch 1. Median of five synchronized runs."),
                    "", "| Prompt tokens | Previous SHRF tok/s | Current SHRF tok/s | Speedup |",
                    "|---:|---:|---:|---:|"])
        for a, b in zip(old["rows"], new["rows"], strict=False):
            assert a["length"] == b["length"]
            out.append(f"| {a['length']:,} | {a['tokens_s']:,.0f} | "
                       f"{b['tokens_s']:,.0f} | {b['tokens_s']/a['tokens_s']:.2f}× |")
        out.append("")


def main():
    cx, ex, rb = map(load, ("complexity", "experiments", "rust_bench"))
    ab = load("ablations_v2") if Path("results/ablations_v2.json").exists() else load("ablations")
    c = cx["shrf"]
    small = cx["compact"]["shrf"]
    second = Path("results/experiments_seed73.json")
    reports = [ex, load("experiments_seed73")] if second.exists() else [ex]
    device = ex["device"]
    out = [
        "# SHRF experiments and accounting", "",
        "[Numerical and portability follow-up](docs/HARDENING.md)", "",
        "## Calculated resident state",
        "",
        ("Decimal GB, batch one, d=4,096, 32 layers. MHA/GQA KV uses FP16. "
        "SHRF stores FP32 moments in PyTorch. The FP16 column is a storage-only model."),
        "",
        "| Context | MHA FP16 | GQA8 FP16 | SHRF D11 FP32 | D11 FP16 modeled | SHRF D9 FP32 | D9 GQA8 ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cx["contexts"]:
        out.append(f"| {r['tokens']:,} | {r['mha']['kv_bytes']/1e9:.3f} | {r['gqa8']['kv_bytes']/1e9:.3f} | "
                   f"{c['hot_fp32_bytes']/1e9:.3f} | {c['hot_fp16_storage_bytes']/1e9:.3f} | "
                   f"{small['hot_fp32_bytes']/1e9:.3f} | {r['gqa8']['kv_bytes']/small['hot_fp32_bytes']:.1f}× |")
    cold = c["ideal_packed_fp32_cold_bytes_per_token"]
    out.extend(["", (f"The ideal packed FP32 cold archive, with 16-bit leaf IDs and 32-bit weights, "
                     f"adds {cold:,} B/token across 32 layers ({cold*128000/1e9:.3f} GB at 128k, "
                     f"{cold/1e3:.3f} GB at 1M, {cold/100:.3f} GB at 10M). SQLite stores exact arrays "
                     "and has additional BLOB and index overhead. State and tape exclude model weights."), "",
                "## Calculated arithmetic per layer", "",
                "| Context | MHA prefill / decode (GFLOPs) | GQA8 prefill / decode (GFLOPs) | SHRF D11 prefill / decode (MFLOPs) |",
                "|---:|---:|---:|---:|"])
    for row in cx["contexts"]:
        def pair(key, row=row):
            r = row[key]
            return f"{r['prefill_flops_per_layer']/1e9:.3f} / {r['decode_flops_per_layer']/1e9:.3f}"
        out.append(f"| {row['tokens']:,} | {pair('mha')} | {pair('gqa8')} | "
                   f"{c['prefill_flops_per_layer']/1e6:.3f} / {c['decode_flops_per_layer']/1e6:.3f} |")
    out.extend(["", ("Prefill is averaged over causal prefixes. The estimates omit sorting and "
                "transcendentals, so see the CPU measurement below for latency."), "",
                "## Synthetic recall",
                "",
                (f"PyTorch {ex['torch']}, device {device} (GeForce GTX 1650 in the recorded CUDA runs). "
                "One decoder block, d=48, E=2, D=3, r=8, q=16, m=6, beam 2. "
                "The recorded runs use 160 AdamW steps, batch 12, and fresh random evaluation episodes. "
                "The model learned its write "
                "gate from loss without an oracle mask. The query-only control disables every write."),
                "",
                "| Task | Seed | Accuracy | Query-only | Chance | Scored tokens | Train seconds |",
                "|---|---:|---:|---:|---:|---:|---:|"])
    for report in reports:
        for name, r in report["runs"].items():
            out.append(f"| {name} | {r['seed']} | {pct(r['eval'])} | {pct(r['query_only'])} | "
                       f"{100*r['chance']:.2f}% | {r['eval']['targets']} | {r['seconds']:.2f} |")
    out.extend(["", "| Task | What success measures | Failure signature |",
                "|---|---|---|",
                "| MQAR-style | Independent random payloads from repeated episode keys | Near query-only accuracy or confusion between payloads |",
                "| Semantic alias | Random payload recovered with a held-out alias in frozen synthetic embedding geometry | Accuracy falls on novel renderings or noisy features |",
                "| Induction | Three-token continuation in free-running decoding | Exact span falls even if individual teacher-forced tokens score well |",
                "| Dense cluster | Bind nearby but distinct key vectors to separate payloads | Interference as distinct bindings per episode increase |",
                "| Needle | Recover an independent payload far from the query | Accuracy declines as distractors start carrying competing facts |", ""])
    for report in reports:
        runs = report["runs"]
        if not {"induction_span", "dense_cluster", "needle"} <= runs.keys():
            continue
        rollout = runs["induction_span"]["rollout"]
        dense = runs["dense_cluster"]["stress"]
        needle = runs["needle"]
        distances = ", ".join(f"{label} {pct(v)} ({v['targets']} answers)"
                              for label, v in needle["distance"].items())
        density = ", ".join(f"{label} keys {pct(v)}" for label, v in dense.items())
        out.extend([(f"Seed {runs['induction_span']['seed']}. Exact autoregressive span "
                    f"{100*rollout['exact_span']:.1f}% ({rollout['targets']//3} spans). Dense load "
                    f"{density}. Needle accuracy by context length and relative position {distances}."), ""])
        if "decoys" in needle:
            out.extend(["Needle at 2,048 tokens with competing key/value decoys drawn from the same "
                        "vocabulary (queried key excluded). " + ", ".join(
                            f"{100*float(k):g}% decoy slots → {pct(v)}" for k, v in needle["decoys"].items())
                        + ". Each ratio scored on 32 queries.", ""])
    out.extend([("The 128k probe uses few needles, short trained contexts, and few evaluation episodes. "
                "The result measures extrapolation in this generator. It provides no estimate of "
                "perplexity, unrestricted long-context recall, or 1M/10M quality."), "",
                "## Ablations", "",
                ("Single seed, 120 training steps. The held-out alias evaluation uses six bindings "
                "per episode by default. The load test increases this to sixteen."),
                "", "| Variant | 6 bindings | 16 bindings | 16 bindings, noise 0.75 |",
                "|---|---:|---:|---:|"])
    for name, r in ab["runs"].items():
        noisy = pct(r["noisy_overload"]) if "noisy_overload" in r else "—"
        out.append(f"| {name} | {pct(r['eval'])} | {pct(r['overload'])} | {noisy} |")
    out.extend(["", ("Both clean loads saturate for many variants. The noisy load and the "
                "controlled numeric probes below provide additional measures."), "",
                "| Corruption | Setting | Accuracy |",
                "|---|---:|---:|"])
    for family, rows in ab["robustness"].items():
        for setting, v in rows.items():
            out.append(f"| {family} | {setting} | {pct(v)} |")
    out.extend(["", ("Noise is scaled by embedding standard deviation. Rank corruption is an "
                "unseen random orthogonal projection; rank zero sets the input to a constant. "
                "This probe stresses distribution shift as well as address collisions."), ""])
    if Path("results/probes.json").exists():
        probes = load("probes")
        out.extend(["### Controlled numeric probes", "",
                    ("Nonlinear regression held-out MSE for cos(5x), averaged over 10 sketch seeds. "
                    "Width zero uses only affine ridge."), "",
                    "| Sketch width | MSE |", "|---:|---:|"])
        for width, mse in probes["nonlinear_mse"].items():
            out.append(f"| {width} | {mse:.5f} |")
        out.extend(["", ("Shrinkage mean-estimation MSE on 8,192 independently sampled child nodes "
                    "per row. The true mean is 0.1 and noise standard deviation is 1. The "
                    "parent mean is given as zero to isolate shrinkage."), "",
                    "| Node observations | Raw child MSE | Shrunk MSE |", "|---:|---:|---:|"])
        for n, mse in probes["shrinkage_mse"].items():
            out.append(f"| {n} | {mse['raw']:.5f} | {mse['shrunken']:.5f} |")
        out.append("")
    out.extend(["## Measured CPU decode", "",
                "Intel Core i5-10400F at 2.90 GHz (nominal).",
                ("Batch one, one Torch thread, median of three runs. Rust includes its Python FFI "
                "wrapper. PyTorch uses FP32 state, Rust FP64. These rows compare SHRF implementations "
                "on the same machine and inputs."), "",
                "| Shape (d, E, D, r, q, m, B) | Tokens/run | PyTorch ms/token | Rust ms/token | Speedup | Max absolute difference |",
                "|---|---:|---:|---:|---:|---:|"])
    for r in rb["runs"]:
        shape = ", ".join(str(r[k]) for k in ("d", "E", "D", "r", "q", "m", "beam"))
        out.append(f"| {shape} | {r['tokens']} | {r['torch_ms_per_token']:.3f} | "
                   f"{r['rust_ms_per_token']:.3f} | {r['speedup']:.1f}× | {r['max_abs_diff']:.2e} |")
    out.extend(["", ("The large row measures one SHRF layer. It supplies no end-to-end Transformer "
                "or wall-clock GPU comparison with fused MHA/GQA."), ""])
    append_current_results(out, cx)
    Path("RESULTS.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
