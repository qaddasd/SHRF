# Reproducing SHRF experiments

Run commands from the repository root after installing
[requirements.txt](../requirements.txt).

```console
cargo build --release --manifest-path rust/shrf-core/Cargo.toml
python -m unittest discover -s tests -v
python -m experiments.demo
python -m shrf.complexity
```

## Synthetic recall and ablations

```console
python -m experiments.run_all --device cuda
python -m experiments.run_all --device cuda --seed 73 --long 512 2048 8192 128000 --output results/experiments_seed73.json
python -m experiments.ablations --device cuda
python -m experiments.probes
python -m experiments.needle_blocks --device cuda
python -m experiments.length --device cuda
python -m experiments.native_length
```

`--device cpu` is available for the Python-only experiments. The Rust
million-token probe accepts `--training-device cpu` and decodes on CPU.

## Language and throughput comparisons

```console
python -m experiments.language --device cuda
python -m experiments.language --device cuda --seed 79 --output results/language_seed79.json
python -m experiments.bench_rust
python -m experiments.bench_stream
python -m experiments.summarize
```

The audit uses an optional four-layer hybrid comparator and separate timing data.

```console
python -m experiments.language --device cuda --layers 4 --width 64 --context 64 --steps 100 --eval-every 50 --batch 8 --models gqa shrf_gated local_only hybrid --output results/language_hybrid_audit.json
python -m experiments.bench_rust --output results/rust_bench_audit.json
```

The held-out corpus comes from the Python documentation bundled with the
interpreter. The result file records the interpreter version and topic
split. To retime an earlier checkout, use the same hardware and invoke
`python -m experiments.bench_stream --source PATH --label v1 --output results/prefill_v1.json`.

## README figure

```console
python -m pip install Pillow
python -m experiments.plot_readme
```

The figure reads the checked-in JSON. See [RESULTS.md](../RESULTS.md) for
the scope and configuration of each benchmark.
