# Bilingual SHRF reasoning experiment

The decoder reuses the SHRF layers of the 960-step byte model. The new
vocabulary adds frequent English and Russian words and characters to all
256 byte values. Decimal digits keep their original byte IDs. The
tokenizer distinguishes problem text, short solutions, final answers, and
the end of an answer.

Training mixes the Tensorov SFT Core train file with English and Russian
OpenAssistant exchanges and generated examples with verified answers.
Tensorov's Parquet stores ChatML as JSON strings in `messages_json`.
Long Tensorov responses can supply a prefix of the answer. Those examples
omit the end-of-answer target. Generated mathematics and logic cases
can supervise a short solution before the final answer.

Install PyArrow to read the Parquet file. Run from the repository root
with the files at the paths in the command.

```console
python -m experiments.reasoning train --parent checkpoints/shrf_distilled/best.pt --tensorov sft_core_train.parquet --oasst 2023-04-12_oasst_ready.messages.jsonl.gz --gsm-test checkpoints/curriculum_sources/gsm_test.jsonl --output checkpoints/shrf_tensorov_reasoning_v2 --steps 360 --context 192 --batch 6 --device cuda
python -m experiments.reasoning generate --checkpoint checkpoints/shrf_tensorov_reasoning_v2/model.pt --prompt "Solve 7x + 11 = 74. Give a number." --show-work
```

The checkpoint includes the vocabulary. Pass it to `--parent` for a new
training stage with the same vocabulary and a fresh optimizer. At
inference, the optional `--arithmetic` flag loads the separate certified
digit model. The evaluation metrics use the learned decoder alone.

## Continue with mathematics

Pass `--gsm-train` to add examples from the GSM8K training split and turn
on the mathematics-focused batch schedule. Eight of every twelve batches
sample arithmetic or mixed mathematics. The other four rehearse English,
Russian, logic, and Python. The arithmetic generator holds out pairs of
operands for each operation before creating their English and Russian
prompts.

```console
python -m experiments.reasoning train --parent checkpoints/shrf_tensorov_reasoning_v2/model.pt --tensorov sft_core_train.parquet --oasst 2023-04-12_oasst_ready.messages.jsonl.gz --gsm-train checkpoints/curriculum_sources/gsm_train.jsonl --gsm-test checkpoints/curriculum_sources/gsm_test.jsonl --output checkpoints/shrf_tensorov_math --steps 300 --eval-every 100 --context 192 --batch 8 --device cuda
```

The optional digit module also understands implicit coefficients, variable
names `x`, `y`, and `n`, and the bilingual instruction to return a number.
It can return exact rational solutions for linear equations. It handles the
equal-box pencil template using multiplication followed by subtraction.

## Evaluation

The text metric compares reference answer probabilities in bits per UTF-8
byte on the same held-out English and Russian prompts. The task tests
compare generated numeric and logical answers against known values. The
Python test checks functions on three inputs using a restricted AST
interpreter. The generated probes test new prompts from the same exercise
families. A held-out GSM8K slice tests open-ended word problems.

Data [Tensorov SFT Core](https://huggingface.co/datasets/tensorov/combined-reasoning-distill-cleaned),
[OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1),
and [GSM8K](https://huggingface.co/datasets/openai/gsm8k).
