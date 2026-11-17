# SHRF instruction training

The curriculum starts from a saved gated SHRF checkpoint and uses an even
rotation through arithmetic, algebra, physics, Python, English writing, and
Russian writing. It initializes a new optimizer for the new data mixture.
Whole prompt and response pairs must fit the configured byte context. Training
weights the reference number four times and the closing assistant tag one
fifth as much as ordinary answer text. Validation bits per byte cover answer
text alone.

| Input | Fields and use |
|---|---|
| `gsm_train.jsonl` | `question` and `answer` from the official GSM8K training split. |
| `gsm_test.jsonl` | The same fields from the official test split. The evaluation samples 32 eligible prompts. |
| `mbpp.jsonl` | `task_id`, `text`, and `code` from full MBPP. Only task IDs 601–974 are eligible for training. |
| `sciq.jsonl` | `question` and `answer` from a physics subset of SciQ train. Normalize the original `correct_answer` field to `answer`. |
| `databricks-dolly-15k.jsonl` | Selected context-free English writing and instruction examples. |
| `2023-04-12_oasst_ready.messages.jsonl.gz` | Optional OpenAssistant parent prompts and top-ranked replies in English and Russian. |

Place the first four files in the directory supplied as `--sources`. Supply
the last two files as `--dolly` and `--oasst`. The code also generates training
and test examples with known numerical answers. Synthetic math, algebra,
physics, and code-reading probes use an operand range outside the synthetic
training range. Normalized evaluation prompts are excluded from training.
Exact scores use greedy decoding and compare the entire numeric answer.

```console
python -m experiments.curriculum --checkpoint checkpoints/fable_shrf_text_5k/best.pt --sources checkpoints/curriculum_sources --dolly databricks-dolly-15k.jsonl --oasst 2023-04-12_oasst_ready.messages.jsonl.gz --output checkpoints/stem --steps 180 --batch 6 --device cuda
python -m experiments.code_checks checkpoints/stem/latest.pt --output checkpoints/stem/code_eval.json
python -m experiments.fable generate --checkpoint checkpoints/stem/best.pt --device cpu --prompt "Explain simple interest with an example."
```

`metrics.json` retains the baseline and validation history. `best.pt` is
selected by average validation bits per byte across domains. `latest.pt`
records the final optimizer step. The six generated function probes cover a
small Python subset and interpret supported AST expressions on known inputs.
The log reports these separately from numeric answers to Python output
questions.

Data sources [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp),
[SciQ](https://huggingface.co/datasets/allenai/sciq),
[Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
and [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1).

## Add model-generated instruction data

The optional `--teacher` JSONL accepts `domain`, `source`, `prompt`, and
`answer`. Include `number` for a numeric reference and `original_prompt`
for a paraphrased problem. The filter reserves both the evaluation wording
and its answer-only and explanation variants. Training borrows a configurable
fraction of examples from the teacher pool in each domain where that pool
has examples. Other domains continue to sample their regular pools.

```console
python -m experiments.curriculum --checkpoint checkpoints/shrf_curriculum_2/SHRF-STEM-840.pt --sources checkpoints/curriculum_sources --dolly databricks-dolly-15k.jsonl --oasst 2023-04-12_oasst_ready.messages.jsonl.gz --teacher checkpoints/curriculum_sources/teachers.jsonl --teacher-fraction 0.55 --output checkpoints/shrf_distilled --steps 240 --context 384 --batch 6 --lr 0.00045 --device cuda
```

The measured run uses viewer extracts from
[OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2),
[MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA),
[Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k),
[CodeAlpaca](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k), and
[Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4).
Some math rows use the reference answer alone when the displayed reasoning
was incomplete. Python rows have a syntax check. The metrics record how many
examples survived the filters and how many were sampled in training.

## Train the decimal arithmetic specialist

The digit cell predicts one output digit and a carry or borrow from an
operation, two input digits, and an incoming carry. It trains on all 1,300
valid decimal column transitions. At load time the runtime checks each
transition before compiling the learned predictions into a lookup table.
Signed addition, subtraction, multiplication, and long division compose
those predictions. A parser dispatches explicit integer expressions, linear
equations, and supported physics templates. Other prompts go to the SHRF
decoder. Free-text reasoning and structured arithmetic have separate
evaluations.

```console
python -m experiments.arithmetic --output checkpoints/shrf_digit_arithmetic --compare checkpoints/shrf_curriculum_2/SHRF-STEM-840.pt
python -m experiments.fable generate --checkpoint checkpoints/shrf_distilled/latest.pt --arithmetic checkpoints/shrf_digit_arithmetic/digit_cell.pt --device cpu --prompt "Solve 176x + 45 = 3917 for x."
```

The structured suite also includes literal Python expressions. It parses
allowed operations and does not execute programs. It does not cover general
word problems or Python function generation.
