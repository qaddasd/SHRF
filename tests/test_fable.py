import hashlib
import json
from argparse import Namespace

import torch

from experiments.fable import IGNORE, prepare_corpora, render_trace, validation_batches


def test_fable_supervises_assistant_text_only():
    trace, spans = render_trace([
        {"role": "user", "content": "a question"},
        {"role": "assistant", "content": "some text", "tool_calls": [{"name": "search"}]},
        {"role": "tool", "content": "a search result"},
        {"role": "assistant", "content": "the answer"},
    ])
    assert b"<tool_calls>" in trace
    assert b"a search result" in trace
    assert [trace[start:stop] for start, stop in spans] == [
        b"some text", b"the answer\n</assistant>\n",
    ]


def test_fable_validation_excludes_verbatim_training_continuations(tmp_path):
    source = tmp_path / "data.jsonl"
    rows = []
    for index in range(128):
        suffix = hashlib.sha256(str(index).encode()).hexdigest()
        rows.append(json.dumps({"messages": [
            {"role": "user", "content": "A shared question"},
            {"role": "assistant", "content": f"A shared answer: {suffix}"},
        ]}))
    source.write_text("\n".join([*rows, rows[0]]) + "\n", encoding="utf-8")
    train, heldout = prepare_corpora(source, tmp_path, 32)[0]
    try:
        assert train.metadata["traces"] + heldout.metadata["traces"] == len(rows)
        args = Namespace(seed=45, context=32, batch=2, eval_batches=3, device="cpu")
        batches, protocol = validation_batches(train, heldout, args)
        train_bytes = train.path.read_bytes()
        continuations = []
        length = protocol["continuation_bytes"]
        for ids, targets in batches:
            assert ids.shape == targets.shape == (2, 32)
            assert torch.all(targets[:, :-length] == IGNORE)
            assert torch.equal(ids[:, -length + 1:], targets[:, -length:-1])
            continuations.extend(bytes(row) for row in targets[:, -length:].tolist())
        assert len(set(continuations)) == protocol["windows"] == 6
        assert all(continuation not in train_bytes for continuation in continuations)
    finally:
        del train.data, heldout.data
