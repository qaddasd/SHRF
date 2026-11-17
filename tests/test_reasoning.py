import json

from experiments.fable import IGNORE
from experiments.reasoning import encode, make_batch, passed
from experiments.reasoning_data import ReasoningCase, arithmetic_cases, unpack_tensorov
from experiments.reasoning_tokenizer import END, FINAL, WORK, ReasoningTokenizer


def test_tokenizer_round_trip_and_bilingual_compression():
    sentence = "Проверим русский текст и English text.\nЦифры 2026."
    tokenizer = ReasoningTokenizer.fit([sentence]*3, 420)
    tokens = tokenizer.encode(sentence)
    assert tokenizer.decode(tokens) == sentence
    assert len(tokens) < len(sentence.encode())
    assert tokenizer.decode(tokenizer.encode("Unseen 👋 юникод")) == "Unseen 👋 юникод"
    assert tokenizer.encode("123") == list(b"123")


def test_reasoning_and_answer_labels_at_correct_offsets():
    tokenizer = ReasoningTokenizer([])
    case = ReasoningCase("math", "3*8?", "24", "Multiply 3 by 8.")
    sample = encode(case, tokenizer, 100)
    ids, labels, weights, valid = make_batch([sample], "cpu")
    offset = len(tokenizer.prompt(case.prompt))-1
    assert (labels[0, :offset] == IGNORE).all()
    assert (weights[0, :offset] == 0).all()
    assert labels[0, offset] == WORK
    final_index = sample[1].index(FINAL)
    assert labels[0, final_index+1] == ord("2")
    assert weights[0, final_index+1] == 3
    assert labels[0, -1] == END
    assert ids.shape == valid.shape


def test_incomplete_tensorov_answer_does_not_learn_false_end():
    tokenizer = ReasoningTokenizer([])
    messages = [{"role": "user", "content": "Explain Rust lifetimes."},
                {"role": "assistant", "content": "Long answer. " * 18}]
    case = unpack_tensorov({"messages_json": json.dumps(messages), "domain": "rust"})
    assert case.domain == "code"
    assert encode(case, tokenizer, 80) is None
    ids, labels, weights = encode(case, tokenizer, 80, allow_prefix=True)
    assert len(ids) == len(labels) == len(weights) == 80
    assert END not in labels
    assert unpack_tensorov({"messages_json": "{invalid"}) is None


def test_code_probe_verifies_behavior_with_no_execution():
    case = ReasoningCase("code", "Write a linear function.", "def solve(x):\n    return 3 * x - 7")
    assert passed(case, "def solve(y):\n    return y*3-7")
    assert not passed(case, "def solve(y):\n    return y*3-8")


def test_arithmetic_splits_hold_out_operands_in_both_languages():
    train = {case.prompt.split(" ", 1)[1] for case in arithmetic_cases(62)}
    heldout = list(arithmetic_cases(62, heldout=True))
    assert train.isdisjoint(case.prompt.split(" ", 1)[1] for case in heldout)
    for english, russian in zip(heldout[::2], heldout[1::2], strict=True):
        assert english.prompt.split(" ", 1)[1] == russian.prompt.split(" ", 1)[1]
        assert english.answer == russian.answer
