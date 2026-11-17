import torch

from experiments.code_checks import check
from experiments.curriculum import batch_cases, encode, numeric_equal, prompt_bytes
from experiments.curriculum_data import Exercise
from experiments.fable import ASSISTANT_END, IGNORE


def test_curriculum_masks_prompt_padding_and_evaluation_end_tag():
    row = Exercise("math", "What is 3+4?", "7", number="7")
    long_row = Exercise("writing_ru", "Напиши приветствие.", "Привет!")
    encoded = [encode(example, 160) for example in (row, long_row)]
    inputs, labels, valid, weights = batch_cases(encoded, "cpu")
    start = len(prompt_bytes(row.prompt)) - 1
    assert labels[0, start] == ord("7")
    assert weights[0, start] == 4
    assert (labels[0, :start] == IGNORE).all()
    assert torch.all(labels[valid == 0] == IGNORE)
    assert torch.all(weights[valid == 0] == 0)
    assert torch.equal(inputs[0, start+1:start+2], labels[0, start:start+1])
    _, validation, _, _ = batch_cases(encoded, "cpu", evaluation=True)
    assert (validation[0, start+1:start+1+len(ASSISTANT_END)] == IGNORE).all()
    assert numeric_equal("Answer: -4.00", "-4")
    assert not numeric_equal("Maybe -4, or 5", "-4")


def test_simple_python_functions_can_be_checked_without_execution():
    assert check("def solve(x):\n    return 1 + abs(x)", ((-2,), (4,)), lambda x: abs(x)+1) == (
        True, True,
    )
    assert check("def solve(xs):\n    return xs * 9999", (([1],),), len) == (True, False)
    assert check("42", ((1,),), abs) == (False, False)
