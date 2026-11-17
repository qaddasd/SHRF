import pytest

from shrf.arithmetic import (
    DecimalArithmetic,
    DigitCell,
    structured_answer,
    transition_examples,
)


@pytest.fixture
def engine():
    rows, labels = transition_examples()
    arithmetic = DecimalArithmetic.__new__(DecimalArithmetic)
    arithmetic.transitions = dict(zip(map(tuple, rows.tolist()), map(tuple, labels.tolist()),
                                      strict=True))
    return arithmetic


def test_composed_digit_operations(engine):
    assert engine.add(999999, 1) == 1000000
    assert engine.subtract(-3, 10) == -13
    assert engine.subtract(1000000, 1) == 999999
    assert engine.multiply(-5001, 70003) == -350085003
    for a, b in [(-212342, 71), (212342, -71), (-212342, -71), (0, 71), (10, 2)]:
        assert engine.divmod(a, b) == divmod(a, b)


def test_specialist_routes_only_supported_forms(engine):
    assert structured_answer(engine, "Find 39 times 15.") == "585"
    assert structured_answer(engine, "Solve 76x - 18 = -246 for x.") == "-3"
    assert structured_answer(engine, "What does Python print?\nprint(-10 // 3)") == "-4"
    assert structured_answer(engine, "What does Python print?\nprint(4 / 2)") is None
    assert structured_answer(engine, "Compute (3 - 10) * 8.") == "-56"
    assert structured_answer(engine, "Реши 17х + 15 = 100.") == "5"
    assert structured_answer(engine, "Найди силу при массе 14 кг и ускорении 3 м/с².") == "42"
    assert structured_answer(engine, "Сколько будет (3+5)×7?") == "56"
    assert structured_answer(engine, "Compute 5 / 2.") is None
    assert structured_answer(engine, "Who invented physics?") is None
    assert structured_answer(engine, "print(__import__('os').getcwd())") is None
    assert structured_answer(engine, "Compute 1 / 0.") is None


def test_arithmetic_handles_reasoning_prompt_variants(engine):
    assert structured_answer(engine, "Solve 7x + 11 = 74. Give a number.") == "9"
    assert structured_answer(engine, "Реши 2y − 1 = 4. Ответ — число.") == "5/2"
    assert structured_answer(engine, "Solve -x = -9 for x.") == "9"
    assert structured_answer(engine, "Solve 3n = 21 for n.") == "7"
    assert structured_answer(engine, "Solve 0x + 3 = 3.") is None
    assert structured_answer(engine, "Solve 2y = 6 for x.") is None
    assert structured_answer(engine, "В 8 коробках по 2 карандашей. Отдали 5. Сколько осталось?") == "11"
    assert structured_answer(
        engine, "There are 3 boxes with 5 pencils each. 10 were given away. How many remain?",
    ) == "5"


def test_decoding_requires_exact_transition_certification():
    cell = DigitCell(8)
    for parameter in cell.parameters():
        parameter.data.zero_()
    with pytest.raises(ValueError, match="failed certification"):
        DecimalArithmetic(cell)
