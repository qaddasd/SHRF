"""Public instruction samples and reproducible school-level exercises."""

import ast
import gzip
import json
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Exercise:
    domain: str
    prompt: str
    answer: str
    source: str = "generated"
    number: str | None = None
    original_prompt: str | None = None


def generated(seed, rounds, heldout=False):
    rng = random.Random(seed)
    for i in range(rounds):
        a, b, c = (rng.randint(2, 24) for _ in range(3))
        if heldout:
            a += 24
        word = rng.choice(("books", "coins", "stickers", "pencils"))
        math = (
            (f"Find {a} + {b} - {c}.", f"{a} + {b} = {a+b}; subtract {c}.", a+b-c),
            (f"Find {a} times {b}.", f"Multiply {a} by {b}.", a*b),
            (f"There are {a} boxes of {b} {word}. Another {c} are loose. How many altogether?",
             f"{a} * {b} + {c} = {a*b} + {c}.", a*b+c),
        )[i % 3]
        x = a-14
        rhs = b*x+c
        algebra = (
            (f"Solve {b}x + {c} = {rhs} for x.", f"Subtract {c} to get {b}x = {b*x}. Divide by {b}.", x),
            (f"Find the slope through (1, {c}) and ({b+1}, {c+a*b}).",
             f"Slope = change in y / change in x = {a*b}/{b}.", a),
            (f"Find the positive root of x^2 - {a*a} = 0.", f"Take the positive square root of {a*a}.", a),
        )[i % 3]
        physics = (
            (f"A {a} kg body accelerates at {b} m/s^2. Find the net force in N.",
             f"Newton's second law gives F = m*a = {a} * {b}.", a*b),
            (f"A cyclist covers {a*b} m in {b} seconds. Find the average speed in m/s.",
             f"Average speed is distance / time = {a*b}/{b}.", a),
            (f"A {b} ohm resistor has {a*b} V across it. Find the current in A.",
             f"Ohm's law gives I = V/R = {a*b}/{b}.", a),
            (f"Find the kinetic energy in J of a {2*a} kg object moving at {b} m/s.",
             f"Kinetic energy is m*v^2/2 = {2*a}*{b}^2/2.", a*b*b),
        )[i % 4]
        code = (
            (f"What does Python print?\nprint({a} // {b})", f"Floor division gives {a//b}.", a//b),
            (f"What does Python print?\nprint(({a} + {b}) % {c})",
             f"Modulo gives the remainder when dividing {a+b} by {c}.", (a+b)%c),
            (f"What does Python print?\nprint(list(range({a},{a+b}))[{b-1}])",
             f"The first element is {a} and the index is zero-based.", a+b-1),
        )[i % 3]
        for domain, (prompt, reasoning, answer) in zip(
            ("math", "algebra", "physics", "programming"), (math, algebra, physics, code), strict=True,
        ):
            if heldout or i % 2 == 0:
                yield Exercise(domain, prompt+" Reply with only the number.", str(answer), number=str(answer))
            else:
                yield Exercise(domain, prompt+" Explain briefly.", f"{reasoning}\nAnswer: {answer}",
                               number=str(answer))
        person = rng.choice(("Alex", "Dana", "Sam", "Maya", "Lin", "Kim"))
        hour = (a % 12) + 8
        yield Exercise("writing_en", f"Ask {person} politely to send page {c} by {hour}:00.",
                       f"Hi {person}, could you please send me page {c} by {hour}:00? Thank you!")
        person = rng.choice(("Аня", "Дана", "Лена", "Марат", "Саша", "Ира"))
        yield Exercise("writing_ru", f"Вежливо попроси: {person}, страница {c} до {hour}:00.",
                       f"{person}, пришли, пожалуйста, страницу {c} до {hour}:00. Спасибо!")


def json_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for row in stream:
            if row.strip():
                yield json.loads(row)


def gsm_cases(path, heldout=False):
    for row in json_rows(path):
        rationale, sep, num = row["answer"].rpartition("####")
        num = num.strip().replace(",", "")
        if not sep or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", num):
            continue
        question = row["question"].strip()
        yield Exercise("gsm8k" if heldout else "math", question+" Reply with only the number.",
                       num, "GSM8K", num)
        if not heldout:
            rationale = re.sub(r"<<[^>]*>>", "", rationale).strip()
            yield Exercise("math", question+" Explain briefly.", f"{rationale}\nAnswer: {num}", "GSM8K")


def public_examples(directory, dolly_file):
    yield from gsm_cases(directory/"gsm_train.jsonl")
    for row in json_rows(directory/"mbpp.jsonl"):
        if 601 <= row["task_id"] <= 974:
            yield Exercise("programming", row["text"].strip(), row["code"].strip(), "MBPP")
    for row in json_rows(directory/"sciq.jsonl"):
        yield Exercise("physics", row["question"].strip(), row["answer"].strip(), "SciQ")
    for row in json_rows(dolly_file):
        if row.get("context") or row.get("category") not in {"open_qa", "general_qa", "creative_writing", "brainstorming"}:
            continue
        yield Exercise("writing_en", row["instruction"].strip(), row["response"].strip(), "Dolly")


def openassistant_examples(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = (json.loads(line) for line in stream)
        by_id = {r["message_id"]: r for r in rows
                 if r["lang"] in {"en", "ru"} and not r["deleted"] and not r.get("synthetic")}
    for row in by_id.values():
        parent = by_id.get(row.get("parent_id"))
        if (row["role"] != "assistant" or row.get("rank") != 0 or parent is None
                or parent["role"] != "prompter" or parent["lang"] != row["lang"]):
            continue
        lang = row["lang"]
        yield Exercise(f"writing_{lang}", parent["text"].strip(), row["text"].strip(),
                       f"OpenAssistant-{lang}")


def teacher_examples(path):
    for record in json_rows(path):
        row = Exercise(**record)
        if row.domain == "programming":
            # The viewer sometimes flattens code indentation. Keep parseable Python.
            try:
                ast.parse(row.answer)
            except SyntaxError:
                continue
        if row.number is not None and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", row.number):
            continue
        yield row
