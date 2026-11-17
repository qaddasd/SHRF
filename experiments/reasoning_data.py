"""Tensorov instruction pairs and verified short reasoning examples."""

import json
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningCase:
    domain: str
    prompt: str
    answer: str
    work: str = ""
    source: str = "verified"


def domain_for(prompt, answer, label):
    text = prompt.casefold()
    if (re.search(r"rust|python|sql|typescript|react|code", str(label), re.I)
            or re.search(r"\b(python|rust|javascript|sql|function|код|функцию)\b", text)
            or "```" in answer):
        return "code"
    if re.search(r"\b(xor|proposition|truth table|implication|логик\w*|импликац\w*)\b", text):
        return "logic"
    if "math" in str(label).casefold() or re.search(
        r"\b(equation|fraction|integral|solve for|derivative|реши|уравнение)\b", text,
    ):
        return "math"
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", prompt))
    return "ru" if cyrillic >= 4 else "en"


def unpack_tensorov(row):
    messages = row.get("messages")
    if messages is None and isinstance(row.get("messages_json"), str):
        try:
            messages = json.loads(row["messages_json"])
        except json.JSONDecodeError:
            return None
    if not isinstance(messages, list):
        return None
    for index, message in enumerate(messages[:-1]):
        following = messages[index+1]
        if (not isinstance(message, dict) or not isinstance(following, dict)
                or message.get("role") != "user" or following.get("role") != "assistant"):
            continue
        prompt, response = message.get("content"), following.get("content")
        if not isinstance(prompt, str) or not isinstance(response, str):
            continue
        prompt, response = prompt.strip(), response.strip()
        trace = re.match(r"\s*<think>\s*(.*?)\s*</think>\s*(.*)$", response, re.S)
        work, answer = trace.groups() if trace else ("", response)
        if 6 <= len(prompt) <= 900 and 1 <= len(answer) <= 12000 and len(work) <= 850:
            return ReasoningCase(domain_for(prompt, answer, row.get("domain")), prompt, answer,
                                 work, "tensorov/combined-reasoning-distill-cleaned")
    return None


def tensorov_cases(path, seed, limit=7500, per_domain=2100, scan_batches=220):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Tensorov Parquet requires pyarrow") from exc
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    message_column = "messages" if "messages" in names else "messages_json"
    if message_column not in names:
        raise ValueError("Tensorov Parquet must contain messages or messages_json")
    columns = [key for key in (message_column, "domain") if key in names]
    indices = list(range(parquet.num_row_groups))
    random.Random(seed).shuffle(indices)
    cases, counts = [], {}
    for step, batch in enumerate(parquet.iter_batches(batch_size=512, row_groups=indices, columns=columns)):
        for row in batch.to_pylist():
            case = unpack_tensorov(row)
            if case is None or counts.get(case.domain, 0) >= per_domain:
                continue
            cases.append(case)
            counts[case.domain] = counts.get(case.domain, 0)+1
        if step+1 >= scan_batches or len(cases) >= limit:
            break
    return cases[:limit]


def arithmetic_cases(seed, *, heldout=False):
    """Yield bilingual problems split by operation and operands."""
    split_rng = random.Random(seed)
    for operator in ("+", "-", "*", "/"):
        values = range(1, 11) if operator in ("*", "/") else range(0, 17)
        pairs = [(a, b) for a in values for b in values
                 if operator not in ("+", "*") or a <= b]
        split_rng.shuffle(pairs)
        split = len(pairs) // 8
        selected = pairs[:split] if heldout else pairs[split:]
        for a, b in selected:
            if operator == "+":
                response = a+b
            elif operator == "-":
                response = a-b
            elif operator == "*":
                response = a*b
            else:
                a, response = a*b, a
            expression = f"{a} {operator} {b}"
            for verb in ("Compute", "Вычисли"):
                yield ReasoningCase("math", f"{verb} {expression}.", str(response), source="verified-arithmetic")


def verified_cases(seed, count):
    rng = random.Random(seed)
    for index in range(count):
        ru = index % 2 == 0
        a, b, c = rng.sample(range(3, 40), 3)
        work = index % 3 != 0
        name = rng.choice(("x", "y", "n"))
        if index % 4 < 2:
            prompt = (f"Реши {a}{name} + {b} = {a*c+b}. Ответ — число." if ru
                      else f"Solve {a}{name} + {b} = {a*c+b}. Give a number.")
            trace = (f"Вычтем {b}: {a}{name} = {a*c}. Делим на {a}." if ru
                     else f"Subtract {b}: {a}{name} = {a*c}. Divide by {a}.")
            number = c
        else:
            prompt = (f"В {a} коробках по {b} карандашей. Отдали {c}. Сколько осталось?" if ru
                      else f"There are {a} boxes with {b} pencils each. {c} were given away. How many remain?")
            trace = (f"Всего {a} × {b} = {a*b}. Вычитаем {c}." if ru
                     else f"Start with {a} × {b} = {a*b}. Subtract {c}.")
            number = a*b-c
        yield ReasoningCase("math", prompt, str(number), trace if work else "")

        p, q = rng.sample(("A", "B", "C", "D", "E", "F", "G", "H"), 2)
        x, y = rng.randrange(2), rng.randrange(2)
        op = rng.choice(("AND", "OR", "XOR"))
        result = {"AND": x & y, "OR": x | y, "XOR": x ^ y}[op]
        order = rng.sample(("Alice", "Dani", "Amina", "Sam", "Robin", "Mila", "Mel", "Kai"), 3)
        if index % 5 == 0:
            first, second, third = order
            prompt = (f"{first} выше {second}, а {second} выше {third}. Кто самый низкий?" if ru
                      else f"{first} is taller than {second}. {second} is taller than {third}. Who is shortest?")
            trace = f"{first} > {second} > {third}."
            answer = third
        else:
            prompt = (f"{p} = {x}, {q} = {y}. Найди {p} {op} {q}. Ответ только 0 или 1." if ru
                      else f"{p} = {x}, {q} = {y}. Compute {p} {op} {q}. Answer only 0 or 1.")
            trace = f"{x} {op} {y} = {result}."
            answer = str(result)
        yield ReasoningCase("logic", prompt, answer, trace if work else "")

        prompt = (f"Напиши функцию Python solve(x), которая возвращает {a}*x - {b}. Только код." if ru
                  else f"Write a Python function solve(x) that returns {a}*x - {b}. Code only.")
        answer = f"def solve(x):\n    return {a} * x - {b}"
        yield ReasoningCase("code", prompt, answer)
