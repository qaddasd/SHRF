"""Learned decimal transitions for structured, exact integer arithmetic."""

import ast
import itertools
import re
from enum import IntEnum
from fractions import Fraction
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Operation(IntEnum):
    ADD = 0
    SUBTRACT = 1
    MULTIPLY = 2


class DigitCell(nn.Module):
    """Predict the output digit and carry from one column."""

    def __init__(self, width=192):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(33, width), nn.GELU(),
                                    nn.Linear(width, width), nn.GELU(), nn.Linear(width, 20))

    def forward(self, rows):
        ops, a, b, carry = rows.unbind(-1)
        features = torch.cat((F.one_hot(ops, 3), F.one_hot(a, 10), F.one_hot(b, 10),
                              F.one_hot(carry, 10)), dim=-1)
        return self.layers(features.to(self.layers[0].weight.dtype)).unflatten(-1, (2, 10))


def transition_examples():
    """Enumerate all valid single-column inputs and their reference outputs."""
    rows, answers = [], []
    for op, carries in ((Operation.ADD, 2), (Operation.SUBTRACT, 2), (Operation.MULTIPLY, 9)):
        for a, b, carry in itertools.product(range(10), range(10), range(carries)):
            rows.append((int(op), a, b, carry))
            if op == Operation.SUBTRACT:
                value = a-b-carry
                answers.append((value % 10, int(value < 0)))
            else:
                value = a+b+carry if op == Operation.ADD else a*b+carry
                answers.append((value % 10, value // 10))
    return torch.tensor(rows), torch.tensor(answers)


def digits(value):
    return tuple(map(int, reversed(str(value))))


def number(columns):
    return int("".join(str(digit) for digit in reversed(columns)))


class DecimalArithmetic:
    """Use a certified neural transition table for multi-digit composition."""

    @torch.inference_mode()
    def __init__(self, cell):
        rows, answers = transition_examples()
        predictions = cell(rows.to(next(cell.parameters()).device)).argmax(-1).cpu()
        if not torch.equal(predictions, answers):
            errors = (~(predictions == answers).all(-1)).sum().item()
            raise ValueError(f"Learned digit cell failed certification on {errors} transitions")
        self.transitions = dict(zip(map(tuple, rows.tolist()), map(tuple, predictions.tolist()),
                                    strict=True))

    @classmethod
    def load(cls, path: Path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        cell = DigitCell(saved["width"])
        cell.load_state_dict(saved["model"])
        return cls(cell.eval())

    def _columns(self, op, a, b):
        carry, out = 0, []
        for left, right in itertools.zip_longest(a, b, fillvalue=0):
            digit, carry = self.transitions[op, left, right, carry]
            out.append(digit)
        if op != Operation.SUBTRACT and carry:
            out.append(carry)
        return number(out)

    def add(self, a, b):
        if (a < 0) != (b < 0):
            largest, smallest = sorted((a, b), key=abs, reverse=True)
            result = self._columns(Operation.SUBTRACT, digits(abs(largest)), digits(abs(smallest)))
            return -result if largest < 0 else result
        result = self._columns(Operation.ADD, digits(abs(a)), digits(abs(b)))
        return -result if a < 0 else result

    def subtract(self, a, b):
        return self.add(a, -b)

    def multiply(self, a, b):
        total = 0
        left = digits(abs(a))
        for offset, right in enumerate(digits(abs(b))):
            if right == 0:
                continue
            out, carry = [], 0
            for value in left:
                digit, carry = self.transitions[Operation.MULTIPLY, value, right, carry]
                out.append(digit)
            out.append(carry)
            partial = number([0]*offset + out)
            total = self.add(total, partial)
        return -total if (a < 0) != (b < 0) else total

    def divmod(self, a, b):
        if b == 0:
            raise ZeroDivisionError("division by zero")
        divisor = abs(b)
        remainder, quotient = 0, []
        for digit in str(abs(a)):
            remainder = int(f"{remainder}{digit}")
            count = 0
            while remainder >= divisor:
                remainder = self.subtract(remainder, divisor)
                count += 1
            quotient.append(str(count))
        result = int("".join(quotient))
        if (a < 0) != (b < 0):
            result = -self.add(result, int(remainder != 0))
            if remainder:
                remainder = self.subtract(divisor, remainder)
        return result, -remainder if b < 0 else remainder

    def exact_divide(self, a, b):
        result, remainder = self.divmod(a, b)
        if remainder:
            raise ValueError("This solver returns only exact integer quotients")
        return result

    def eval_expression(self, expression):
        if len(expression) > 160 or not re.fullmatch(r"[0-9+*/%() \t-]+", expression):
            raise ValueError("Expected a bounded integer arithmetic expression")
        root = ast.parse(expression.strip(), mode="eval")
        operators = {
            ast.Add: self.add, ast.Sub: self.subtract, ast.Mult: self.multiply,
            ast.Div: self.exact_divide, ast.FloorDiv: lambda a, b: self.divmod(a, b)[0],
            ast.Mod: lambda a, b: self.divmod(a, b)[1],
        }

        def walk(node):
            match node:
                case ast.Constant(value=value) if type(value) is int:
                    result = value
                case ast.UnaryOp(op=ast.USub(), operand=operand):
                    result = -walk(operand)
                case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                    result = walk(operand)
                case ast.BinOp(left=a, op=operator, right=b) if type(operator) in operators:
                    result = operators[type(operator)](walk(a), walk(b))
                case _:
                    raise ValueError("Unsupported arithmetic operation")
            if len(str(abs(result))) > 64:
                raise ValueError("Arithmetic exceeds the 64-digit limit")
            return result

        return walk(root.body)


INTEGER = r"[+-]?[0-9]{1,64}"
NATURAL = r"[0-9]{1,30}"
LINEAR = re.compile(
    rf"(?:(?:solve|реши(?:те)?)\s+)?(?P<a>[+-]?(?:[0-9]{{1,64}}\s*\*?)?)\s*"
    rf"(?P<variable>[xхyуn])(?:\s*(?P<sign>[+-])\s*(?P<b>{NATURAL}))?"
    rf"\s*=\s*(?P<c>{INTEGER})(?:\s+(?:for|для)\s+(?P=variable))?",
    re.I,
)
ANSWER_INSTRUCTION = re.compile(
    r"\s+(?:reply with only the number|give a number|answer only with a number|"
    r"ответ\s*[-—:]\s*число|ответ только число)\.?$", re.I,
)
BOXES = (
    re.compile(rf"there are ({NATURAL}) boxes with ({NATURAL}) pencils each\. "
               rf"({NATURAL}) were given away\. how many remain", re.I),
    re.compile(rf"в ({NATURAL}) коробках по ({NATURAL}) карандашей\. "
               rf"отдали ({NATURAL})\. сколько осталось", re.I),
)
PHYSICS = (
    (re.compile(rf"a ({NATURAL}) kg body accelerates at ({NATURAL}) m/s\^2\. find the net force in N", re.I),
     "force"),
    (re.compile(rf"a cyclist covers ({NATURAL}) m in ({NATURAL}) seconds\. find the average speed in m/s", re.I),
     "divide"),
    (re.compile(rf"a ({NATURAL}) ohm resistor has ({NATURAL}) V across it\. find the current in A", re.I),
     "current"),
    (re.compile(rf"find the kinetic energy in J of a ({NATURAL}) kg object moving at ({NATURAL}) m/s", re.I),
     "kinetic"),
    (re.compile(rf"найди(?:те)? силу при массе ({NATURAL}) кг и ускорении ({NATURAL}) м/с(?:²|\^2)", re.I),
     "force"),
    (re.compile(rf"найди(?:те)? скорость при пути ({NATURAL}) м за ({NATURAL}) с", re.I),
     "divide"),
)


def structured_answer(engine, prompt):
    """Return an answer for the supported prompt forms, or None for language generation."""
    prompt = ANSWER_INSTRUCTION.sub("", prompt.strip()).rstrip("?.").strip()
    prompt = prompt.replace("−", "-")
    try:
        if match := LINEAR.fullmatch(prompt):
            coefficient = match["a"].rstrip("*").strip()
            a = int(coefficient+"1") if coefficient in ("", "+", "-") else int(coefficient)
            b = int(match["sign"]+match["b"]) if match["b"] else 0
            numerator = engine.subtract(int(match["c"]), b)
            quotient, remainder = engine.divmod(numerator, a)
            return str(Fraction(numerator, a)) if remainder else str(quotient)
        for pattern in BOXES:
            if match := pattern.fullmatch(prompt):
                a, b, taken = map(int, match.groups())
                remaining = engine.subtract(engine.multiply(a, b), taken)
                return str(remaining) if remaining >= 0 else None
        for pattern, formula in PHYSICS:
            if match := pattern.fullmatch(prompt):
                a, b = map(int, match.groups())
                if formula == "force":
                    return str(engine.multiply(a, b))
                if formula == "current":
                    return str(engine.exact_divide(b, a))
                if formula == "divide":
                    return str(engine.exact_divide(a, b))
                energy = engine.multiply(a, engine.multiply(b, b))
                return str(engine.exact_divide(energy, 2))
        if match := re.fullmatch(r"what does python print\?\s*print\((.*)\)", prompt, flags=re.I):
            expression = match[1]
            if re.search(r"(?<!/)/(?!/)", expression):
                return None
            return str(engine.eval_expression(expression))
        expression = re.sub(r"^(?:what is|compute|find|найди(?:те)?|вычисли(?:те)?|сколько будет)\s+",
                            "", prompt, flags=re.I)
        expression = expression.replace("×", "*").replace("÷", "/")
        expression = re.sub(r"\s+times\s+", "*", expression, flags=re.I)
        return str(engine.eval_expression(expression))
    except (SyntaxError, ValueError, ArithmeticError):
        return None
