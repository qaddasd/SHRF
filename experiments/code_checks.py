"""Evaluate small generated Python functions with a restricted AST interpreter."""

import argparse
import ast
import json
import operator
from pathlib import Path

import torch

from .curriculum import greedy, save_json
from .language import TinyDecoder

OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
FUNCTIONS = {"abs": abs, "len": len, "sum": sum, "min": min, "max": max}
CASES = (
    ("solve(x)", "return triple x minus one", ((-7,), (0,), (12,)), lambda x: 3*x-1),
    ("solve(x)", "return one more than the absolute value of x", ((-2,), (0,), (9,)), lambda x: abs(x)+1),
    ("solve(x, y)", "return the perimeter of an x by y rectangle", ((2, 9), (1, 1), (6, 3)),
     lambda x, y: 2*(x+y)),
    ("solve(x, y)", "return the remainder after dividing x by y", ((7, 3), (20, 7), (-4, 3)),
     lambda x, y: x % y),
    ("solve(xs)", "return the sum of the values in xs minus its length", (([],), ([8, 1],), ([-3, 7, 2],)),
     lambda xs: sum(xs)-len(xs)),
    ("solve(xs)", "return the difference between the largest and smallest element of xs",
     (([0],), ([4, 1, 6],), ([-8, 1, -2],)), lambda xs: max(xs)-min(xs)),
)


def interpret(expression, bindings):
    match expression:
        case ast.Name(id=name) if name in bindings:
            return bindings[name]
        case ast.Constant(value=value) if type(value) is int and abs(value) < 10000:
            return value
        case ast.UnaryOp(op=ast.USub(), operand=value):
            operand = interpret(value, bindings)
            if type(operand) is not int:
                raise ValueError("Unary arithmetic requires integer operands")
            return -operand
        case ast.BinOp(left=left, op=op, right=right) if type(op) in OPERATORS:
            a, b = interpret(left, bindings), interpret(right, bindings)
            if type(a) is not int or type(b) is not int:
                raise ValueError("Binary arithmetic requires integer operands")
            return OPERATORS[type(op)](a, b)
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in FUNCTIONS:
            return FUNCTIONS[name](*(interpret(arg, bindings) for arg in args))
        case _:
            raise ValueError("Expression is outside the supported Python subset")


def check(program, test_inputs, expected):
    lines = program.strip().splitlines()
    if len(lines) > 2 and lines[0].strip() in {"```", "```python"} and lines[-1].strip() == "```":
        program = "\n".join(lines[1:-1])
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return False, False
    match tree.body:
        case [ast.FunctionDef(name="solve", args=signature, body=[ast.Return(value=expression)],
                              decorator_list=[], returns=None)]:
            names = [arg.arg for arg in signature.args]
            if (signature.posonlyargs or signature.kwonlyargs or signature.vararg or signature.kwarg
                    or signature.defaults or len(names) != len(test_inputs[0])):
                return True, False
            try:
                return True, all(
                    interpret(expression, dict(zip(names, values, strict=True))) == expected(*values)
                    for values in test_inputs
                )
            except (ValueError, TypeError, ArithmeticError):
                return True, False
        case _:
            have_function = any(isinstance(node, ast.FunctionDef) for node in tree.body)
            return have_function, False


def run(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    architecture = saved["metadata"]["architecture"]
    model = TinyDecoder(**architecture).to(device).eval()
    model.load_state_dict(saved["model"])
    results = []
    for signature, instruction, inputs, expected in CASES:
        prompt = f"Write a Python function {signature} to {instruction}. Output Python code only."
        output = greedy(model, prompt, device, architecture["context"], 128)
        syntax, correct = check(output, inputs, expected)
        results.append({"prompt": prompt, "syntax": syntax, "passes": correct})
    return {"step": saved["step"], "syntax": sum(row["syntax"] for row in results),
            "passes": sum(row["passes"] for row in results), "total": len(CASES), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = [run(path, args.device) for path in args.checkpoints]
    save_json(args.output, report)
    print(json.dumps([{k: v for k, v in row.items() if k != "cases"} for row in report]))


if __name__ == "__main__":
    main()
