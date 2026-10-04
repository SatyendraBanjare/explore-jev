"""Benchmark: Jev vs. a direct Claude Sonnet 5.5 call, on the three primitives in example.py.

Jev answers one typed question (Noul, Choice, or Score) in a single cheap call.
Claude is a general chat model asked to return the same judgment as JSON. Each
primitive is run over a list of (input, expected) cases so the benchmark also
reports accuracy, not just latency and cost, for the *same job* on *both*
providers.

Install: pip install typesafe-sdk anthropic
Auth:    export TYPESAFE_API_KEY=...   and   export ANTHROPIC_API_KEY=...

Tasks:   by default, every *.json file in benchmark_test_cases/ is loaded and run.
         Pass --suite <folder> to run all JSON files in a different folder, or
         --file <path> to run a single JSON file. Each file is shaped as
         {
           "noul": {
             "question_key": "is_injection",
             "instructions": "...",
             "criteria": {"true": "...", "false": "..."},
             "claude_question": "...",
             "cases": [{"input": ..., "expected": ...}, ...]
           },
           "choice": {..., "criteria": {"option": "description", ...}, "cases": [...]},
           "score": {..., "criteria": ["level 0 description", "level 1 description", ...], "cases": [...]}
         }

Output:  everything printed is also saved to benchmark_results.txt (override with
         --output <path>).

Run: python benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import anthropic
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()  # reads TYPESAFE_API_KEY and ANTHROPIC_API_KEY from a .env file, if present

DEFAULT_CASES_PATH = Path(__file__).with_name("benchmark_test_cases")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("benchmark_results.txt")

CLAUDE_MODEL = "claude-sonnet-5-5"
CLAUDE_INPUT_PRICE_PER_TOKEN = 2.00 / 1_000_000
CLAUDE_OUTPUT_PRICE_PER_TOKEN = 10.00 / 1_000_000
JEV_INPUT_PRICE_PER_TOKEN = 0.000000042

# A provider call function takes (client, case_input, spec) and returns
# (answer, latency_ms, input_tokens, output_tokens, cost_usd).
CallFn = Callable[[Any, Any, dict], tuple[dict, float, int, int, float]]
CorrectFn = Callable[[dict, Any, dict], bool]


@dataclass
class Task:
    name: str
    source: str
    spec: dict[str, Any]
    cases: list[dict[str, Any]]
    jev_fn: CallFn
    claude_fn: CallFn
    correct_fn: CorrectFn


@dataclass
class Result:
    provider: str
    task: str
    case_input: Any
    expected: Any
    correct: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    answer: Any


class _Tee:
    """Writes to several streams at once, so console output is also saved to a file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _jev_cost(usage: Any) -> float:
    return usage.input_tokens * JEV_INPUT_PRICE_PER_TOKEN


def _claude_json(response: anthropic.types.Message) -> dict:
    """Claude sometimes wraps JSON in a code fence despite instructions not to."""
    text = next(b.text for b in response.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    return json.loads(text)


def _claude_call(client: anthropic.Anthropic, prompt: str) -> tuple[dict, float, int, int, float]:
    start = time.perf_counter()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = _claude_json(response)
    cost = (
        response.usage.input_tokens * CLAUDE_INPUT_PRICE_PER_TOKEN
        + response.usage.output_tokens * CLAUDE_OUTPUT_PRICE_PER_TOKEN
    )
    return answer, latency_ms, response.usage.input_tokens, response.usage.output_tokens, cost


# ---------------------------------------------------------------------------
# Scenario 1: Noul -- a yes/no probability
# ---------------------------------------------------------------------------

def jev_noul(client: TypeSafeClient, message: str, spec: dict) -> tuple[dict, float, int, int, float]:
    key = spec["question_key"]
    start = time.perf_counter()
    response = client.system_one(
        state=message,
        questions={key: Noul(instructions=spec["instructions"], criteria=spec["criteria"])},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = response.answers[key]
    result = {key: answer.noul > 0.5, "noul": round(answer.noul, 2)}
    return result, latency_ms, response.usage.input_tokens, response.usage.output_tokens, _jev_cost(response.usage)


def claude_noul(client: anthropic.Anthropic, message: str, spec: dict) -> tuple[dict, float, int, int, float]:
    key = spec["question_key"]
    prompt = (
        f"Message: {json.dumps(message)}\n\n"
        f"{spec['claude_question']} Reply with ONLY a JSON object of the exact shape "
        f'{{"{key}": true|false, "probability": <0..1 float>}}. No prose, no markdown.'
    )
    return _claude_call(client, prompt)


def noul_correct(answer: dict, expected: bool, spec: dict) -> bool:
    return answer[spec["question_key"]] == expected


# ---------------------------------------------------------------------------
# Scenario 2: Choice -- pick one of the declared options
# ---------------------------------------------------------------------------

def jev_choice(client: TypeSafeClient, ticket: str, spec: dict) -> tuple[dict, float, int, int, float]:
    key = spec["question_key"]
    start = time.perf_counter()
    response = client.system_one(
        state={"ticket": ticket},
        questions={key: Choice(instructions=spec["instructions"], criteria=spec["criteria"])},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = response.answers[key]
    result = {key: answer.choice, "confidence": round(answer.confidence, 2)}
    return result, latency_ms, response.usage.input_tokens, response.usage.output_tokens, _jev_cost(response.usage)


def claude_choice(client: anthropic.Anthropic, ticket: str, spec: dict) -> tuple[dict, float, int, int, float]:
    key = spec["question_key"]
    options = "\n".join(f'- "{k}": {v}' for k, v in spec["criteria"].items())
    prompt = (
        f"Ticket: {json.dumps(ticket)}\n\n"
        f"{spec['claude_question']} Options:\n{options}\n\n"
        f'Reply with ONLY a JSON object of the exact shape {{"{key}": <one option key>, '
        '"confidence": <0..1 float>}. No prose, no markdown.'
    )
    return _claude_call(client, prompt)


def choice_correct(answer: dict, expected: str, spec: dict) -> bool:
    return answer[spec["question_key"]] == expected


# ---------------------------------------------------------------------------
# Scenario 3: Score -- a position on an ordered scale
# ---------------------------------------------------------------------------

def jev_score(client: TypeSafeClient, ticket: str, spec: dict) -> tuple[dict, float, int, int, float]:
    key = spec["question_key"]
    start = time.perf_counter()
    response = client.system_one(
        state={"ticket": ticket},
        questions={key: Score(instructions=spec["instructions"], criteria=spec["criteria"])},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = response.answers[key]
    result = {key: round(answer.score, 2)}
    return result, latency_ms, response.usage.input_tokens, response.usage.output_tokens, _jev_cost(response.usage)


def claude_score(client: anthropic.Anthropic, ticket: str, spec: dict) -> tuple[dict, float, int, int, float]:
    key = spec["question_key"]
    levels = "\n".join(f"{i}: {level}" for i, level in enumerate(spec["criteria"]))
    max_level = len(spec["criteria"]) - 1
    prompt = (
        f"Ticket: {json.dumps(ticket)}\n\n"
        f"{spec['claude_question']} Levels:\n{levels}\n\n"
        "Reply with ONLY a JSON object of the exact shape "
        f'{{"{key}": <float between 0 and {max_level}, can be fractional>}}. No prose, no markdown.'
    )
    return _claude_call(client, prompt)


def score_correct(answer: dict, expected: float, spec: dict) -> bool:
    return abs(answer[spec["question_key"]] - expected) <= 0.5


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# task -> (jev_fn, claude_fn, correct_fn)
TASK_FNS: dict[str, tuple[CallFn, CallFn, CorrectFn]] = {
    "noul": (jev_noul, claude_noul, noul_correct),
    "choice": (jev_choice, claude_choice, choice_correct),
    "score": (jev_score, claude_score, score_correct),
}


def _tasks_from_def(
    name: str, source: str, task_def: dict, jev_fn: CallFn, claude_fn: CallFn, correct_fn: CorrectFn,
) -> Task | None:
    if not task_def or not task_def.get("cases"):
        return None
    spec = {k: v for k, v in task_def.items() if k != "cases"}
    task_name = f"{name}:{task_def['question_key']}" if "question_key" in task_def else name
    return Task(task_name, source, spec, task_def["cases"], jev_fn, claude_fn, correct_fn)


def load_tasks_from_file(cases_file: Path) -> list[Task]:
    raw = json.loads(cases_file.read_text())
    tasks = []
    for name, (jev_fn, claude_fn, correct_fn) in TASK_FNS.items():
        task_defs = raw.get(name)
        if not task_defs:
            continue
        # A primitive key can hold either a single task def or a list of them.
        for task_def in (task_defs if isinstance(task_defs, list) else [task_defs]):
            task = _tasks_from_def(name, cases_file.name, task_def, jev_fn, claude_fn, correct_fn)
            if task:
                tasks.append(task)
    return tasks


def load_tasks(cases_path: Path) -> list[Task]:
    if cases_path.is_dir():
        tasks = []
        for cases_file in sorted(cases_path.glob("*.json")):
            tasks.extend(load_tasks_from_file(cases_file))
        return tasks
    return load_tasks_from_file(cases_path)


def run_case(fn: CallFn, correct_fn: CorrectFn, client: Any, provider: str, task: Task, case: dict[str, Any]) -> Result:
    try:
        answer, latency_ms, in_tok, out_tok, cost = fn(client, case["input"], task.spec)
    except Exception as exc:
        return Result(
            provider, task.name, case["input"], case["expected"],
            False, 0.0, 0, 0, 0.0, {"error": f"{type(exc).__name__}: {exc}"},
        )
    return Result(
        provider, task.name, case["input"], case["expected"],
        correct_fn(answer, case["expected"], task.spec),
        latency_ms, in_tok, out_tok, cost, answer,
    )


def _truncate(text: str, width: int = 40) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


TABLE_HEADER = (
    f"{'task':<8} {'provider':<8} {'ok':>3} {'latency_ms':>10} "
    f"{'in_tok':>7} {'out_tok':>8} {'cost_usd':>10}  {'input':<41} answer"
)


def print_table_header() -> None:
    print(TABLE_HEADER)
    print("-" * len(TABLE_HEADER))


def print_row(r: Result) -> None:
    print(
        f"{r.task:<8} {r.provider:<8} {('Y' if r.correct else 'N'):>3} {r.latency_ms:>10.1f} "
        f"{r.input_tokens:>7} {r.output_tokens:>8} {r.cost_usd:>10.6f}  "
        f"{_truncate(r.case_input):<41} {r.answer}"
    )


def _box_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]

    def border(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def render_row(cells: list[str], align: Callable[[str, int], str]) -> str:
        return "│ " + " │ ".join(align(c, w) for c, w in zip(cells, widths)) + " │"

    top, sep, bottom = border("┌", "┬", "┐"), border("├", "┼", "┤"), border("└", "┴", "┘")
    lines = [top, render_row(headers, str.center), sep]
    for i, row in enumerate(rows):
        lines.append(render_row(row, str.ljust))
        if i != len(rows) - 1:
            lines.append(sep)
    lines.append(bottom)
    return "\n".join(lines)


def _task_stats(results: list[Result], task_name: str, provider: str) -> tuple[float, float, float]:
    rows = [r for r in results if r.task == task_name and r.provider == provider]
    accuracy = sum(r.correct for r in rows) / len(rows)
    avg_latency = sum(r.latency_ms for r in rows) / len(rows)
    avg_cost = sum(r.cost_usd for r in rows) / len(rows)
    return accuracy, avg_latency, avg_cost


def print_summary(results: list[Result], tasks: list[Task]) -> None:
    box_rows: list[list[str]] = []

    def flush_box(source: str) -> None:
        if box_rows:
            print(f"\nSummary ({source}): jev vs. Claude Sonnet 5.5\n")
            print(_box_table(["Task", "jev acc", "claude acc", "speedup", "cost ratio"], box_rows))
            box_rows.clear()

    current_source = tasks[0].source if tasks else None
    for task in tasks:
        if task.source != current_source:
            flush_box(current_source)
            current_source = task.source

        print(f"\n[{task.name}]")
        jev_acc, jev_latency, jev_cost = _task_stats(results, task.name, "jev")
        claude_acc, claude_latency, claude_cost = _task_stats(results, task.name, "claude")
        print(f"  jev    accuracy={jev_acc:.0%}  avg_latency_ms={jev_latency:.1f}  avg_cost_usd={jev_cost:.6f}")
        print(f"  claude accuracy={claude_acc:.0%}  avg_latency_ms={claude_latency:.1f}  avg_cost_usd={claude_cost:.6f}")
        speedup = claude_latency / jev_latency if jev_latency else float("inf")
        cost_ratio = claude_cost / jev_cost if jev_cost else float("inf")
        print(f"  -> jev is {speedup:.1f}x faster, {cost_ratio:.0f}x cheaper than Claude Sonnet 5.5")

        box_rows.append([
            task.name, f"{jev_acc:.0%}", f"{claude_acc:.0%}", f"{speedup:.1f}x", f"{cost_ratio:.0f}x",
        ])

    flush_box(current_source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--suite", type=Path,
        help="A folder of JSON task-definition files, all of which are loaded and run "
        f"(default: {DEFAULT_CASES_PATH.name}/)",
    )
    group.add_argument(
        "--file", type=Path,
        help="A single JSON file of task definitions (instructions/criteria/cases) to run",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Text file to save the printed output to (default: {DEFAULT_OUTPUT_PATH.name})",
    )
    args = parser.parse_args()

    cases_path = args.file or args.suite or DEFAULT_CASES_PATH
    if args.file and not cases_path.is_file():
        parser.error(f"--file {cases_path} is not a file")
    if args.suite and not cases_path.is_dir():
        parser.error(f"--suite {cases_path} is not a folder")

    tasks = load_tasks(cases_path)

    jev_client = TypeSafeClient()
    claude_client = anthropic.Anthropic()

    real_stdout = sys.stdout
    with open(args.output, "w") as output_file:
        sys.stdout = _Tee(real_stdout, output_file)
        try:
            results: list[Result] = []
            print_table_header()
            try:
                for task in tasks:
                    for case in task.cases:
                        jev_result = run_case(task.jev_fn, task.correct_fn, jev_client, "jev", task, case)
                        print_row(jev_result)
                        results.append(jev_result)

                        claude_result = run_case(task.claude_fn, task.correct_fn, claude_client, "claude", task, case)
                        print_row(claude_result)
                        results.append(claude_result)
            finally:
                jev_client.close()

            print_summary(results, tasks)
        finally:
            sys.stdout = real_stdout

    print(f"\nSaved output to {args.output}")


if __name__ == "__main__":
    main()
