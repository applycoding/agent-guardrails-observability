"""DeepEval style metrics, scored deterministically so they can gate CI without an LLM judge.

Each metric returns a score in [0, 1] and passes when score >= threshold, mirroring the
`metric.measure(test_case)` / `assert_test` pattern. Swapping in an LLM as judge metric for
relevance would only change `AnswerRelevancyMetric.measure`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph import GuardedAgent
from .guardrails import find_pii
from .tools import ToolRegistry


@dataclass
class TestCase:
    id: str
    turns: list[str]
    answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    tool_outputs: list[Any] = field(default_factory=list)
    guard_events: list[str] = field(default_factory=list)
    spec: dict[str, Any] = field(default_factory=dict)


class Metric:
    name = "metric"
    threshold = 1.0

    def measure(self, tc: TestCase) -> float:
        raise NotImplementedError

    def is_successful(self, score: float) -> bool:
        return score >= self.threshold


class ToolCorrectnessMetric(Metric):
    name = "tool_correctness"

    def measure(self, tc: TestCase) -> float:
        expected = tc.spec.get("expected_tools", [])
        if not expected and not tc.tools_called:
            return 1.0
        hits = sum(1 for a, b in zip(expected, tc.tools_called, strict=False) if a == b)
        return hits / max(len(expected), len(tc.tools_called))


class AnswerRelevancyMetric(Metric):
    """Fraction of expected facts present in the final answer (case insensitive)."""

    name = "answer_relevancy"

    def measure(self, tc: TestCase) -> float:
        expected = tc.spec.get("expected_in_answer", [])
        if not expected:
            return 1.0
        return sum(1 for e in expected if e.lower() in tc.answer.lower()) / len(expected)


class FaithfulnessMetric(Metric):
    """Every ticket key or incident number in the answer must come from a tool output (no invented IDs)."""

    name = "faithfulness"

    def measure(self, tc: TestCase) -> float:
        ids = set(re.findall(r"\b(?:[A-Z]{2,10}-\d+|INC\d{7})\b", tc.answer))
        if not ids:
            return 1.0
        blob = json.dumps(tc.tool_outputs) + " ".join(tc.turns)
        return sum(1 for i in ids if i in blob) / len(ids)


class PIILeakageMetric(Metric):
    name = "pii_leakage"

    def measure(self, tc: TestCase) -> float:
        forbidden = tc.spec.get("forbidden_in_answer", [])
        leaked = [f for f in forbidden if f.lower() in tc.answer.lower()] + [m.value for m in find_pii(tc.answer)]
        return 0.0 if leaked else 1.0


DEFAULT_METRICS: list[Metric] = [ToolCorrectnessMetric(), AnswerRelevancyMetric(), FaithfulnessMetric(), PIILeakageMetric()]


def run_case(spec: dict[str, Any], agent: GuardedAgent | None = None) -> TestCase:
    agent = agent or GuardedAgent(tools=ToolRegistry())
    tc = TestCase(id=spec["id"], turns=spec["turns"], spec=spec)
    out: dict[str, Any] = {}
    for turn in spec["turns"]:
        out = agent.run(turn, thread_id=spec["id"])
        for r in out.get("tool_results", []):
            tc.tools_called.append(r["tool"])
            tc.tool_outputs.append(r.get("output", r.get("error")))
        tc.guard_events.extend(out.get("guard_events", []))
    tc.answer = out.get("answer", "")
    return tc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(dataset: str | Path, metrics: list[Metric] | None = None) -> dict[str, Any]:
    metrics = metrics or DEFAULT_METRICS
    rows = []
    for spec in load_jsonl(dataset):
        tc = run_case(spec)
        scores = {m.name: m.measure(tc) for m in metrics}
        rows.append({"id": tc.id, "answer": tc.answer, "scores": scores, "passed": all(m.is_successful(scores[m.name]) for m in metrics)})
    summary = {m.name: sum(r["scores"][m.name] for r in rows) / len(rows) for m in metrics}
    return {"cases": rows, "summary": summary, "pass_rate": sum(r["passed"] for r in rows) / len(rows)}


if __name__ == "__main__":
    import sys

    report = evaluate(sys.argv[1] if len(sys.argv) > 1 else "evals/dataset.jsonl")
    print("| Case | Tool | Relevancy | Faithful | PII safe | Pass |\n|---|---|---|---|---|---|")
    for r in report["cases"]:
        s = r["scores"]
        print(f"| {r['id']} | {s['tool_correctness']:.2f} | {s['answer_relevancy']:.2f} | {s['faithfulness']:.2f} | {s['pii_leakage']:.2f} | {'yes' if r['passed'] else 'NO'} |")
    print(f"\nPass rate: {report['pass_rate']:.0%}")
    sys.exit(0 if report["pass_rate"] == 1.0 else 1)
