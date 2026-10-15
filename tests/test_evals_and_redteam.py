from pathlib import Path

import pytest

from agentops.evals import AnswerRelevancyMetric, FaithfulnessMetric, TestCase, evaluate, load_jsonl
from agentops.redteam import run_attack

ROOT = Path(__file__).resolve().parents[1]


def test_eval_suite_meets_thresholds():
    report = evaluate(ROOT / "evals" / "dataset.jsonl")
    failing = [c for c in report["cases"] if not c["passed"]]
    assert not failing, failing
    assert report["summary"]["pii_leakage"] == 1.0


def test_faithfulness_catches_invented_ids():
    tc = TestCase(id="x", turns=["hi"], answer="Created OPS-99", tool_outputs=[{"key": "OPS-1"}])
    assert FaithfulnessMetric().measure(tc) == 0.0


def test_relevancy_partial_credit():
    tc = TestCase(id="x", turns=[], answer="OPS-1 created", spec={"expected_in_answer": ["OPS-1", "High"]})
    assert AnswerRelevancyMetric().measure(tc) == 0.5


@pytest.mark.parametrize("spec", load_jsonl(ROOT / "redteam" / "attacks.jsonl"), ids=lambda s: s["id"])
def test_redteam_attack_defended(spec):
    r = run_attack(spec)
    assert r["defended"], r
