"""Run the fixed Lite/Pro comparison cases without changing product facts.

The default mode uses scripted decisions and scripted Java observations.  It is
the deterministic safety gate for the Pro executor.  ``--mode real`` reuses
the same observations while allowing the configured model to make decisions;
that result is reported separately and is never mixed into the scripted score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clothing_assistant.agent.pro.executor import run_pro_agent
from clothing_assistant.agent.pro.validation import validate_pro_result
from clothing_assistant.infrastructure.java_tool_client import tool_result


DEFAULT_CASES = ROOT / "tests" / "fixtures" / "pro" / "comparison_cases.json"
DEFAULT_OUTPUT = ROOT / "docs" / "evals" / "lite-pro-comparison.json"
EXPECTED_SCHEMA_VERSION = "assistant-pro-eval-v1"
EXPECTED_DATA_VERSION = "task11-pro-comparison-v1"
TOOL_FACTS_VERSION = "java-fixture-2026-09"
_TOOL_SOURCES = {
    "search_products": "java",
    "get_product_detail": "java",
    "check_availability": "java",
    "search_product_knowledge": "rag",
    "search_policy": "policy",
    "recommend_size": "size",
}


def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    """Load and validate the versioned Task11 comparison fixture.

    Args:
        path: JSON fixture path.

    Returns:
        The ordered case dictionaries from the fixture.

    Raises:
        ValueError: If the fixture does not satisfy the small evaluator schema.
    """
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("comparison fixture must be an object")
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("unsupported comparison fixture schema")
    if payload.get("data_version") != EXPECTED_DATA_VERSION:
        raise ValueError("unsupported comparison fixture data version")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("comparison fixture must contain cases")

    required = {"id", "query", "mode", "hard_constraints", "expected", "fake"}
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not required <= set(case):
            raise ValueError("every comparison case needs id, query, mode, constraints, expected and fake")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("comparison case ids must be unique non-empty strings")
        seen.add(case_id)
        if case["mode"] != "pro" or not isinstance(case["hard_constraints"], dict):
            raise ValueError(f"invalid mode or hard constraints in {case_id}")
        expected = case["expected"]
        fake = case["fake"]
        if not isinstance(expected, dict) or expected.get("hard_constraints") != case["hard_constraints"]:
            raise ValueError(f"expected hard constraints must mirror the case in {case_id}")
        if not isinstance(expected.get("must_answer"), list):
            raise ValueError(f"must_answer is required in {case_id}")
        if not isinstance(expected.get("allowed_skus"), list) or "failure_type" not in expected:
            raise ValueError(f"expected product and failure fields are required in {case_id}")
        if not isinstance(fake, dict) or not isinstance(fake.get("decisions"), list) or not fake["decisions"]:
            raise ValueError(f"fake decisions are required in {case_id}")
    return cases


def _condition_matches(condition: object, snapshot: object) -> bool:
    """Match one small declarative condition against the executor snapshot."""
    if not isinstance(snapshot, Mapping):
        return False
    observations = snapshot.get("observations")
    if not isinstance(observations, (list, tuple)):
        observations = []
    last = observations[-1] if observations else {}
    if condition == "start":
        return not observations
    if condition == "always":
        return True
    if condition == "last_status_ok":
        return last.get("status") == "ok"
    if condition == "last_status_empty":
        return last.get("status") == "empty"
    if condition == "last_status_no_evidence":
        return last.get("status") == "no_evidence"
    if condition == "last_status_unavailable":
        return last.get("status") == "unavailable"
    if condition == "last_stock_zero":
        return isinstance(last.get("data"), Mapping) and last["data"].get("available_stock") == 0
    if condition == "last_stock_positive":
        return isinstance(last.get("data"), Mapping) and last["data"].get("available_stock", 0) > 0
    if isinstance(condition, str) and condition.startswith("last_tool_"):
        return last.get("tool") == condition.removeprefix("last_tool_")
    if isinstance(condition, dict):
        if "observation_count" in condition:
            return len(observations) == condition["observation_count"]
        return all(last.get(key) == value for key, value in condition.items())
    return False


def _select_scripted_action(decisions: list[dict[str, Any]], snapshot: object) -> dict[str, Any]:
    """Choose the first fixture action whose condition matches current evidence."""
    for decision in decisions:
        if isinstance(decision, dict) and _condition_matches(decision.get("when", "always"), snapshot):
            action = decision.get("action")
            if isinstance(action, dict):
                return action
    return {"kind": "finish", "answer": "没有更多可执行步骤", "product_refs": []}


def _fake_tool_result(case: dict[str, Any], action: object):
    """Return the fixed Java/knowledge observation matching one tool action."""
    tool_name = getattr(action, "tool", "")
    arguments = getattr(action, "arguments", None)
    arguments = arguments.model_dump(exclude_none=True) if arguments is not None else {}
    for spec in case["fake"].get("tool_results", []):
        if not isinstance(spec, dict) or spec.get("tool") != tool_name:
            continue
        match = spec.get("match", {})
        if isinstance(match, dict) and all(arguments.get(key) == value for key, value in match.items()):
            return tool_result(
                _TOOL_SOURCES[tool_name],
                spec.get("status", "empty"),
                spec.get("data"),
                evidence_id=spec.get("evidence_id"),
                missing_fields=spec.get("missing_fields", []),
                error_code=spec.get("error_code"),
            )
    return tool_result(_TOOL_SOURCES.get(tool_name, "java"), "empty", [])


def _score_case(
    case: dict[str, Any],
    executor_result: dict[str, Any],
    done: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    """Compare one validated result with hard fixture expectations."""
    expected = case["expected"]
    failures: list[dict[str, Any]] = []
    pairs = [[item["spu_id"], item["sku_id"]] for item in done.get("product_refs", [])]
    allowed = sorted(expected["allowed_skus"], key=lambda item: (item["spu_id"], item["sku_id"]))
    actual_sorted = sorted(pairs)
    expected_pairs = sorted(
        [[item["spu_id"], item["sku_id"]] for item in allowed]
    )
    if actual_sorted != expected_pairs:
        failures.append({"reason": "unexpected_product_pairs", "expected": expected_pairs, "actual": actual_sorted})
    if done.get("stop_reason") != expected.get("stop_reason"):
        failures.append({"reason": "unexpected_stop_reason", "expected": expected.get("stop_reason"), "actual": done.get("stop_reason")})
    answer = str(done.get("answer", ""))
    for required_text in expected.get("must_answer", []):
        if required_text not in answer:
            failures.append({"reason": "missing_required_answer_text", "text": required_text})

    observed_tools = [item.get("tool") for item in executor_result.get("observations", [])]
    if expected.get("tools") != observed_tools:
        failures.append({"reason": "unexpected_tool_trace", "expected": expected.get("tools"), "actual": observed_tools})

    failure_type = expected.get("failure_type")
    requirements = done.get("requirements", [])
    statuses = {item.get("status") for item in requirements if isinstance(item, dict)}
    if failure_type == "policy_no_evidence" and statuses != {"unconfirmed"}:
        failures.append({"reason": "policy_evidence_was_not_marked_unconfirmed", "actual": sorted(statuses)})
    if failure_type == "size_basis_unconfirmed":
        basis = done.get("product_refs", [{}])[0].get("basis") if done.get("product_refs") else None
        if basis != "generic_rule":
            failures.append({"reason": "size_basis_was_not_downgraded", "actual": basis})
    if failure_type == "inventory_timeout" and executor_result.get("metrics", {}).get("tool_calls") != 2:
        failures.append({"reason": "timeout_retry_count_mismatch", "actual": executor_result.get("metrics", {}).get("tool_calls")})
    if failure_type in {"invalid_product", "no_match", "cancelled", "loop_exhausted"} and pairs:
        failures.append({"reason": "failure_case_exposed_product", "actual": pairs})

    return {
        "case_id": case["id"],
        "mode": "pro",
        "source": source,
        "query": case["query"],
        "hard_constraints": case["hard_constraints"],
        "passed": not failures,
        "failures": failures,
        "failure_type": failure_type,
        "stop_reason": done.get("stop_reason"),
        "answer": answer,
        "product_pairs": pairs,
        "observed_tools": observed_tools,
        "requirements": requirements,
        "metrics": executor_result.get("metrics", {}),
        "capabilities": {
            "compound_requirements": True,
            "result_driven_routing": True,
            "cross_run_product_isolation": True,
        },
    }


def _run_pro_case(case: dict[str, Any], *, scripted: bool) -> dict[str, Any]:
    """Execute one case with either scripted or configured model decisions."""
    run_id = f"eval-{case['id']}"
    cancelled = False
    tool_calls = 0
    fake = case["fake"]
    context = {
        "request_id": f"request-{case['id']}",
        "thread_id": f"thread-{case['id']}",
        "run_id": run_id,
        "explicit_filters": case["hard_constraints"],
        "stop_requested": lambda: cancelled,
    }

    def execute(action: object, *, timeout: float):
        nonlocal cancelled, tool_calls
        del timeout
        tool_calls += 1
        result = _fake_tool_result(case, action)
        if fake.get("cancel_after_tool_calls") and tool_calls >= fake["cancel_after_tool_calls"]:
            cancelled = True
        return result

    kwargs: dict[str, Any] = {"execute_tool": execute, "context": context}
    if scripted:
        kwargs["decide"] = lambda snapshot, **_: _select_scripted_action(fake["decisions"], snapshot)
    if fake.get("limits"):
        kwargs["limits"] = fake["limits"]
    executor_result = run_pro_agent(case["query"], **kwargs)
    done = validate_pro_result(executor_result, context=context)
    return _score_case(
        case,
        executor_result,
        done,
        source="scripted_fake" if scripted else "configured_model",
    )


def run_fake_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one deterministic Pro case used by the regression gate."""
    return _run_pro_case(case, scripted=True)


def run_lite_baseline(case: dict[str, Any]) -> dict[str, Any]:
    """Describe Lite as a reference flow without pretending it is a model score."""
    return {
        "case_id": case["id"],
        "mode": "lite",
        "source": "scripted_baseline",
        "query": case["query"],
        "hard_constraints": case["hard_constraints"],
        "evaluation_status": "reference_only",
        "passed": None,
        "failures": [],
        "failure_type": None,
        "stop_reason": "fixed_flow_reference",
        "answer": None,
        "product_pairs": [],
        "observed_tools": ["legacy_fixed_flow"],
        "requirements": [],
        "metrics": {
            "decisions": 1,
            "tool_calls": 1,
            "elapsed_ms": None,
            "token_usage": None,
        },
        "capabilities": {
            "compound_requirements": False,
            "result_driven_routing": False,
            "cross_run_product_isolation": False,
        },
    }


def build_comparison_report(cases: list[dict[str, Any]], *, mode: str = "fake") -> dict[str, Any]:
    """Build a report with isolated Lite, fake Pro and optional real Pro rows."""
    if mode not in {"fake", "real", "both"}:
        raise ValueError("mode must be fake, real or both")
    runs: list[dict[str, Any]] = []
    for case in cases:
        runs.append(run_lite_baseline(case))
        if mode in {"fake", "both"}:
            runs.append(run_fake_case(case))
        if mode in {"real", "both"}:
            try:
                runs.append(_run_pro_case(case, scripted=False))
            except Exception as error:
                runs.append({
                    "case_id": case["id"],
                    "mode": "pro",
                    "source": "configured_model",
                    "query": case["query"],
                    "hard_constraints": case["hard_constraints"],
                    "passed": False,
                    "failures": [{"reason": "runner_error", "type": type(error).__name__}],
                    "failure_type": "runner_error",
                    "stop_reason": "runner_error",
                    "answer": "",
                    "product_pairs": [],
                    "observed_tools": [],
                    "requirements": [],
                    "metrics": {"decisions": None, "tool_calls": None, "elapsed_ms": None, "token_usage": None},
                })

    comparisons = []
    for case in cases:
        case_runs = [row for row in runs if row["case_id"] == case["id"]]
        comparisons.append({
            "case_id": case["id"],
            "lite": next(row for row in case_runs if row["mode"] == "lite"),
            "pro_fake": next((row for row in case_runs if row["source"] == "scripted_fake"), None),
            "pro_real": next((row for row in case_runs if row["source"] == "configured_model"), None),
        })

    fake_rows = [row for row in runs if row["source"] == "scripted_fake"]
    real_rows = [row for row in runs if row["source"] == "configured_model"]
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "data_version": EXPECTED_DATA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "case_ids": [case["id"] for case in cases],
            "queries": [case["query"] for case in cases],
            "tool_facts_version": TOOL_FACTS_VERSION,
        },
        "model": {
            "source": "scripted_fake" if mode == "fake" else "configured_model",
            "name": os.getenv("KIMI_CHAT_MODEL", "unknown"),
            "real_model_executed": bool(real_rows),
            "usage_policy": "missing usage is null; cost is omitted without a verified price",
        },
        "summary": {
            "case_count": len(cases),
            "run_count": len(runs),
            "pro_fake_passed": sum(row.get("passed") is True for row in fake_rows),
            "pro_fake_failed": sum(row.get("passed") is False for row in fake_rows),
            "real_model_executed": bool(real_rows),
            "real_model_passed": sum(row.get("passed") is True for row in real_rows),
            "real_model_failed": sum(row.get("passed") is False for row in real_rows),
            "lite_reference_rows": sum(row["mode"] == "lite" for row in runs),
        },
        "runs": runs,
        "comparisons": comparisons,
    }


def write_report(report: dict[str, Any], path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write a UTF-8 JSON report and create its parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the comparison evaluator."""
    parser = argparse.ArgumentParser(description="Compare the fixed Lite reference with Pro decisions.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Versioned comparison fixture JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report destination.")
    parser.add_argument("--mode", choices=("fake", "real", "both"), default="fake", help="Run scripted, configured-model, or both Pro paths.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluator and print a compact result summary."""
    args = build_arg_parser().parse_args(argv)
    report = build_comparison_report(load_cases(args.cases), mode=args.mode)
    output_path = write_report(report, args.output)
    summary = report["summary"]
    print(
        f"Lite/Pro comparison written to {output_path}; "
        f"fake Pro {summary['pro_fake_passed']}/{summary['case_count']} passed; "
        f"real model executed={summary['real_model_executed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
