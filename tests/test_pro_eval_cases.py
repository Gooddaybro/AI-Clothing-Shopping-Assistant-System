import json
from pathlib import Path

from scripts.eval_pro_comparison import build_comparison_report, load_cases, run_fake_case


ROOT = Path(__file__).parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "pro" / "comparison_cases.json"


def test_comparison_fixture_covers_the_task11_scenarios():
    cases = load_cases(FIXTURE_PATH)

    assert len(cases) == 10
    assert {case["id"] for case in cases} == {
        "single_explicit_request",
        "compound_five_requirements",
        "switch_when_a_out_of_stock",
        "no_product_within_budget",
        "policy_without_evidence",
        "size_without_product_chart",
        "inventory_timeout",
        "decision_loop_exhausted",
        "cross_mode_invalid_product",
        "cancelled_request",
    }

    for case in cases:
        expected = case["expected"]
        assert expected["hard_constraints"] == case["hard_constraints"]
        assert isinstance(expected["must_answer"], list)
        assert isinstance(expected["allowed_skus"], list)
        assert "failure_type" in expected
        assert case["fake"]["decisions"]


def test_fake_pro_evaluation_follows_observed_stock_and_rejects_foreign_product():
    cases = {case["id"]: case for case in load_cases(FIXTURE_PATH)}

    switched = run_fake_case(cases["switch_when_a_out_of_stock"])
    assert switched["passed"] is True
    assert switched["product_pairs"] == [[102, 202]]
    assert switched["observed_tools"] == ["check_availability", "check_availability"]

    invalid = run_fake_case(cases["cross_mode_invalid_product"])
    assert invalid["passed"] is True
    assert invalid["product_pairs"] == []
    assert invalid["stop_reason"] == "validation_failed"


def test_comparison_report_separates_scripted_fake_results_from_real_model_results():
    cases = load_cases(FIXTURE_PATH)
    report = build_comparison_report(cases, mode="fake")

    assert report["data_version"] == "task11-pro-comparison-v1"
    assert report["input"]["tool_facts_version"] == "java-fixture-2026-09"
    assert report["model"]["source"] == "scripted_fake"
    assert {row["mode"] for row in report["runs"]} == {"lite", "pro"}
    assert {row["source"] for row in report["runs"]} == {"scripted_fake", "scripted_baseline"}
    assert sum(row["mode"] == "pro" and row["source"] == "scripted_fake" for row in report["runs"]) == 10
    assert report["summary"]["pro_fake_passed"] == 10
    assert report["summary"]["real_model_executed"] is False
    assert all(row["capabilities"]["result_driven_routing"] for row in report["runs"] if row["mode"] == "pro")
    json.dumps(report, ensure_ascii=False)
