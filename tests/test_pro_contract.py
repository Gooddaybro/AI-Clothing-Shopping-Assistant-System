"""Executable boundary examples for the independently versioned Pro protocol."""

import copy
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).parents[1] / "contracts" / "assistant-streaming-chat"


def load(kind, name):
    path = ROOT / kind / name
    assert path.exists(), f"Missing Pro contract: {path.name}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("boundary", ["request", "done"])
def test_valid_example(boundary):
    schema = load("schemas", f"v2-pro-{boundary}.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(load("examples", f"v2-pro-{boundary}.json"), schema)


@pytest.mark.parametrize("field,value", [("agent_mode", "pro_max"), ("tool_run_token", ""), ("unexpected", True)])
def test_request_rejects_invalid_boundary(field, value):
    schema = load("schemas", "v2-pro-request.schema.json")
    payload = copy.deepcopy(load("examples", "v2-pro-request.json"))
    payload[field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


def test_done_does_not_accept_credentials():
    schema = load("schemas", "v2-pro-done.schema.json")
    payload = load("examples", "v2-pro-done.json")
    payload["tool_run_token"] = "secret"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


@pytest.mark.parametrize("field,value", [
    ("explicit_filters", {"budget_max": "NaN"}),
    ("explicit_filters", {"budget_max": "-1.00"}),
    ("explicit_filters", {"user_id": 88}),
    ("query", "   "),
])
def test_request_rejects_unsafe_filters_and_blank_query(field, value):
    schema = load("schemas", "v2-pro-request.schema.json")
    payload = load("examples", "v2-pro-request.json")
    payload[field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


@pytest.mark.parametrize("field,value", [
    ("stop_reason", "invented_state"),
    ("requirements", [{"id": "r1", "text": "退换", "status": "invented", "evidence_ids": []}]),
    ("metrics", {"decisions": -1, "tool_calls": 0, "elapsed_ms": 0, "token_usage": None}),
    ("product_refs", [{"spu_id": 1, "sku_id": 2, "reason": "推荐", "sale_price": "1.00"}]),
])
def test_done_rejects_invalid_states_and_fabricated_card_facts(field, value):
    schema = load("schemas", "v2-pro-done.schema.json")
    payload = load("examples", "v2-pro-done.json")
    payload[field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)
