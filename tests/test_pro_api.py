import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from clothing_assistant.api.app import app


def _request(**overrides):
    value = {
        "contract_version": "assistant-v2",
        "agent_mode": "pro",
        "request_id": "req-v2-api",
        "run_id": "run-v2-api",
        "thread_id": "thread-v2-api",
        "query": "推荐一件通勤外套",
        "chat_history": [],
        "user_context": {"user_id": 1001, "height_cm": 175, "weight_kg": 70},
        "explicit_filters": {"budget_max": "300.00"},
        "current_product_refs": [],
        "tool_run_token": "run-secret",
    }
    value.update(overrides)
    return value


def _executor_result(run_id="run-v2-api"):
    return {
        "run_id": run_id,
        "stop_reason": "validation_required",
        "proposal": {
            "kind": "finish",
            "answer": "推荐这件通勤外套。",
            "product_refs": [
                {
                    "spu_id": 1,
                    "sku_id": 11,
                    "reason": "适合通勤",
                    "evidence_ids": ["e-search"],
                }
            ],
        },
        "evidence": [
            {
                "run_id": run_id,
                "evidence_id": "e-search",
                "tool": "search_products",
                "source": "java",
                "status": "ok",
                "arguments": {},
                "data": [
                    {
                        "spu_id": 1,
                        "sku_id": 11,
                        "sale_price": "299.90",
                        "available_stock": 2,
                    }
                ],
            }
        ],
        "observations": [],
        "metrics": {
            "decisions": 1,
            "tool_calls": 1,
            "elapsed_ms": 3,
            "token_usage": None,
        },
    }


def test_v2_sync_requires_internal_token():
    with patch.dict(
        os.environ,
        {"AI_RUNTIME_ENV": "production", "APP_INTERNAL_API_TOKEN": "internal-secret"},
    ):
        response = TestClient(app).post("/v2/chat", json=_request())

    assert response.status_code == 401
    assert response.json()["error"] == "internal_auth_required"


def test_v2_sync_rejects_wrong_contract_version_or_mode():
    with patch.dict(
        os.environ,
        {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""},
    ):
        for change in ({"contract_version": "assistant-v1"}, {"agent_mode": "lite"}):
            response = TestClient(app).post("/v2/chat", json=_request(**change))
            assert response.status_code == 422


def test_v2_sync_returns_validated_done_payload():
    with patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}):
        with patch(
            "clothing_assistant.api.pro_routes.run_pro_agent",
            return_value=_executor_result(),
        ) as run_agent:
            response = TestClient(app).post("/v2/chat", json=_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "assistant-v2"
    assert payload["agent_mode"] == "pro"
    assert payload["run_id"] == "run-v2-api"
    assert payload["product_refs"][0]["spu_id"] == 1
    assert "sale_price" not in payload["product_refs"][0]
    run_agent.assert_called_once()


def test_v2_sync_is_bounded_by_request_schema():
    with patch.dict(
        os.environ,
        {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""},
    ):
        response = TestClient(app).post("/v2/chat", json=_request(query="x" * 2001))

    assert response.status_code == 422


def test_v2_sync_hides_executor_failure_details():
    with (
        patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}),
        patch(
            "clothing_assistant.api.pro_routes.run_pro_agent",
            side_effect=RuntimeError("secret tool response"),
        ),
    ):
        response = TestClient(app).post("/v2/chat", json=_request())

    assert response.status_code == 503
    assert response.json() == {
        "error": "internal_error",
        "message": "AI service failed to process the request.",
        "run_id": "run-v2-api",
    }
    assert "secret tool response" not in response.text
