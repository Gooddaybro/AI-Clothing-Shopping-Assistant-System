import asyncio
import json
import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from clothing_assistant.api.app import app
from clothing_assistant.api.pro_routes import generate_pro_chat_stream

from tests.test_pro_api import _executor_result, _request


def _events(body):
    result = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = json.loads(next(line.removeprefix("data: ") for line in lines if line.startswith("data: ")))
        result.append((event, data))
    return result


def test_v2_stream_emits_progress_then_validated_tokens_then_done():
    def fake_executor(_query, *, context, **_kwargs):
        context["progress_callback"]("search_products", "started")
        context["progress_callback"]("search_products", "completed")
        return _executor_result()

    with (
        patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}),
        patch("clothing_assistant.api.pro_routes.run_pro_agent", side_effect=fake_executor),
    ):
        with TestClient(app).stream("POST", "/v2/chat/stream", json=_request()) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    events = _events(body)
    kinds = [event for event, _ in events]
    assert kinds[:2] == ["progress", "progress"]
    assert set(kinds[2:]) == {"token", "done"}
    assert kinds[-1] == "done"
    assert all(data["run_id"] == "run-v2-api" for event, data in events if event == "progress")
    assert [data["stage"] for event, data in events if event == "progress"] == [
        "started",
        "completed",
    ]
    assert "售价" not in "".join(data.get("content", "") for event, data in events if event == "token")
    assert events[-1][1]["answer"] == "".join(
        data["content"] for event, data in events if event == "token"
    )


def test_v2_stream_emits_one_safe_error_and_no_token_on_executor_failure():
    with (
        patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}),
        patch(
            "clothing_assistant.api.pro_routes.run_pro_agent",
            side_effect=RuntimeError("raw reasoning"),
        ),
    ):
        with TestClient(app).stream("POST", "/v2/chat/stream", json=_request()) as response:
            body = response.read().decode("utf-8")

    events = _events(body)
    assert [event for event, _ in events] == ["error"]
    assert events[0][1]["code"] == "internal_error"
    assert "raw reasoning" not in body


def test_v2_stream_data_payloads_are_single_line_json():
    def fake_executor(_query, *, context, **_kwargs):
        context["progress_callback"]("search_products", "started")
        context["progress_callback"]("search_products", "completed")
        return _executor_result()

    with (
        patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}),
        patch("clothing_assistant.api.pro_routes.run_pro_agent", side_effect=fake_executor),
    ):
        with TestClient(app).stream("POST", "/v2/chat/stream", json=_request()) as response:
            body = response.read().decode("utf-8")

    for line in body.splitlines():
        if line.startswith("data: "):
            json.loads(line.removeprefix("data: "))
            assert "\n" not in line.removeprefix("data: ")


def test_v2_stream_progress_wraps_real_executor_tool_attempts():
    from clothing_assistant.agent.pro.schemas import parse_action
    from clothing_assistant.infrastructure.java_tool_client import tool_result

    captured = {}

    def fake_executor(_query, *, execute_tool, **_kwargs):
        captured["result"] = execute_tool(
            parse_action(
                {
                    "kind": "tool",
                    "tool": "search_products",
                    "arguments": {},
                }
            ),
            timeout=1,
        )
        return _executor_result()

    with (
        patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}),
        patch("clothing_assistant.api.pro_routes.run_pro_agent", side_effect=fake_executor),
        patch(
            "clothing_assistant.api.pro_routes.execute_pro_tool",
            return_value=tool_result("java", "empty"),
        ) as execute_tool,
    ):
        with TestClient(app).stream("POST", "/v2/chat/stream", json=_request()) as response:
            events = _events(response.read().decode("utf-8"))

    assert response.status_code == 200
    assert captured["result"].status == "empty"
    execute_tool.assert_called_once()
    progress = [data for event, data in events if event == "progress"]
    assert [(item["tool"], item["stage"]) for item in progress] == [
        ("search_products", "started"),
        ("search_products", "completed"),
    ]


def test_v2_stream_passes_disconnect_to_executor():
    stopped = asyncio.Event()

    class DisconnectingRequest:
        def __init__(self):
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    def fake_executor(_query, *, context, **_kwargs):
        while not context["stop_requested"]():
            pass
        stopped_loop = context["stop_requested"]()
        if stopped_loop:
            stopped.set()
        return _executor_result()

    async def collect():
        request = DisconnectingRequest()
        return [event async for event in generate_pro_chat_stream(_request_model(), request)]

    from clothing_assistant.api.pro_routes import ProChatRequest

    with (
        patch("clothing_assistant.api.pro_routes.run_pro_agent", side_effect=fake_executor),
        patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}),
    ):
        events = asyncio.run(collect())

    assert events == []
    assert stopped.is_set()


def test_v2_sync_and_stream_share_the_same_validated_done_shape():
    fake_result = _executor_result()
    with (
        patch.dict(os.environ, {"AI_RUNTIME_ENV": "development", "APP_INTERNAL_API_TOKEN": ""}),
        patch("clothing_assistant.api.pro_routes.run_pro_agent", return_value=fake_result),
    ):
        sync_payload = TestClient(app).post("/v2/chat", json=_request()).json()
        with TestClient(app).stream("POST", "/v2/chat/stream", json=_request()) as response:
            stream_payload = _events(response.read().decode("utf-8"))[-1][1]

    assert stream_payload == sync_payload


def _request_model():
    from clothing_assistant.api.pro_routes import ProChatRequest

    return ProChatRequest.model_validate(_request())
