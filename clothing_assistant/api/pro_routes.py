"""Internal Python entry points for the versioned Pro assistant contract.

Java owns authentication, identity and product facts.  This module only accepts
the trusted v2 request assembled by Java, runs the isolated Pro executor, and
publishes a validated draft.  Public Java streaming and persistence are added
in the Java service; this endpoint never publishes credentials or raw tool
observations.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from clothing_assistant.agent.pro.executor import run_pro_agent
from clothing_assistant.agent.pro.tools import execute_tool as execute_pro_tool
from clothing_assistant.agent.pro.validation import validate_pro_result
from clothing_assistant.api.streaming import (
    build_pro_done_event,
    build_pro_error_event,
    build_pro_progress_event,
    build_pro_token_event,
    iter_answer_chunks,
)
from clothing_assistant.config_data import get_internal_api_token
from clothing_assistant.infrastructure.java_tool_client import JavaToolClient


router = APIRouter(prefix="/v2", tags=["assistant-v2"])

_NON_BLANK = Field(min_length=1, max_length=128, pattern=r"\S")
_ERROR_STATUS = 503
_JAVA_BASE_URL_ENV = "JAVA_ASSISTANT_BASE_URL"
_JAVA_TOKEN_ENV = "JAVA_ASSISTANT_INTERNAL_TOKEN"
_MAX_STREAM_QUEUE = 64
_KNOWN_TOOLS = frozenset(
    {
        "search_products",
        "get_product_detail",
        "check_availability",
        "search_product_knowledge",
        "search_policy",
        "recommend_size",
    }
)
_PROGRESS_MESSAGES = {
    "search_products": "正在查询商品",
    "get_product_detail": "正在读取商品详情",
    "check_availability": "正在核验库存",
    "search_product_knowledge": "正在查询商品知识",
    "search_policy": "正在查询售后政策",
    "recommend_size": "正在计算尺码建议",
}
_ERROR_MESSAGES = {
    "timeout": ("dependency_timeout", "AI dependency timed out."),
    "rate_limited": ("dependency_unavailable", "AI dependency is temporarily unavailable."),
    "connection_error": ("dependency_unavailable", "AI dependency is temporarily unavailable."),
    "upstream_5xx": ("dependency_unavailable", "AI dependency is temporarily unavailable."),
    "invalid_response": (
        "dependency_invalid_response",
        "AI dependency returned an invalid response.",
    ),
    "validation_failed": ("unsafe_model_output", "AI output failed safety validation."),
    "unsafe_model_output": ("unsafe_model_output", "AI output failed safety validation."),
    "cancelled": ("cancelled", "The Pro request was cancelled."),
    "internal_error": ("internal_error", "AI service failed to process the request."),
}


class ProHistoryItem(BaseModel):
    """One bounded history turn supplied by Java as read-only context."""

    model_config = ConfigDict(extra="forbid", strict=True)

    user_query: StrictStr = Field(max_length=2000)
    assistant_answer: StrictStr = Field(max_length=8000)


class ProExplicitFilters(BaseModel):
    """Java's explicit filters; unknown fields are rejected at the boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    style: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    season: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    material: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    fit: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    gender: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    color: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    size: StrictStr | None = Field(default=None, min_length=1, max_length=100, pattern=r"\S")
    budget_max: StrictStr | None = Field(
        default=None,
        max_length=20,
        pattern=r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$",
    )


class ProProductRefInput(BaseModel):
    """A previously displayed reference supplied as non-authoritative context."""

    model_config = ConfigDict(extra="forbid", strict=True)

    spu_id: StrictInt = Field(gt=0)
    sku_id: StrictInt | None = Field(default=None, gt=0)


class ProChatRequest(BaseModel):
    """Strict Java-to-Python payload for the isolated Pro executor."""

    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["assistant-v2"]
    agent_mode: Literal["pro"]
    request_id: StrictStr = _NON_BLANK
    run_id: StrictStr = _NON_BLANK
    thread_id: StrictStr = _NON_BLANK
    query: StrictStr = Field(min_length=1, max_length=2000, pattern=r"\S")
    chat_history: list[ProHistoryItem] = Field(default_factory=list, max_length=100)
    user_context: dict[str, Any] = Field(default_factory=dict)
    explicit_filters: ProExplicitFilters = Field(default_factory=ProExplicitFilters)
    current_product_refs: list[ProProductRefInput] = Field(default_factory=list, max_length=20)
    tool_run_token: StrictStr = Field(min_length=1, max_length=128, pattern=r"\S")

    @field_validator("request_id", "run_id", "thread_id", "tool_run_token")
    @classmethod
    def identifiers_must_not_contain_whitespace(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("identifiers must not contain whitespace")
        return value

    @field_validator("user_context")
    @classmethod
    def context_must_be_bounded(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 64 or any(not isinstance(key, str) or len(key) > 128 for key in value):
            raise ValueError("user_context is too large")
        return value


class ProDoneProductRef(BaseModel):
    """Whitelisted product reference fields in Python's v2 terminal payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    spu_id: StrictInt = Field(gt=0)
    sku_id: StrictInt = Field(gt=0)
    reason: StrictStr = Field(min_length=1, max_length=2000)
    size_advice: StrictStr | None = Field(default=None, max_length=500)
    basis: Literal["generic_rule", "product_chart"] | None = None
    evidence_ids: list[StrictStr] = Field(
        default_factory=list,
        max_length=20,
    )


class ProDoneRequirement(BaseModel):
    """Whitelisted requirement status fields in a v2 terminal payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: StrictStr = Field(min_length=1, max_length=128, pattern=r"\S")
    text: StrictStr = Field(min_length=1, max_length=500)
    status: Literal["pending", "satisfied", "needs_input", "unconfirmed"]
    evidence_ids: list[StrictStr] = Field(default_factory=list, max_length=20)


class ProTokenUsage(BaseModel):
    """Exact nullable usage shape defined by the v2 contract."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)


class ProDoneMetrics(BaseModel):
    """Bounded execution metrics that are safe to return to Java."""

    model_config = ConfigDict(extra="forbid", strict=True)

    decisions: StrictInt = Field(ge=0)
    tool_calls: StrictInt = Field(ge=0)
    elapsed_ms: int | float = Field(ge=0)
    token_usage: ProTokenUsage | None = None


class ProDone(BaseModel):
    """Validated Python terminal result; Java still owns public card approval."""

    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["assistant-v2"]
    agent_mode: Literal["pro"]
    request_id: StrictStr = Field(min_length=1, max_length=128)
    run_id: StrictStr = Field(min_length=1, max_length=128)
    thread_id: StrictStr = Field(min_length=1, max_length=128)
    answer: StrictStr = Field(min_length=1, max_length=8000, pattern=r"\S")
    product_refs: list[ProDoneProductRef] = Field(max_length=20)
    requirements: list[ProDoneRequirement] = Field(max_length=12)
    stop_reason: Literal[
        "completed",
        "needs_input",
        "budget_exhausted",
        "validation_failed",
        "dependency_unavailable",
        "invalid_response",
        "cancelled",
    ]
    metrics: ProDoneMetrics


def build_java_tool_client(request: ProChatRequest, inbound_token: str | None) -> JavaToolClient | None:
    """Build a fixed Java gateway client from process configuration only.

    The request may carry the run token, but it can never choose the Java URL
    or gateway credential.  Missing configuration intentionally fails closed;
    the executor then returns a bounded dependency-unavailable result.
    """
    base_url = os.getenv(_JAVA_BASE_URL_ENV, "").strip()
    gateway_token = os.getenv(_JAVA_TOKEN_ENV, "").strip() or get_internal_api_token()
    if not base_url or not gateway_token:
        return None
    try:
        return JavaToolClient(
            base_url,
            gateway_token,
            request.run_id,
            request.tool_run_token,
            timeout=5.0,
        )
    except ValueError:
        return None


def _context_for_request(
    request: ProChatRequest,
    *,
    stop_requested,
    progress_callback,
    java_client: JavaToolClient | None,
) -> dict[str, Any]:
    """Convert the trusted request into the executor's non-secret context."""
    return {
        "request_id": request.request_id,
        "thread_id": request.thread_id,
        "run_id": request.run_id,
        "user_context": dict(request.user_context),
        "explicit_filters": request.explicit_filters.model_dump(exclude_none=True),
        "chat_history": [item.model_dump() for item in request.chat_history],
        "current_product_refs": [item.model_dump(exclude_none=True) for item in request.current_product_refs],
        "stop_requested": stop_requested,
        "progress_callback": progress_callback,
        "java_client": java_client,
    }


def _execute_tool_with_progress(action, *, timeout: float, java_client, progress_callback):
    """Wrap each real executor tool attempt with safe lifecycle events."""
    tool = getattr(action, "tool", "")
    if tool in _KNOWN_TOOLS:
        progress_callback(tool, "started")
    try:
        return execute_pro_tool(
            tool,
            action.arguments.model_dump(exclude_none=True),
            java_client=java_client,
            timeout=timeout,
        )
    finally:
        if tool in _KNOWN_TOOLS:
            progress_callback(tool, "completed")


def _safe_done(request: ProChatRequest, result: Mapping[str, Any], *, context: Mapping[str, Any]) -> dict[str, Any]:
    """Run Task7 validation and enforce the exact request-bound v2 done shape."""
    validated = validate_pro_result(result, context=context)
    if not isinstance(validated, Mapping):
        raise ValueError("validator returned a non-object")

    # Pick only fields defined by the shared contract.  Request identity is
    # always taken from the trusted request, never from model or validator text.
    refs = []
    for item in validated.get("product_refs", []):
        if isinstance(item, Mapping):
            refs.append(
                {
                    key: item[key]
                    for key in ("spu_id", "sku_id", "reason", "size_advice", "basis", "evidence_ids")
                    if key in item
                }
            )
    requirements = []
    for item in validated.get("requirements", []):
        if isinstance(item, Mapping):
            requirements.append(
                {
                    key: item[key]
                    for key in ("id", "text", "status", "evidence_ids")
                    if key in item
                }
            )
    metrics = validated.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("validator returned invalid metrics")
    safe_metrics = {
        "decisions": metrics.get("decisions"),
        "tool_calls": metrics.get("tool_calls"),
        "elapsed_ms": metrics.get("elapsed_ms"),
        "token_usage": metrics.get("token_usage"),
    }
    done = ProDone.model_validate(
        {
            "contract_version": "assistant-v2",
            "agent_mode": "pro",
            "request_id": request.request_id,
            "run_id": request.run_id,
            "thread_id": request.thread_id,
            "answer": validated.get("answer"),
            "product_refs": refs,
            "requirements": requirements,
            "stop_reason": validated.get("stop_reason"),
            "metrics": safe_metrics,
        }
    )
    return done.model_dump()


def _run_and_validate(
    request: ProChatRequest,
    *,
    stop_requested=lambda: False,
    progress_callback=lambda _tool, _stage: None,
    inbound_token: str | None = None,
) -> dict[str, Any]:
    """Shared synchronous core used by both v2 endpoints."""
    java_client = build_java_tool_client(request, inbound_token)
    context = _context_for_request(
        request,
        stop_requested=stop_requested,
        progress_callback=progress_callback,
        java_client=java_client,
    )

    def execute(action, *, timeout):
        return _execute_tool_with_progress(
            action,
            timeout=timeout,
            java_client=java_client,
            progress_callback=progress_callback,
        )

    result = run_pro_agent(
        request.query,
        execute_tool=execute,
        context=context,
    )
    return _safe_done(request, result, context=context)


def _safe_error(run_id: str, error: BaseException) -> dict[str, Any]:
    """Map internal failures to a stable, non-sensitive v2 error payload."""
    reason = getattr(error, "reason", None)
    if isinstance(error, ValueError):
        code, message = _ERROR_MESSAGES["unsafe_model_output"]
    else:
        code, message = _ERROR_MESSAGES.get(reason, _ERROR_MESSAGES["internal_error"])
    return {"code": code, "message": message, "run_id": run_id}


def _progress_callback_for_queue(event_queue: queue.Queue, run_id: str, stop_requested):
    """Return a sequence-bearing callback that can be called by the worker."""
    lock = threading.Lock()
    sequence = 0

    def emit(tool: str, stage: str) -> None:
        nonlocal sequence
        if stop_requested() or tool not in _KNOWN_TOOLS or stage not in {"started", "completed"}:
            return
        with lock:
            sequence += 1
            payload = {
                "run_id": run_id,
                "sequence": sequence,
                "tool": tool,
                "stage": stage,
                "message": _PROGRESS_MESSAGES[tool],
            }
        while not stop_requested():
            try:
                event_queue.put(("progress", payload), timeout=0.05)
                return
            except queue.Full:
                continue

    return emit


def _put_worker_event(event_queue: queue.Queue, item: tuple[str, Any], stop_requested) -> bool:
    """Put a worker event without blocking forever after a client disconnects."""
    while not stop_requested():
        try:
            event_queue.put(item, timeout=0.05)
            return True
        except queue.Full:
            continue
    return False


async def _is_disconnected(request: Request) -> bool:
    """Treat a broken request probe as a disconnect and cancel safely."""
    try:
        return await request.is_disconnected()
    except Exception:
        return True


async def generate_pro_chat_stream(
    chat_request: ProChatRequest,
    http_request: Request,
    *,
    inbound_token: str | None = None,
):
    """Run Pro in a bounded worker and publish progress, tokens, then one done."""
    event_queue: queue.Queue = queue.Queue(maxsize=_MAX_STREAM_QUEUE)
    stop_event = threading.Event()
    progress_callback = _progress_callback_for_queue(
        event_queue,
        chat_request.run_id,
        stop_event.is_set,
    )

    def worker() -> None:
        try:
            done = _run_and_validate(
                chat_request,
                stop_requested=stop_event.is_set,
                progress_callback=progress_callback,
                inbound_token=inbound_token,
            )
            if stop_event.is_set():
                return
            _put_worker_event(event_queue, ("done", done), stop_event.is_set)
        except Exception as error:
            if stop_event.is_set():
                return
            try:
                _put_worker_event(
                    event_queue,
                    ("error", _safe_error(chat_request.run_id, error)),
                    stop_event.is_set,
                )
            except queue.Full:
                return

    worker_thread = threading.Thread(target=worker, name="pro-v2-stream-worker", daemon=True)
    worker_thread.start()
    terminal_sent = False
    try:
        while True:
            if await _is_disconnected(http_request):
                stop_event.set()
                return
            try:
                kind, payload = await asyncio.to_thread(event_queue.get, True, 0.1)
            except (queue.Empty, FutureTimeoutError):
                continue
            if kind == "progress":
                yield build_pro_progress_event(payload)
                continue
            if kind == "done":
                for chunk in iter_answer_chunks(payload["answer"]):
                    if stop_event.is_set():
                        return
                    yield build_pro_token_event(chunk)
                if stop_event.is_set():
                    return
                yield build_pro_done_event(payload)
                terminal_sent = True
                return
            if kind == "error":
                yield build_pro_error_event(payload)
                terminal_sent = True
                return
    finally:
        if not terminal_sent:
            stop_event.set()


@router.post("/chat")
async def pro_chat(
    chat_request: ProChatRequest,
    request: Request,
):
    """Return one validated internal Pro v2 result for Java."""
    inbound_token = request.headers.get("X-Internal-Token")
    try:
        payload = await asyncio.to_thread(
            _run_and_validate,
            chat_request,
            inbound_token=inbound_token,
        )
    except Exception as error:
        return JSONResponse(
            status_code=_ERROR_STATUS,
            content={"error": _safe_error(chat_request.run_id, error)["code"],
                     "message": _safe_error(chat_request.run_id, error)["message"],
                     "run_id": chat_request.run_id},
        )
    return payload


@router.post("/chat/stream")
async def pro_chat_stream(
    chat_request: ProChatRequest,
    request: Request,
):
    """Stream safe v2 progress and the validated final answer to Java."""
    return StreamingResponse(
        generate_pro_chat_stream(
            chat_request,
            request,
            inbound_token=request.headers.get("X-Internal-Token"),
        ),
        media_type="text/event-stream;charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


__all__ = [
    "ProChatRequest",
    "ProDone",
    "ProTokenUsage",
    "build_java_tool_client",
    "generate_pro_chat_stream",
    "router",
]
