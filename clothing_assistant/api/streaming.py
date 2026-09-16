"""SSE helpers for the Java-facing assistant streaming endpoint."""

import json
from collections.abc import Iterable
from typing import Any, Callable


class UnsafeStreamContent(RuntimeError):
    """Signal that buffered model text violates a deterministic fact rule."""


class SafeTokenBuffer:
    """Hold a suffix while releasing only cumulatively validated model text."""

    def __init__(self, tail_chars: int, validator: Callable[[str], str | None]):
        if tail_chars < 1:
            raise ValueError("tail_chars must be positive")
        self.tail_chars = tail_chars
        self.validator = validator
        self.text = ""
        self.emitted_text = ""

    def _validate(self) -> None:
        if self.validator(self.text):
            raise UnsafeStreamContent("stream content failed deterministic validation")

    def push(self, fragment: str) -> list[str]:
        if not fragment:
            return []

        self.text += fragment
        self._validate()
        safe_end = max(len(self.emitted_text), len(self.text) - self.tail_chars)
        safe_text = self.text[len(self.emitted_text):safe_end]
        if not safe_text:
            return []

        self.emitted_text += safe_text
        return [safe_text]

    def finish(self) -> list[str]:
        self._validate()
        remaining = self.text[len(self.emitted_text):]
        if not remaining:
            return []

        self.emitted_text += remaining
        return [remaining]


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    """Serialize one SSE event with a single-line JSON data payload."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def iter_answer_chunks(answer: str, chunk_size: int = 12) -> Iterable[str]:
    """Split final answer text into stable display chunks for SSE token events."""
    if not answer:
        return []

    for index in range(0, len(answer), chunk_size):
        yield answer[index:index + chunk_size]
 #   return [answer[index:index + chunk_size] for index in range(0, len(answer), chunk_size)]


def get_agent_intent(agent_result: dict[str, Any]) -> str:
    """Read the stable intent value from the existing agent debug payload."""
    debug = agent_result.get("debug") or {}
    intent_result = debug.get("intent_result") or {}
    return intent_result.get("intent") or "unknown"


def build_stream_done_payload(agent_result: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Build the Python-to-Java done payload defined by the shared v1 contract."""
    return {
        "request_id": request_id,
        "answer": agent_result.get("answer", ""),
        "intent": get_agent_intent(agent_result),
        "product_refs": agent_result.get("product_refs", []),
        "rejected_reasons": agent_result.get("rejected_reasons", {}),
    }


def iter_stream_events(agent_result: dict[str, Any], request_id: str) -> Iterable[str]:
    """Yield token events followed by the final done event."""
    answer = agent_result.get("answer") or ""
    for chunk in iter_answer_chunks(answer):
        yield format_sse_event("token", {"content": chunk})

    yield format_sse_event("done", build_stream_done_payload(agent_result, request_id))


def build_error_event(code: str, message: str) -> str:
    """Build a safe SSE error event without exposing internal exception details."""
    return format_sse_event("error", {"code": code, "message": message})


def build_pro_progress_event(payload: dict[str, Any]) -> str:
    """Build one v2 progress event from a server-selected safe payload."""
    return format_sse_event("progress", payload)


def build_pro_token_event(content: str) -> str:
    """Build one v2 token event; callers pass only validated answer text."""
    return format_sse_event("token", {"content": content})


def build_pro_done_event(payload: dict[str, Any]) -> str:
    """Build the v2 terminal done event from the validated result."""
    return format_sse_event("done", payload)


def build_pro_error_event(payload: dict[str, Any]) -> str:
    """Build one v2 terminal error event without adding internal details."""
    return format_sse_event("error", payload)
