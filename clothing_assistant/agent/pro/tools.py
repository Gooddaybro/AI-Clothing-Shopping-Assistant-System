"""Adapters that expose Java facts and local knowledge as bounded Pro tools.

The adapter layer is deliberately independent from the Lite pipeline.  It
validates model arguments before dispatch, keeps the Java gateway fixed, and
turns every dependency observation into the shared ``ToolResult`` shape.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from math import isfinite
from typing import Any

from pydantic import ValidationError

from clothing_assistant.agent.pro.schemas import (
    AvailabilityArguments,
    ProductDetailArguments,
    ProductKnowledgeArguments,
    PolicyArguments,
    SearchProductsArguments,
    SizeArguments,
)
from clothing_assistant.infrastructure.java_tool_client import (
    JAVA_TOOLS,
    TOOL_STATUSES,
    JavaToolClient,
    ToolResult,
    tool_result,
)


TOOL_ARGUMENT_MODELS = {
    "search_products": SearchProductsArguments,
    "get_product_detail": ProductDetailArguments,
    "check_availability": AvailabilityArguments,
    "search_product_knowledge": ProductKnowledgeArguments,
    "search_policy": PolicyArguments,
    "recommend_size": SizeArguments,
}
LOCAL_SOURCES = {
    "search_product_knowledge": "rag",
    "search_policy": "policy",
    "recommend_size": "size",
}
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 5.0
MAX_BLOCKED_ADAPTERS = 8
_ADAPTER_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_BLOCKED_ADAPTERS,
    thread_name_prefix="pro-tool-adapter",
)


class _LocalTimeout(Exception):
    """Internal marker for a local runner that exceeded its tool deadline."""


def _source_for(name: str) -> str:
    return "java" if name in JAVA_TOOLS else LOCAL_SOURCES.get(name, "java")


def _invalid_result(source: str, fields: list[str], error_code: str) -> ToolResult:
    return tool_result(source, "needs_input", missing_fields=fields, error_code=error_code)


def _argument_model(name: str, arguments: object):
    if not isinstance(name, str):
        return None, tool_result("java", "forbidden", error_code="unknown_tool")
    model_type = TOOL_ARGUMENT_MODELS.get(name)
    if model_type is None:
        return None, tool_result("java", "forbidden", error_code="unknown_tool")
    try:
        return model_type.model_validate(arguments), None
    except ValidationError as error:
        fields = sorted(
            {
                str(item["loc"][0]) if item.get("loc") else "arguments"
                for item in error.errors()
            }
        )
        if name == "recommend_size" and {"height", "weight"}.intersection(
            arguments if isinstance(arguments, Mapping) else set()
        ):
            return None, _invalid_result("size", fields, "ambiguous_measurement_units")
        return None, _invalid_result(_source_for(name), fields, "invalid_arguments")


def _bounded_call(runner: Callable[[str], Any], query: str, timeout: float) -> Any:
    """Run a legacy synchronous capability without allowing it to block Pro."""
    # A synchronous dependency cannot be forcefully cancelled once it is
    # blocked in I/O.  Reuse a bounded pool so repeated deadlines cannot create
    # an unbounded collection of background threads.  A timed-out future is
    # cancelled when possible and its worker is reclaimed when the runner
    # returns.
    future = _ADAPTER_EXECUTOR.submit(runner, query)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as error:
        future.cancel()
        raise _LocalTimeout from error


def _timeout_value(timeout: float | None) -> float | None:
    if timeout is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return None
    if not isfinite(value) or value <= 0:
        return None
    return min(value, MAX_TIMEOUT_SECONDS)


def _coerce_java_result(result: Any) -> ToolResult:
    if isinstance(result, ToolResult):
        if result.source != "java":
            return tool_result("java", "unavailable", error_code="invalid_tool_response")
        return result
    if not isinstance(result, Mapping) or result.get("status") not in TOOL_STATUSES:
        return tool_result("java", "unavailable", error_code="invalid_tool_response")
    missing_fields = result.get("missing_fields", [])
    if not isinstance(missing_fields, list) or any(
        not isinstance(field, str) or not field for field in missing_fields
    ):
        return tool_result("java", "unavailable", error_code="invalid_tool_response")
    error_code = result.get("error_code")
    if error_code is not None and not isinstance(error_code, str):
        return tool_result("java", "unavailable", error_code="invalid_tool_response")
    response_source = result.get("source", "java")
    if response_source not in {None, "java"}:
        return tool_result("java", "unavailable", error_code="invalid_tool_response")
    evidence_id = result.get("evidence_id")
    if result["status"] == "ok" and (
        not isinstance(evidence_id, str) or not evidence_id.strip()
    ):
        return tool_result("java", "unavailable", error_code="invalid_tool_response")
    return tool_result(
        "java",
        result["status"],
        result.get("data"),
        evidence_id=evidence_id if isinstance(evidence_id, str) and evidence_id else None,
        missing_fields=missing_fields,
        error_code=error_code,
    )


def _first_evidence_id(chunks: list[Mapping[str, Any]]) -> str | None:
    for chunk in chunks:
        for key in ("evidence_id", "chunk_id"):
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _run_knowledge(args: ProductKnowledgeArguments, runner: Callable[[str], Any] | None,
                   timeout: float) -> ToolResult:
    if runner is None:
        from clothing_assistant.tools.rag_tool import run_rag_tool

        runner = run_rag_tool
    query = args.query
    if args.spu_id is not None:
        query = f"{query}。目标商品范围：SPU {args.spu_id}。"
    result = _bounded_call(runner, query, timeout)
    if not isinstance(result, Mapping):
        return tool_result("rag", "unavailable", error_code="invalid_tool_response")
    rag_meta = result.get("rag_meta") or {}
    if isinstance(rag_meta, Mapping) and rag_meta.get("degraded_reason"):
        return tool_result("rag", "unavailable", error_code="retrieval_unavailable")
    chunks = result.get("retrieved_chunks") or []
    if not isinstance(chunks, list) or any(not isinstance(chunk, Mapping) for chunk in chunks):
        return tool_result("rag", "unavailable", error_code="invalid_tool_response")
    data = {
        "retrieved_chunks": chunks,
        "scope": f"spu:{args.spu_id}" if args.spu_id is not None else "general",
        "product_scope_verified": False,
    }
    if args.spu_id is not None:
        data["limitation"] = "检索结果未提供该 SPU 的认证属性，只能作为一般知识参考。"
    return tool_result(
        "rag",
        "ok" if chunks else "no_evidence",
        data,
        evidence_id=_first_evidence_id(chunks),
    )


def _run_policy(args: PolicyArguments, runner: Callable[[str], Any] | None,
                timeout: float) -> ToolResult:
    if runner is None:
        from clothing_assistant.tools.policy_tool import run_policy_tool

        runner = run_policy_tool
    result = _bounded_call(runner, args.query, timeout)
    if not isinstance(result, Mapping):
        return tool_result("policy", "unavailable", error_code="invalid_tool_response")
    chunks = result.get("policy_chunks") or []
    has_source = result.get("has_policy_source") is True and bool(chunks)
    if not isinstance(chunks, list) or any(not isinstance(chunk, Mapping) for chunk in chunks):
        return tool_result("policy", "unavailable", error_code="invalid_tool_response")
    data = {"policy_chunks": chunks, "policy_answer": result.get("policy_answer")}
    return tool_result(
        "policy",
        "ok" if has_source else "no_evidence",
        data,
        evidence_id=_first_evidence_id(chunks),
    )


def _run_size(args: SizeArguments, runner: Callable[[str], Any] | None,
              timeout: float) -> ToolResult:
    if runner is None:
        from clothing_assistant.tools.size_tool import run_size_tool

        runner = run_size_tool
    query = f"身高{args.height_cm:g}cm 体重{args.weight_kg:g}kg"
    if args.preferred_fit:
        fit_label = {
            "relaxed": "宽松",
            "regular": "合身",
            "tight": "修身",
        }.get(args.preferred_fit, args.preferred_fit)
        query += f" 穿着偏好{fit_label}"
    if args.spu_id is not None:
        query += f" 商品范围SPU {args.spu_id}"
    result = _bounded_call(runner, query, timeout)
    if not isinstance(result, Mapping):
        return tool_result("size", "unavailable", error_code="invalid_tool_response")
    data = dict(result)
    chart_evidence = (
        data.get("basis") == "product_chart"
        and data.get("product_chart_available") is True
        and data.get("product_chart_evidence") is True
        and isinstance(data.get("evidence_id"), str)
        and bool(data["evidence_id"])
    )
    data["basis"] = "product_chart" if chart_evidence else "generic_rule"
    data["product_chart_available"] = chart_evidence
    if not chart_evidence:
        data["limitation"] = "当前仅使用通用身高体重规则，未核验目标商品尺码表。"
    status = "ok" if data.get("recommended_size") else "no_evidence"
    return tool_result(
        "size",
        status,
        data,
        evidence_id=data.get("evidence_id") if chart_evidence else None,
    )


def execute_tool(
    name: str,
    arguments: object,
    *,
    java_client: JavaToolClient | None,
    knowledge_runner: Callable[[str], Any] | None = None,
    policy_runner: Callable[[str], Any] | None = None,
    size_runner: Callable[[str], Any] | None = None,
    timeout: float | None = None,
) -> ToolResult:
    """Validate and execute one of the six registered Pro tools."""
    parsed, invalid = _argument_model(name, arguments)
    if invalid is not None:
        return invalid
    bounded_timeout = _timeout_value(timeout)
    source = _source_for(name)
    if bounded_timeout is None:
        return tool_result(source, "unavailable", error_code="deadline_exceeded")
    payload = parsed.model_dump(exclude_none=True)

    if name in JAVA_TOOLS:
        if java_client is None:
            return tool_result("java", "unavailable", error_code="missing_dependency")
        try:
            return _coerce_java_result(
                java_client.call_tool(name, payload, timeout=bounded_timeout)
            )
        except ValueError:
            return tool_result("java", "forbidden", error_code="invalid_arguments")
        except Exception:
            return tool_result("java", "unavailable", error_code="gateway_unavailable")

    try:
        if name == "search_product_knowledge":
            return _run_knowledge(parsed, knowledge_runner, bounded_timeout)
        if name == "search_policy":
            return _run_policy(parsed, policy_runner, bounded_timeout)
        return _run_size(parsed, size_runner, bounded_timeout)
    except _LocalTimeout:
        return tool_result(source, "unavailable", error_code="timeout")
    except Exception:
        return tool_result(source, "unavailable", error_code="dependency_unavailable")


__all__ = ["TOOL_ARGUMENT_MODELS", "ToolResult", "execute_tool"]
