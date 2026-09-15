"""Run-local Pro graph. Results are internal proposals, never public approvals."""

import json
import time
from collections.abc import Mapping
from decimal import Decimal
from math import isfinite
from uuid import uuid4
from langgraph.graph import StateGraph, END
from clothing_assistant.config_data import get_pro_limits
from clothing_assistant.infrastructure.llm_client import DependencyError, classify_dependency_error
from clothing_assistant.infrastructure.java_tool_client import ToolResult, tool_result
from .decision import decide as default_decide, plain, call_with_deadline
from .schemas import parse_action, Requirement
from .state import new_state, _freeze, apply_requirement_updates
from .tools import execute_tool as default_tool, LOCAL_SOURCES

FILTERS = {"category", "style", "season", "material", "fit", "gender", "budget_max"}
PROFILE = {
    "height_cm",
    "weight_kg",
    "gender",
    "preferred_fit",
    "preferred_styles",
    "preferred_colors",
    "disliked_colors",
    "preferred_categories",
    "budget_min",
    "budget_max",
}


def run_pro_agent(
    query: str,
    *,
    decide=None,
    execute_tool=None,
    clock=time.monotonic,
    limits: Mapping | None = None,
    context: Mapping | None = None,
) -> dict:
    """Inject decide(snapshot, timeout=...) and execute_tool(action, timeout=...).

    Callbacks must honor their timeout; production adapters enforce I/O deadlines.
    No checkpoints, credentials, raw exceptions, or model reasoning are retained.

    Args:
        query: Entire original compound request.
        decide: One model attempt receiving an immutable snapshot and timeout.
        execute_tool: Validated action runner receiving the remaining tool timeout.
        clock: Monotonic seconds provider, injectable for deadline tests.
        limits: Optional lower model_calls, tool_calls and total_seconds bounds.
        context: Trusted Java context; runtime handles are excluded from snapshots.

    Returns:
        Internal proposal, evidence, unfinished requirements and actual metrics.
        Business validation is deliberately pending until Task7.

    Raises:
        ValueError: If a limit is invalid.
    """
    context = context or {}
    bounds = get_pro_limits()
    for key, value in (limits or {}).items():
        if (
            key not in bounds
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid Pro limits")
        if key != "total_seconds" and not isinstance(value, int):
            raise ValueError("Count limits must be integers")
        bounds[key] = min(bounds[key], value)
    started = clock()
    deadline = started + bounds["total_seconds"]
    cancelled = context.get("stop_requested", lambda: False)
    filters = {
        k: v
        for k, v in context.get("explicit_filters", {}).items()
        if k in FILTERS and v is not None
    }
    state = new_state(context.get("run_id") or str(uuid4()), query, filters)
    state["requirements"].insert(0, Requirement(id="query", text=query[:500], source="user:query"))
    observations = []
    cache = {}
    usage = {}
    usage_complete = True
    question = None
    metrics = {"decisions": 0, "tool_calls": 0}
    current = {}
    stop = None
    repair_used = False
    retry_used = False
    if decide is None:

        def decide(snapshot, *, timeout):
            return default_decide(snapshot, timeout=timeout, stop_requested=cancelled)

    if execute_tool is None:

        def execute_tool(action, *, timeout):
            tool_deadline = clock() + timeout

            def invoke_tool():
                if cancelled():
                    raise DependencyError("tool", "cancelled", False)
                remaining = tool_deadline - clock()
                if remaining <= 0:
                    raise TimeoutError()
                return default_tool(
                    action.tool,
                    action.arguments.model_dump(exclude_none=True),
                    java_client=context.get("java_client"),
                    timeout=remaining,
                )

            return call_with_deadline(invoke_tool, timeout=timeout)

    def guard():
        if cancelled():
            return "cancelled"
        if clock() >= deadline:
            return "budget_exhausted"
        return None

    def safe_profile_value(value):
        return (
            value is None
            or isinstance(value, (str, int, float))
            or (isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value))
        )

    def safe_observation(value):
        if isinstance(value, Mapping):
            return {
                k: safe_observation(v)
                for k, v in value.items()
                if str(k).lower().replace("_", "").replace("-", "")
                not in {
                    "userid",
                    "token",
                    "runtoken",
                    "toolruntoken",
                    "internaltoken",
                    "authorization",
                    "apikey",
                    "password",
                    "credentials",
                    "xinternaltoken",
                    "xproruntoken",
                }
            }
        if isinstance(value, (list, tuple)):
            return [safe_observation(v) for v in value]
        return value

    def snapshot():
        return _freeze(
            {
                "query": query,
                "hard_constraints": state["hard_constraints"],
                "hard_constraint_sources": state["hard_constraint_sources"],
                "requirements": [r.model_dump() for r in state["requirements"]],
                "observations": observations,
                "user_context": {
                    k: v
                    for k, v in context.get("user_context", {}).items()
                    if k in PROFILE and safe_profile_value(v)
                },
                "chat_history": [
                    {
                        k: item[k]
                        for k in ("user_query", "assistant_answer")
                        if isinstance(item.get(k), str)
                    }
                    for item in context.get("chat_history", [])
                    if isinstance(item, Mapping)
                ],
                "current_product_refs": [
                    {
                        k: item[k]
                        for k in ("spu_id", "sku_id")
                        if isinstance(item.get(k), int)
                        and not isinstance(item[k], bool)
                        and item[k] > 0
                    }
                    for item in context.get("current_product_refs", [])
                    if isinstance(item, Mapping)
                ],
                "repair_requested": repair_used,
            }
        )

    def decision_node(_):
        nonlocal stop, repair_used, retry_used, usage_complete, question
        stop = guard()
        if stop:
            return {}
        if metrics["decisions"] >= bounds["model_calls"]:
            stop = "budget_exhausted"
            return {}
        metrics["decisions"] += 1
        try:
            response = decide(snapshot(), timeout=deadline - clock())
        except Exception as error:
            usage_complete = False
            stop = guard()
            if stop:
                return {}
            classified = (
                error if isinstance(error, DependencyError) else classify_dependency_error(error)
            )
            if classified.retryable and not retry_used:
                retry_used = True
                current.clear()
                return {}
            stop = "dependency_unavailable"
            return {}
        metadata = getattr(response, "usage_metadata", None)
        token_keys = ("input_tokens", "output_tokens", "total_tokens")
        if isinstance(metadata, Mapping) and all(
            isinstance(metadata.get(key), int)
            and not isinstance(metadata[key], bool)
            and metadata[key] >= 0
            for key in token_keys
        ):
            for key in token_keys:
                usage[key] = usage.get(key, 0) + metadata[key]
        else:
            usage_complete = False
        stop = guard()
        if stop:
            return {}
        try:
            payload = getattr(response, "content", response)
            if isinstance(payload, str):
                payload = json.loads(payload)
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            action = parse_action(payload)
            apply_requirement_updates(state, [u.model_dump() for u in action.requirement_updates])
        except (ValueError, TypeError):
            if not repair_used:
                repair_used = True
                current.clear()
                return {}
            stop = "invalid_response"
            return {}
        if action.kind == "clarify":
            question = action.question
            state["requirements"][0] = state["requirements"][0].model_copy(
                update={"status": "needs_input"}
            )
        current["action"] = action
        return {}

    def execute_node(_):
        nonlocal stop, retry_used, question
        stop = guard()
        if stop:
            return {}
        action = current["action"]
        args = action.arguments.model_dump(exclude_none=True)
        if action.tool == "search_products":
            for key, value in filters.items():
                if key in args and (
                    Decimal(args[key]) > Decimal(value)
                    if key == "budget_max"
                    else args[key] != value
                ):
                    stop = "invalid_response"
                    return {}
                if key not in args:
                    args[key] = value
            action = parse_action({"kind": "tool", "tool": action.tool, "arguments": args})
        key = (action.tool, json.dumps(args, sort_keys=True))
        cached = cache.get(key)
        if cached and (action.tool != "check_availability" or clock() - cached[0] < 15):
            observations.append(cached[1])
            return {}
        while True:
            stop = guard()
            if stop:
                return {}
            if metrics["tool_calls"] >= bounds["tool_calls"]:
                stop = "budget_exhausted"
                return {}
            metrics["tool_calls"] += 1
            retryable = None
            try:
                result = execute_tool(action, timeout=min(5.0, deadline - clock()))
                if not isinstance(result, ToolResult):
                    result = ToolResult(**result)
            except Exception as error:
                classified = (
                    error
                    if isinstance(error, DependencyError)
                    else classify_dependency_error(error, "tool")
                )
                retryable = classified.retryable
                result = tool_result(
                    LOCAL_SOURCES.get(action.tool, "java"),
                    "unavailable",
                    error_code=classified.reason,
                )
            observation = safe_observation(
                dict(result.model_dump(), tool=action.tool, arguments=args)
            )
            observations.append(observation)
            if result.status == "ok":
                state["evidence"].append(dict(observation, run_id=state["run_id"]))
                cache[key] = (clock(), observation)
            stop = guard()
            if stop:
                return {}
            if (
                result.status == "unavailable"
                and result.error_code
                in {
                    "timeout",
                    "connection_error",
                    "rate_limited",
                    "upstream_5xx",
                    "gateway_unavailable",
                    "dependency_unavailable",
                }
                and retryable is not False
                and not retry_used
            ):
                retry_used = True
                continue
            if result.status == "needs_input":
                stop = "needs_input"
                question = (
                    "请补充必要信息：" + "、".join(result.missing_fields)
                    if result.missing_fields
                    else "请补充完成请求所需的信息。"
                )
                state["requirements"][0] = state["requirements"][0].model_copy(
                    update={"status": "needs_input"}
                )
            return {}

    def route(_):
        if stop:
            return END
        action = current.get("action")
        if action is None:
            return "decide"
        return {"tool": "execute", "finish": "validate", "clarify": END}[action.kind]

    def validate_node(_):
        nonlocal stop
        stop = guard() or "validation_required"
        return {}

    graph = StateGraph(dict)
    graph.add_node("decide", decision_node)
    graph.add_node("execute", execute_node)
    graph.add_node("validate", validate_node)
    graph.set_entry_point("decide")
    graph.add_conditional_edges("decide", route)
    graph.add_conditional_edges("execute", lambda _: END if stop else "decide")
    graph.add_edge("validate", END)
    graph.compile().invoke({}, {"recursion_limit": 64})
    action = current.get("action")
    if stop is None:
        stop = "needs_input"
    return {
        "run_id": state["run_id"],
        "query": query,
        "public_approved": False,
        "stop_reason": stop,
        "proposal": action.model_dump() if stop == "validation_required" else None,
        "question": question if stop == "needs_input" else None,
        "requirements": [r.model_dump() for r in state["requirements"]],
        "evidence": plain(state["evidence"]),
        "observations": plain(observations),
        "metrics": dict(
            metrics,
            elapsed_ms=max(0, int((clock() - started) * 1000)),
            token_usage={
                key: usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")
            }
            if usage and usage_complete
            else None,
        ),
    }
