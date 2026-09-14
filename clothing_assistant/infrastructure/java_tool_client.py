"""Client for the run-scoped Java Pro tool gateway.

The Java service owns product, price, and inventory facts.  This module keeps
that boundary narrow: requests can only target the three actions exposed by
the v2 contract and always use the configured gateway URL and run credentials.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx


JAVA_TOOLS = frozenset({"search_products", "get_product_detail", "check_availability"})
TOOL_STATUSES = frozenset(
    {"ok", "empty", "needs_input", "unavailable", "no_evidence", "forbidden"}
)
TOOL_SOURCES = frozenset({"java", "rag", "policy", "size"})

_SEARCH_ARGUMENTS = frozenset(
    {"category", "style", "season", "material", "fit", "gender", "recall_text", "budget_max"}
)
_DETAIL_ARGUMENTS = frozenset({"spu_id"})
_AVAILABILITY_ARGUMENTS = frozenset({"spu_id", "color", "size"})
_MONEY_PATTERN = r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$"


class _FrozenList(tuple):
    """Tuple-backed list view that retains intuitive list comparisons."""

    __hash__ = tuple.__hash__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return super().__eq__(other)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, _FrozenList)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ToolResult(Mapping[str, Any]):
    """The common observation shape returned by every Pro tool adapter.

    ``ToolResult`` intentionally supports both attribute access and the small
    mapping interface used by existing agent code.  The mapping view is a
    compatibility convenience; it does not expose client credentials.
    """

    status: str
    data: Any = None
    evidence_id: str = ""
    source: str = "java"
    missing_fields: list[str] = field(default_factory=list)
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in TOOL_STATUSES:
            raise ValueError("invalid tool result status")
        if self.source not in TOOL_SOURCES:
            raise ValueError("invalid tool result source")
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("tool result evidence_id must be non-empty")
        if not isinstance(self.missing_fields, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in self.missing_fields
        ):
            raise ValueError("tool result missing_fields must be a list of strings")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise ValueError("tool result error_code must be a string or null")
        object.__setattr__(self, "data", _freeze(self.data))
        object.__setattr__(self, "missing_fields", _FrozenList(self.missing_fields))

    def model_dump(self, *, mode: str | None = None) -> dict[str, Any]:
        """Return the contract-shaped mapping used in JSON responses."""
        return {
            "status": self.status,
            "data": _thaw(self.data),
            "evidence_id": self.evidence_id,
            "source": self.source,
            "missing_fields": _thaw(self.missing_fields),
            "error_code": self.error_code,
        }

    def __getitem__(self, key: str) -> Any:
        return self.model_dump()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Read a contract field using the mapping-style API."""
        return self.model_dump().get(key, default)

    def keys(self) -> Iterator[str]:
        return iter(self.model_dump())

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __len__(self) -> int:
        return 6

    def items(self):
        return self.model_dump().items()

    def __repr__(self) -> str:
        return (
            "ToolResult(status={!r}, data={!r}, evidence_id={!r}, source={!r}, "
            "missing_fields={!r}, error_code={!r})"
        ).format(
            self.status,
            self.data,
            self.evidence_id,
            self.source,
            self.missing_fields,
            self.error_code,
        )


def tool_result(
    source: str,
    status: str,
    data: Any = None,
    *,
    evidence_id: str | None = None,
    missing_fields: list[str] | None = None,
    error_code: str | None = None,
) -> ToolResult:
    """Construct a validated result and generate a non-sensitive evidence id."""
    if evidence_id is None or not isinstance(evidence_id, str) or not evidence_id:
        evidence_id = f"ev-{source}-{uuid4().hex}"
    return ToolResult(
        status=status,
        data=data,
        evidence_id=evidence_id,
        source=source,
        missing_fields=list(missing_fields or []),
        error_code=error_code,
    )


def _validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip() or base_url != base_url.strip():
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if any(character.isspace() for character in base_url):
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        # Accessing port validates malformed port syntax in urllib.
        parsed.port
    except ValueError:
        raise ValueError("base_url must be an absolute HTTP(S) URL") from None
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain query or fragment")
    return base_url.rstrip("/")


def _validate_action_arguments(name: str, arguments: object) -> None:
    if not isinstance(name, str) or name not in JAVA_TOOLS:
        raise ValueError("unsupported Java tool")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")

    allowed = {
        "search_products": _SEARCH_ARGUMENTS,
        "get_product_detail": _DETAIL_ARGUMENTS,
        "check_availability": _AVAILABILITY_ARGUMENTS,
    }[name]
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError("tool arguments contain unsupported fields")

    if name == "search_products":
        for key, value in arguments.items():
            if key == "budget_max":
                if (
                    not isinstance(value, str)
                    or len(value) > 20
                    or re.fullmatch(_MONEY_PATTERN, value) is None
                ):
                    raise ValueError("budget_max must be a decimal string")
            elif value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > 100
            ):
                raise ValueError("search filters must be bounded non-empty strings")
        return

    spu_id = arguments.get("spu_id")
    if not isinstance(spu_id, int) or isinstance(spu_id, bool) or spu_id <= 0:
        raise ValueError("spu_id must be a positive integer")
    if name == "check_availability":
        for key in ("color", "size"):
            value = arguments.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 100:
                raise ValueError("color and size must be bounded non-empty strings")


class JavaToolClient:
    """Call only the fixed, authenticated Java Pro tool gateway."""

    def __init__(
        self,
        base_url: str,
        internal_token: str,
        run_id: str,
        run_token: str,
        *,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        if not isinstance(internal_token, str) or not internal_token.strip():
            raise ValueError("internal token is required")
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or run_id != run_id.strip()
            or any(character.isspace() for character in run_id)
        ):
            raise ValueError("run id is required")
        if not isinstance(run_token, str) or not run_token.strip():
            raise ValueError("run token is required")
        try:
            configured_timeout = float(timeout)
        except (TypeError, ValueError):
            raise ValueError("timeout must be positive") from None
        if configured_timeout <= 0:
            raise ValueError("timeout must be positive")

        # Credentials are kept private and are never interpolated into errors,
        # repr(), URLs, or result data.
        self._headers = {
            "X-Internal-Token": internal_token,
            "X-Pro-Run-Token": run_token,
        }
        self._run_id = run_id
        self._transport = transport
        self._timeout = min(configured_timeout, 5.0)

    def __repr__(self) -> str:
        return f"JavaToolClient(base_url={self._base_url!r}, run_id={self._run_id!r})"

    def _request_url(self, name: str) -> str:
        encoded_run_id = quote(self._run_id, safe="")
        return f"{self._base_url}/internal/assistant/runs/{encoded_run_id}/tools/{name}"

    def call_tool(self, name: str, arguments: Mapping[str, Any], *, timeout: float | None = None) -> ToolResult:
        """Strict convenience API that rejects unsupported actions before HTTP."""
        _validate_action_arguments(name, arguments)
        return self.call(name, arguments, timeout=timeout)

    def call(
        self,
        name: str,
        arguments: Mapping[str, Any],
        timeout: float | None = None,
    ) -> ToolResult:
        """Call a whitelisted action and map transport failures safely."""
        try:
            _validate_action_arguments(name, arguments)
        except ValueError:
            return tool_result("java", "forbidden", error_code="invalid_arguments")

        request_timeout = self._timeout if timeout is None else timeout
        try:
            request_timeout = float(request_timeout)
        except (TypeError, ValueError):
            request_timeout = 0
        if not isfinite(request_timeout) or request_timeout <= 0:
            return tool_result("java", "unavailable", error_code="deadline_exceeded")
        request_timeout = min(request_timeout, 5.0)

        try:
            with httpx.Client(
                transport=self._transport,
                timeout=request_timeout,
                follow_redirects=False,
            ) as client:
                response = client.post(
                    self._request_url(name),
                    headers=self._headers,
                    json=dict(arguments),
                )
        except httpx.TimeoutException:
            return tool_result("java", "unavailable", error_code="timeout")
        except httpx.HTTPError:
            return tool_result("java", "unavailable", error_code="connection_error")
        except Exception:
            # Transport implementations are external code.  Do not let their
            # exception text echo credentials into logs or model observations.
            return tool_result("java", "unavailable", error_code="gateway_unavailable")

        if response.status_code in {401, 403}:
            return tool_result("java", "forbidden", error_code="forbidden")
        if response.status_code == 429:
            return tool_result("java", "unavailable", error_code="rate_limited")
        if response.status_code >= 500:
            return tool_result("java", "unavailable", error_code="upstream_5xx")
        if response.status_code != 200:
            return tool_result("java", "forbidden", error_code="gateway_rejected")

        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            return tool_result("java", "unavailable", error_code="invalid_response")

        # Some older internal clients still wrap the gateway result in a
        # code/data envelope.  Accept it while preserving the v2 raw shape.
        if isinstance(body, dict) and "code" in body and "data" in body:
            code = body.get("code")
            if code not in {0, 200, "0", "200"}:
                return tool_result("java", "unavailable", error_code="gateway_error")
            body = body.get("data")

        if not isinstance(body, dict) or body.get("status") not in TOOL_STATUSES:
            return tool_result("java", "unavailable", error_code="invalid_response")

        status = body["status"]
        missing_fields = body.get("missing_fields", [])
        error_code = body.get("error_code")
        if not isinstance(missing_fields, list) or any(
            not isinstance(value, str) or not value for value in missing_fields
        ):
            return tool_result("java", "unavailable", error_code="invalid_response")
        if error_code is not None and not isinstance(error_code, str):
            return tool_result("java", "unavailable", error_code="invalid_response")

        # Java is the only permitted source for these actions.  The server's
        # optional source field is checked when supplied, then normalized.
        response_source = body.get("source", "java")
        if response_source not in {None, "java"}:
            return tool_result("java", "unavailable", error_code="invalid_response")
        evidence_id = body.get("evidence_id")
        if status == "ok" and (
            not isinstance(evidence_id, str) or not evidence_id.strip()
        ):
            return tool_result("java", "unavailable", error_code="invalid_tool_response")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            evidence_id = None
        return tool_result(
            "java",
            status,
            body.get("data"),
            evidence_id=evidence_id,
            missing_fields=missing_fields,
            error_code=error_code,
        )


__all__ = ["JAVA_TOOLS", "TOOL_STATUSES", "ToolResult", "JavaToolClient", "tool_result"]
