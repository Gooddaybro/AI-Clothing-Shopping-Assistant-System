"""Behavior tests for the Pro tool adapters.

The fakes in this module are deliberately local.  Pro tool tests must never
open a real Java or embedding connection.
"""

import httpx
import pytest


def java_response(*, status="ok", data=None, evidence_id="java-ev-1", source="java",
                  missing_fields=None, error_code=None):
    return {
        "status": status,
        "data": data,
        "evidence_id": evidence_id,
        "source": source,
        "missing_fields": missing_fields or [],
        "error_code": error_code,
    }


def make_client(handler):
    from clothing_assistant.infrastructure.java_tool_client import JavaToolClient

    transport = httpx.MockTransport(handler)
    return JavaToolClient(
        "http://java.test",
        "internal-secret",
        "run-1",
        "run-secret",
        transport=transport,
    )


def test_java_client_uses_fixed_gateway_and_rejects_untrusted_routing_inputs():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=java_response(data=[{"spu_id": 1, "sku_id": 2}]),
            request=request,
        )

    client = make_client(handler)
    result = client.call_tool("search_products", {"category": "outerwear"})

    assert result.status == "ok"
    assert str(seen[0].url) == "http://java.test/internal/assistant/runs/run-1/tools/search_products"
    assert seen[0].headers["X-Internal-Token"] == "internal-secret"
    assert seen[0].headers["X-Pro-Run-Token"] == "run-secret"
    assert seen[0].read() == b'{"category":"outerwear"}'
    assert "internal-secret" not in repr(client)
    assert "run-secret" not in repr(client)

    with pytest.raises(ValueError):
        client.call_tool("search_policy", {"query": "returns"})
    with pytest.raises(ValueError):
        client.call_tool("search_products", {"user_id": 7})
    with pytest.raises(ValueError):
        client.call_tool("search_products", {"url": "https://attacker.test"})


@pytest.mark.parametrize("base_url", ["", "ftp://java.test", "http:///missing-host",
                                       "http://user:password@java.test"])
def test_java_client_rejects_unsafe_base_url_or_empty_credentials(base_url):
    from clothing_assistant.infrastructure.java_tool_client import JavaToolClient

    with pytest.raises(ValueError):
        JavaToolClient(base_url, "internal", "run-1", "run-secret")

    with pytest.raises(ValueError):
        JavaToolClient("http://java.test", "", "run-1", "run-secret")
    with pytest.raises(ValueError):
        JavaToolClient("http://java.test", "internal", "", "run-secret")
    with pytest.raises(ValueError):
        JavaToolClient("http://java.test", "internal", "run-1", "")


def test_http_timeout_becomes_unavailable_without_exposing_credentials():
    from clothing_assistant.agent.pro.tools import execute_tool

    def handler(request):
        raise httpx.ReadTimeout("upstream timed out", request=request)

    client = make_client(handler)
    result = execute_tool(
        "search_products",
        {"category": "outerwear"},
        java_client=client,
    )

    assert result.status == "unavailable"
    assert result.source == "java"
    assert result.error_code == "timeout"
    assert "internal-secret" not in str(result)
    assert "run-secret" not in str(result)


def test_zero_inventory_is_a_successful_java_observation():
    from clothing_assistant.agent.pro.tools import execute_tool

    def handler(request):
        return httpx.Response(
            200,
            json=java_response(
                data={"spu_id": 10, "sku_id": 20, "color": "black", "size": "L",
                      "available_stock": 0},
            ),
            request=request,
        )

    result = execute_tool(
        "check_availability",
        {"spu_id": 10, "color": "black", "size": "L"},
        java_client=make_client(handler),
    )

    assert result.status == "ok"
    assert result.data["available_stock"] == 0


def test_missing_sku_remains_empty_instead_of_becoming_dependency_failure():
    from clothing_assistant.agent.pro.tools import execute_tool

    def handler(request):
        return httpx.Response(
            200,
            json=java_response(status="empty", data=None, evidence_id="java-empty-1"),
            request=request,
        )

    result = execute_tool(
        "check_availability",
        {"spu_id": 10, "color": "black", "size": "L"},
        java_client=make_client(handler),
    )

    assert result.status == "empty"
    assert result.source == "java"


def test_product_knowledge_query_is_scoped_and_carries_source_evidence():
    from clothing_assistant.agent.pro.tools import execute_tool

    calls = []

    def knowledge_runner(query):
        calls.append(query)
        return {
            "retrieved_chunks": [{
                "chunk_id": "material-003",
                "file_name": "材质知识.txt",
                "content": "商品材质说明",
                "score": 0.1,
            }],
            "source_count": 1,
        }

    result = execute_tool(
        "search_product_knowledge",
        {"query": "怎么洗", "spu_id": 42},
        java_client=None,
        knowledge_runner=knowledge_runner,
    )

    assert calls and "目标商品范围" in calls[0] and "42" in calls[0]
    assert result.status == "ok"
    assert result.source == "rag"
    assert result.evidence_id == "material-003"


def test_policy_without_a_source_is_no_evidence():
    from clothing_assistant.agent.pro.tools import execute_tool

    result = execute_tool(
        "search_policy",
        {"query": "能退货吗"},
        java_client=None,
        policy_runner=lambda query: {
            "has_policy_source": False,
            "policy_chunks": [],
            "source_count": 0,
            "policy_answer": "暂无政策资料",
        },
    )

    assert result.status == "no_evidence"
    assert result.source == "policy"
    assert result.evidence_id


def test_size_adapter_uses_explicit_units_and_marks_generic_basis():
    from clothing_assistant.agent.pro.tools import execute_tool

    calls = []

    def size_runner(query):
        calls.append(query)
        return {
            "recommended_size": "L",
            "alternative": "XL",
            "reason": "通用规则命中",
            "match_type": "matched",
        }

    result = execute_tool(
        "recommend_size",
        {"height_cm": 175, "weight_kg": 70, "spu_id": 42},
        java_client=None,
        size_runner=size_runner,
    )

    assert calls == ["身高175cm 体重70kg 商品范围SPU 42"]
    assert result.status == "ok"
    assert result.source == "size"
    assert result.data["basis"] == "generic_rule"
    assert result.data["product_chart_available"] is False
    assert result.data["limitation"]


def test_bare_size_numbers_are_ambiguous_and_do_not_call_runner():
    from clothing_assistant.agent.pro.tools import execute_tool

    called = False

    def size_runner(query):
        nonlocal called
        called = True
        return {}

    result = execute_tool(
        "recommend_size",
        {"height": 175, "weight": 70},
        java_client=None,
        size_runner=size_runner,
    )

    assert result.status == "needs_input"
    assert result.error_code == "ambiguous_measurement_units"
    assert called is False


def test_unknown_tool_is_forbidden_before_any_dependency_call():
    from clothing_assistant.agent.pro.tools import execute_tool

    result = execute_tool("run_sql", {}, java_client=None)

    assert result.status == "forbidden"
    assert result.error_code == "unknown_tool"


def test_malformed_java_response_is_safe_unavailable():
    from clothing_assistant.infrastructure.java_tool_client import JavaToolClient

    client = make_client(lambda request: httpx.Response(200, text="not-json", request=request))

    result = client.call_tool("search_products", {})

    assert result.status == "unavailable"
    assert result.error_code == "invalid_response"


def test_java_success_without_evidence_is_not_accepted_as_authoritative():
    from clothing_assistant.infrastructure.java_tool_client import JavaToolClient

    client = make_client(
        lambda request: httpx.Response(
            200,
            json={"status": "ok", "data": {"spu_id": 1}},
            request=request,
        )
    )

    result = client.call_tool("search_products", {})

    assert result.status == "unavailable"
    assert result.error_code == "invalid_tool_response"


def test_tool_result_data_and_missing_fields_are_deeply_immutable():
    from clothing_assistant.agent.pro.tools import execute_tool

    result = execute_tool(
        "search_policy",
        {"query": "能退货吗"},
        java_client=None,
        policy_runner=lambda query: {
            "has_policy_source": True,
            "policy_chunks": [{"chunk_id": "policy-1", "content": "可退货"}],
        },
    )

    with pytest.raises(TypeError):
        result.data["policy_chunks"] = []
    with pytest.raises(TypeError):
        result.data["policy_chunks"][0]["content"] = "被篡改"
    with pytest.raises(AttributeError):
        result.missing_fields.append("another")

    copied = result.model_dump()
    copied["data"]["policy_chunks"][0]["content"] = "copy only"
    assert result.data["policy_chunks"][0]["content"] == "可退货"


def test_size_adapter_passes_preferred_fit_to_legacy_runner():
    from clothing_assistant.agent.pro.tools import execute_tool

    calls = []

    def runner(query):
        calls.append(query)
        return {"recommended_size": "L"}

    result = execute_tool(
        "recommend_size",
        {"height_cm": 175, "weight_kg": 70, "preferred_fit": "relaxed"},
        java_client=None,
        size_runner=runner,
    )

    assert result.status == "ok"
    assert calls == ["身高175cm 体重70kg 穿着偏好宽松"]


def test_spoofed_product_chart_claim_stays_generic():
    from clothing_assistant.agent.pro.tools import execute_tool

    result = execute_tool(
        "recommend_size",
        {"height_cm": 175, "weight_kg": 70},
        java_client=None,
        size_runner=lambda query: {
            "recommended_size": "L",
            "basis": "product_chart",
            "product_chart_available": True,
            "evidence_id": "model-made",
        },
    )

    assert result.data["basis"] == "generic_rule"
    assert result.data["product_chart_available"] is False
