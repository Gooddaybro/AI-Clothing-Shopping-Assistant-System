"""Stable outer-provider substitutes for cross-service integration tests."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re


@dataclass(frozen=True)
class DeterministicMessage:
    """Minimal provider response compatible with LangChain message consumers."""

    content: str


class DeterministicChatModel:
    """Return stable chat and demand-parser responses without external network calls."""

    def __init__(self, *, demand_parser: bool = False):
        self._demand_parser = demand_parser

    def invoke(self, messages):
        content = self._demand_response(messages) if self._demand_parser else self._chat_response()
        return DeterministicMessage(content)

    def stream(self, messages):
        content = self.invoke(messages).content
        for start in range(0, len(content), 8):
            yield DeterministicMessage(content[start:start + 8])

    @staticmethod
    def _chat_response() -> str:
        return "集成测试确定性回复：已收到请求，请结合 Java 商品候选继续选择。"

    @staticmethod
    def _message_content(message) -> str:
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")

    def _demand_response(self, messages) -> str:
        raw_request = self._message_content(messages[-1]) if messages else ""
        try:
            request = json.loads(raw_request)
        except (TypeError, json.JSONDecodeError):
            request = {}
        current_message = str(
            request.get("currentMessage")
            or request.get("current_message")
            or request.get("message")
            or raw_request
        )

        slots = {}
        confidence = {}
        evidence = {}

        def add_slot(name, value, quotation):
            slots[name] = value
            confidence[name] = 1.0
            evidence[name] = [{"text": quotation, "source": "CURRENT_MESSAGE"}]

        if "通勤" in current_message:
            add_slot("scene", ["COMMUTE"], "通勤")
        if "简约" in current_message or "简洁" in current_message:
            quotation = "简约" if "简约" in current_message else "简洁"
            add_slot("style", ["MINIMAL"], quotation)
        budget = re.search(r"(?:预算|不超过|以内)[^0-9]{0,4}(\d+)", current_message)
        if budget:
            add_slot("budgetMax", int(budget.group(1)), budget.group(0))

        if not slots:
            return json.dumps(
                {
                    "schemaVersion": "1.0",
                    "action": "CLARIFY",
                    "slots": {},
                    "slotConfidence": {},
                    "evidence": {},
                    "needsClarification": True,
                    "clarificationSlot": "scene",
                    "clarificationCandidateValue": None,
                    "clarificationQuestion": "请补充穿着场景，例如通勤或日常。",
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "schemaVersion": "1.0",
                "action": "MERGE",
                "slots": slots,
                "slotConfidence": confidence,
                "evidence": evidence,
                "needsClarification": False,
                "clarificationSlot": None,
                "clarificationCandidateValue": None,
                "clarificationQuestion": None,
            },
            ensure_ascii=False,
        )


class DeterministicEmbeddings:
    """Generate repeatable normalized vectors from text content."""

    dimension = 16

    @classmethod
    def _vector(cls, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()[:cls.dimension]
        values = [(value - 127.5) / 127.5 for value in digest]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 12) for value in values]

    def embed_documents(self, texts):
        return [self._vector(str(text)) for text in texts]

    def embed_query(self, text):
        return self._vector(str(text))
