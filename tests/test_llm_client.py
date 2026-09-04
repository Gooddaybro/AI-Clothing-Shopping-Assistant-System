import os
import unittest
from unittest.mock import patch

import httpx
import openai

from clothing_assistant.config_data import CHAT_MODEL_NAME, CHAT_TEMPERATURE, KIMI_BASE_URL
from clothing_assistant.infrastructure import llm_client


class KimiChatClientTests(unittest.TestCase):
    def test_kimi_k25_uses_the_provider_required_temperature(self):
        self.assertEqual(CHAT_TEMPERATURE, 1)

    def test_kimi_client_uses_moonshot_openai_compatibility(self):
        chat_client_type = getattr(llm_client, "ChatOpenAI", None)
        self.assertIsNotNone(chat_client_type)

        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "test-moonshot-key"}, clear=True), patch(
            "clothing_assistant.infrastructure.llm_client.ChatOpenAI"
        ) as chat_client:
            llm_client.get_chat_model()

        chat_client.assert_called_once_with(
            model=CHAT_MODEL_NAME,
            temperature=CHAT_TEMPERATURE,
            api_key="test-moonshot-key",
            base_url=KIMI_BASE_URL,
            request_timeout=30.0,
            max_retries=0,
        )

    def test_kimi_client_requires_moonshot_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MOONSHOT_API_KEY"):
                llm_client.get_chat_model()

    def test_integration_runtime_uses_deterministic_model_without_provider_key(self):
        with patch.dict(
            os.environ,
            {"AI_RUNTIME_ENV": "integration", "AI_DETERMINISTIC_PROVIDER": "true"},
            clear=True,
        ):
            model = llm_client.get_chat_model()
            response = model.invoke([{"role": "user", "content": "通勤穿搭"}])
            fragments = [chunk.content for chunk in model.stream([])]

        self.assertIn("集成测试", response.content)
        self.assertEqual("".join(fragments), response.content)


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeStream:
    def __init__(self, values):
        self.values = iter(values)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return FakeChunk(value)

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self, stream):
        self.provider_stream = stream

    def stream(self, _messages):
        return self.provider_stream


class FakeHttpError(RuntimeError):
    def __init__(self, status_code):
        super().__init__("provider detail must stay private")
        self.status_code = status_code


class LlmStreamingPolicyTests(unittest.TestCase):
    def test_native_openai_connection_errors_are_retryable(self):
        request = httpx.Request("POST", "https://example.test/chat/completions")
        cases = (
            (openai.APITimeoutError(request), "timeout"),
            (openai.APIConnectionError(request=request), "connection_error"),
        )

        for error, reason in cases:
            with self.subTest(reason=reason):
                classified = llm_client.classify_dependency_error(error)

            self.assertTrue(classified.retryable)
            self.assertEqual(classified.reason, reason)

    def test_stream_chat_content_yields_real_provider_fragments(self):
        provider_stream = FakeStream(["你", "好", ""])

        fragments = list(
            llm_client.stream_chat_content(
                ["message"],
                model_factory=lambda: FakeModel(provider_stream),
            )
        )

        self.assertEqual(fragments, ["你", "好"])
        self.assertTrue(provider_stream.closed)

    def test_stream_chat_content_retries_timeout_before_output(self):
        streams = [FakeStream([TimeoutError("private timeout")]), FakeStream(["ok"])]

        with patch("clothing_assistant.infrastructure.llm_client.get_llm_max_retries", return_value=1):
            fragments = list(
                llm_client.stream_chat_content(
                    ["message"],
                    model_factory=lambda: FakeModel(streams.pop(0)),
                    sleep=lambda _: None,
                )
            )

        self.assertEqual(fragments, ["ok"])
        self.assertEqual(streams, [])

    def test_stream_chat_content_does_not_retry_after_output(self):
        factory_calls = 0

        def model_factory():
            nonlocal factory_calls
            factory_calls += 1
            return FakeModel(FakeStream(["partial", TimeoutError("private timeout")]))

        stream = llm_client.stream_chat_content(
            ["message"],
            model_factory=model_factory,
            sleep=lambda _: None,
        )
        self.assertEqual(next(stream), "partial")
        with self.assertRaises(llm_client.DependencyError) as raised:
            next(stream)

        self.assertEqual(factory_calls, 1)
        self.assertEqual(raised.exception.reason, "timeout")
        self.assertNotIn("private timeout", str(raised.exception))

    def test_stream_chat_content_does_not_retry_non_retryable_4xx(self):
        factory_calls = 0

        def model_factory():
            nonlocal factory_calls
            factory_calls += 1
            return FakeModel(FakeStream([FakeHttpError(400)]))

        with self.assertRaises(llm_client.DependencyError) as raised:
            list(
                llm_client.stream_chat_content(
                    ["message"],
                    model_factory=model_factory,
                    sleep=lambda _: None,
                )
            )

        self.assertEqual(factory_calls, 1)
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(raised.exception.reason, "upstream_4xx")

    def test_stream_chat_content_stops_and_closes_provider_iterator(self):
        provider_stream = FakeStream(["must not be read"])

        fragments = list(
            llm_client.stream_chat_content(
                ["message"],
                model_factory=lambda: FakeModel(provider_stream),
                stop_requested=lambda: True,
            )
        )

        self.assertEqual(fragments, [])
        self.assertTrue(provider_stream.closed)


if __name__ == "__main__":
    unittest.main()
