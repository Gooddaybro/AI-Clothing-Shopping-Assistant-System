import os
import unittest
from unittest.mock import patch

from clothing_assistant.config_data import (
    get_llm_max_concurrency,
    get_llm_max_retries,
    get_llm_timeout_seconds,
    get_rag_timeout_seconds,
    get_stream_safety_tail_chars,
    is_deterministic_provider_enabled,
    is_debug_response_enabled,
)


class DebugResponseConfigurationTests(unittest.TestCase):
    def test_debug_response_enabled_only_for_normalized_true(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_debug_response_enabled())

            os.environ["DEBUG_RESPONSE_ENABLED"] = "false"
            self.assertFalse(is_debug_response_enabled())

            os.environ["DEBUG_RESPONSE_ENABLED"] = " TrUe "
            self.assertTrue(is_debug_response_enabled())

            os.environ["DEBUG_RESPONSE_ENABLED"] = "1"
            self.assertFalse(is_debug_response_enabled())


class PhaseTwoRuntimeConfigurationTests(unittest.TestCase):
    def test_phase_two_runtime_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_llm_timeout_seconds(), 30.0)
            self.assertEqual(get_llm_max_retries(), 2)
            self.assertEqual(get_llm_max_concurrency(), 8)
            self.assertEqual(get_rag_timeout_seconds(), 20.0)
            self.assertEqual(get_stream_safety_tail_chars(), 64)

    def test_phase_two_runtime_values_are_trimmed_and_parsed(self):
        with patch.dict(
            os.environ,
            {
                "LLM_TIMEOUT_SECONDS": " 12.5 ",
                "LLM_MAX_RETRIES": " 3 ",
                "LLM_MAX_CONCURRENCY": " 4 ",
                "RAG_TIMEOUT_SECONDS": " 7.25 ",
                "STREAM_SAFETY_TAIL_CHARS": " 96 ",
            },
            clear=True,
        ):
            self.assertEqual(get_llm_timeout_seconds(), 12.5)
            self.assertEqual(get_llm_max_retries(), 3)
            self.assertEqual(get_llm_max_concurrency(), 4)
            self.assertEqual(get_rag_timeout_seconds(), 7.25)
            self.assertEqual(get_stream_safety_tail_chars(), 96)

    def test_phase_two_runtime_rejects_invalid_values(self):
        cases = (
            ("LLM_TIMEOUT_SECONDS", "zero", get_llm_timeout_seconds),
            ("LLM_TIMEOUT_SECONDS", "0", get_llm_timeout_seconds),
            ("LLM_MAX_RETRIES", "4", get_llm_max_retries),
            ("LLM_MAX_RETRIES", "-1", get_llm_max_retries),
            ("LLM_MAX_CONCURRENCY", "0", get_llm_max_concurrency),
            ("RAG_TIMEOUT_SECONDS", "-1", get_rag_timeout_seconds),
            ("STREAM_SAFETY_TAIL_CHARS", "31", get_stream_safety_tail_chars),
        )

        for name, value, getter in cases:
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, name):
                        getter()


class DeterministicProviderConfigurationTests(unittest.TestCase):
    def test_provider_is_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_deterministic_provider_enabled())

    def test_provider_is_allowed_only_in_test_or_integration(self):
        for runtime in ("test", "integration"):
            with self.subTest(runtime=runtime), patch.dict(
                os.environ,
                {"AI_RUNTIME_ENV": runtime, "AI_DETERMINISTIC_PROVIDER": "true"},
                clear=True,
            ):
                self.assertTrue(is_deterministic_provider_enabled())

        for runtime in ("development", "production"):
            with self.subTest(runtime=runtime), patch.dict(
                os.environ,
                {"AI_RUNTIME_ENV": runtime, "AI_DETERMINISTIC_PROVIDER": "true"},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "test or integration"):
                    is_deterministic_provider_enabled()
