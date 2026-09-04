"""LLM client factory for chat-model access."""

import os
import threading
import time

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError

from clothing_assistant.config_data import (
    CHAT_MODEL_NAME,
    CHAT_TEMPERATURE,
    KIMI_BASE_URL,
    get_llm_max_concurrency,
    get_llm_max_retries,
    get_llm_timeout_seconds,
    is_deterministic_provider_enabled,
)
from clothing_assistant.infrastructure.deterministic_provider import DeterministicChatModel


_MODEL_SEMAPHORE = None
_MODEL_SEMAPHORE_SIZE = None
_MODEL_SEMAPHORE_LOCK = threading.Lock()


class DependencyError(RuntimeError):
    """Safe classification for an upstream dependency failure."""

    def __init__(self, dependency: str, reason: str, retryable: bool):
        super().__init__(f"{dependency} failed: {reason}")
        self.dependency = dependency
        self.reason = reason
        self.retryable = retryable


def _get_model_semaphore():
    global _MODEL_SEMAPHORE, _MODEL_SEMAPHORE_SIZE
    size = get_llm_max_concurrency()
    with _MODEL_SEMAPHORE_LOCK:
        if _MODEL_SEMAPHORE is None or _MODEL_SEMAPHORE_SIZE != size:
            _MODEL_SEMAPHORE = threading.BoundedSemaphore(size)
            _MODEL_SEMAPHORE_SIZE = size
    return _MODEL_SEMAPHORE


def classify_dependency_error(error: Exception, dependency: str = "llm") -> DependencyError:
    status_code = getattr(error, "status_code", None)
    if isinstance(error, (TimeoutError, APITimeoutError)):
        return DependencyError(dependency, "timeout", True)
    if isinstance(error, (ConnectionError, APIConnectionError)):
        return DependencyError(dependency, "connection_error", True)
    if status_code == 429:
        return DependencyError(dependency, "rate_limited", True)
    if isinstance(status_code, int) and status_code >= 500:
        return DependencyError(dependency, "upstream_5xx", True)
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return DependencyError(dependency, "upstream_4xx", False)
    return DependencyError(dependency, "invalid_response", False)


def get_chat_model():
    if is_deterministic_provider_enabled():
        return DeterministicChatModel()
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY is required to generate Kimi chat responses.")

    return ChatOpenAI(
        model=CHAT_MODEL_NAME,
        temperature=CHAT_TEMPERATURE,
        api_key=api_key,
        base_url=KIMI_BASE_URL,
        request_timeout=get_llm_timeout_seconds(),
        max_retries=0,
    )


def get_demand_intent_model():
    """Create a no-retry model with the parser's strict eight-second boundary."""
    if is_deterministic_provider_enabled():
        return DeterministicChatModel(demand_parser=True)
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY is required to parse demand intent.")
    return ChatOpenAI(
        model=CHAT_MODEL_NAME,
        temperature=0,
        api_key=api_key,
        base_url=KIMI_BASE_URL,
        request_timeout=8,
        max_retries=0,
    )


def invoke_chat_content(messages, *, model_factory=None):
    """Invoke the demand parser once and return textual model content."""
    factory = model_factory or get_demand_intent_model
    with _get_model_semaphore():
        try:
            content = getattr(factory().invoke(messages), "content", "")
        except Exception as error:
            raise classify_dependency_error(error) from None
    if not isinstance(content, str) or not content:
        raise DependencyError("llm", "invalid_response", False)
    return content


def stream_chat_content(
    messages,
    *,
    model_factory=None,
    stop_requested=None,
    sleep=None,
):
    """Yield provider fragments with bounded retry before public output."""
    model_factory = model_factory or get_chat_model
    stop_requested = stop_requested or (lambda: False)
    sleep = sleep or time.sleep
    max_retries = get_llm_max_retries()
    emitted = False

    with _get_model_semaphore():
        for attempt in range(max_retries + 1):
            provider_stream = None
            try:
                model = model_factory()
                provider_stream = iter(model.stream(messages))
                if stop_requested():
                    return

                for chunk in provider_stream:
                    if stop_requested():
                        return
                    content = getattr(chunk, "content", "")
                    if not isinstance(content, str) or not content:
                        continue
                    emitted = True
                    yield content
                return
            except Exception as error:
                dependency_error = classify_dependency_error(error)
                if dependency_error.retryable and not emitted and attempt < max_retries:
                    sleep(0.1 * (2 ** attempt))
                    continue
                raise dependency_error from None
            finally:
                close = getattr(provider_stream, "close", None)
                if close is not None:
                    close()
