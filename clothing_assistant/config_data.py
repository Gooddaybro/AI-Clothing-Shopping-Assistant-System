from pathlib import Path
import os

from dotenv import load_dotenv

# 应该只放配置，不放业务逻辑。
# 定义项目里的“路径配置”和“默认要加载哪些知识文件”。

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

PROJECT_DISPLAY_NAME = "AI Clothing Shopping Assistant System"
PROJECT_API_TITLE = f"{PROJECT_DISPLAY_NAME} API"
DATA_DIR = BASE_DIR / "data"
PRODUCT_CATALOG_PATH = DATA_DIR / "product_catalog.json"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"
FILE_HASH_RECORD_PATH = BASE_DIR / "knowledge_file_hashes.json"
VECTOR_COLLECTION_NAME = "clothing_knowledge_base"
EMBEDDING_MODEL_NAME = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v4")
JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
JINA_EMBEDDING_TIMEOUT_SECONDS = 30
CHAT_MODEL_NAME = os.getenv("KIMI_CHAT_MODEL", "kimi-k2.5")
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
# Kimi K2.5 当前仅接受 temperature=1；较低随机度由证据校验而不是该参数保证。
CHAT_TEMPERATURE = 1
ENABLE_LLM_PREFERENCE_MAPPER = os.getenv("ENABLE_LLM_PREFERENCE_MAPPER", "false").lower() == "true"
RUNTIME_ENVIRONMENT_ENV = "AI_RUNTIME_ENV"
CHECKPOINTER_BACKEND_ENV = "LANGGRAPH_CHECKPOINTER_BACKEND"
CHECKPOINTER_DSN_ENV = "LANGGRAPH_CHECKPOINTER_DSN"
INTERNAL_API_TOKEN_ENV = "APP_INTERNAL_API_TOKEN"
DETERMINISTIC_PROVIDER_ENV = "AI_DETERMINISTIC_PROVIDER"


def _get_positive_float(name: str, default: str) -> float:
    raw_value = os.getenv(name, default).strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _get_int(name: str, default: str, *, minimum: int, maximum: int | None = None) -> int:
    raw_value = os.getenv(name, default).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < minimum or maximum is not None and value > maximum:
        expected = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise RuntimeError(f"{name} must be {expected}")
    return value


def get_llm_timeout_seconds() -> float:
    return _get_positive_float("LLM_TIMEOUT_SECONDS", "30")


def get_llm_max_retries() -> int:
    return _get_int("LLM_MAX_RETRIES", "2", minimum=0, maximum=3)


def get_llm_max_concurrency() -> int:
    return _get_int("LLM_MAX_CONCURRENCY", "8", minimum=1)


def get_rag_timeout_seconds() -> float:
    return _get_positive_float("RAG_TIMEOUT_SECONDS", "20")


def get_stream_safety_tail_chars() -> int:
    return _get_int("STREAM_SAFETY_TAIL_CHARS", "64", minimum=32)


def get_runtime_environment() -> str:
    return os.getenv(RUNTIME_ENVIRONMENT_ENV, "development").strip().lower()


def is_deterministic_provider_enabled() -> bool:
    """Enable stable provider substitutes only in explicit non-production runtimes."""
    enabled = os.getenv(DETERMINISTIC_PROVIDER_ENV, "false").strip().lower() == "true"
    if enabled and get_runtime_environment() not in {"test", "integration"}:
        raise RuntimeError("deterministic provider is allowed only in test or integration")
    return enabled


def get_internal_api_token() -> str:
    return os.getenv(INTERNAL_API_TOKEN_ENV, "").strip()


def is_internal_auth_required() -> bool:
    return get_runtime_environment() == "production" or bool(get_internal_api_token())


def get_max_chat_request_bytes() -> int:
    raw_value = os.getenv("MAX_CHAT_REQUEST_BYTES", "262144").strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError("MAX_CHAT_REQUEST_BYTES must be an integer") from error
    if value < 1024:
        raise RuntimeError("MAX_CHAT_REQUEST_BYTES must be at least 1024")
    return value


def get_checkpointer_backend() -> str:
    backend = os.getenv(CHECKPOINTER_BACKEND_ENV, "").strip().lower()
    if not backend:
        return "postgres" if get_runtime_environment() == "production" else "memory"
    if backend not in {"memory", "postgres"}:
        raise RuntimeError("LANGGRAPH_CHECKPOINTER_BACKEND must be memory or postgres")
    if get_runtime_environment() == "production" and backend != "postgres":
        raise RuntimeError("production requires LANGGRAPH_CHECKPOINTER_BACKEND=postgres")
    return backend


def get_checkpointer_dsn() -> str | None:
    dsn = os.getenv(CHECKPOINTER_DSN_ENV, "").strip()
    if get_checkpointer_backend() == "postgres" and not dsn:
        raise RuntimeError("LANGGRAPH_CHECKPOINTER_DSN is required for postgres")
    return dsn or None


def is_debug_response_enabled() -> bool:
    """Return whether this local process may expose internal debug payloads."""
    return os.getenv("DEBUG_RESPONSE_ENABLED", "false").strip().lower() == "true"


DEFAULT_TOP_K = 4
RAG_TOP_K = 3
# 由 2026-07-11 真实检索评测选出：0.25 可拒绝当前两条知识外问题。
RAG_DISTANCE_THRESHOLD = 0.25
SIZE_KNOWLEDGE_FILE = "尺码推荐.txt"
COLOR_KNOWLEDGE_FILE = "颜色选择.txt"
CARE_KNOWLEDGE_FILE = "洗涤养护.txt"
SCENE_KNOWLEDGE_FILE = "场景穿搭.txt"
MATERIAL_KNOWLEDGE_FILE = "材质知识.txt"
FIT_KNOWLEDGE_FILE = "版型知识.txt"
KNOWLEDGE_FILE_DOMAINS = {
    SIZE_KNOWLEDGE_FILE: "size",
    COLOR_KNOWLEDGE_FILE: "color",
    CARE_KNOWLEDGE_FILE: "care",
    SCENE_KNOWLEDGE_FILE: "scene",
    MATERIAL_KNOWLEDGE_FILE: "material",
    FIT_KNOWLEDGE_FILE: "fit",
}
DEFAULT_TEST_QUERY = (
    "我身高168，体重65kg，想买一件日常穿的T恤，"
    "推荐什么尺码和颜色？洗的时候需要注意什么？"
)

KNOWLEDGE_FILES = [
    "尺码推荐.txt",
    "洗涤养护.txt",
    "颜色选择.txt",
    "场景穿搭.txt",
    "材质知识.txt",
    "版型知识.txt",
]
