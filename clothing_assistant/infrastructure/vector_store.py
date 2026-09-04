import json
import math
import os
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import httpx

from clothing_assistant.config_data import (
    DEFAULT_TEST_QUERY,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    JINA_EMBEDDING_URL,
    VECTOR_DB_DIR,
    get_rag_timeout_seconds,
    is_deterministic_provider_enabled,
)
from clothing_assistant.infrastructure.deterministic_provider import DeterministicEmbeddings
from clothing_assistant.infrastructure.knowledge_base import build_knowledge_chunks, load_knowledge_files


_EMBEDDINGS_CACHE = None
_VECTOR_DATA_CACHE = None
VECTOR_STORE_FILE = VECTOR_DB_DIR / "simple_vector_store.json"
VECTOR_STORE_META_FILE = VECTOR_DB_DIR / "vector_store_meta.json"
VECTOR_STORE_POINTER_FILE = VECTOR_DB_DIR / "current.json"
VECTOR_STORE_VERSIONS_DIR = VECTOR_DB_DIR / "versions"
VERSION_VECTOR_FILE_NAME = "simple_vector_store.json"
VERSION_META_FILE_NAME = "vector_store_meta.json"


class JinaEmbeddings:
    """Minimal Jina embedding adapter for document and query retrieval tasks."""

    def __init__(self, api_key=None, model=EMBEDDING_MODEL_NAME):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model = model

        if not self.api_key:
            raise RuntimeError("JINA_API_KEY is required to build or query the RAG vector store.")

    def _embed(self, inputs, task):
        response = httpx.post(
            JINA_EMBEDDING_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": inputs,
                "embedding_type": "float",
                "task": task,
            },
            timeout=get_rag_timeout_seconds(),
        )
        response.raise_for_status()
        data = response.json().get("data")

        if not isinstance(data, list):
            raise RuntimeError("Jina embedding response does not contain a data list.")

        try:
            ordered_data = sorted(data, key=lambda item: item["index"])
            embeddings = [item["embedding"] for item in ordered_data]
        except (KeyError, TypeError):
            raise RuntimeError("Jina embedding response has an invalid item shape.") from None

        if len(embeddings) != len(inputs):
            raise RuntimeError("Jina embedding response count does not match the input count.")

        return embeddings

    def embed_documents(self, texts):
        inputs = list(texts)
        if not inputs:
            return []
        return self._embed(inputs, "retrieval.passage")

    def embed_query(self, text):
        return self._embed([text], "retrieval.query")[0]


def write_json_atomically(path, value):
    """Write JSON through a sibling temporary file, preserving the old target on failure.

    Args:
        path: JSON file to replace after serialization completes.
        value: JSON-serializable value to persist.

    Raises:
        TypeError: If ``value`` cannot be serialized as JSON.
        OSError: If the temporary file cannot be written or replaced.
    """
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False)
        temporary_path.replace(path)
    finally:
        # 序列化或替换失败时，旧索引仍在；只清理未完成的临时文件。
        temporary_path.unlink(missing_ok=True)


# 初始化 embedding 客户端：知识文本和用户问题必须使用同一模型，向量才有可比性。
def get_embeddings():
    global _EMBEDDINGS_CACHE

    # Streamlit 点击按钮会重跑页面脚本；缓存客户端对象避免重复初始化。
    if _EMBEDDINGS_CACHE is None:
        _EMBEDDINGS_CACHE = DeterministicEmbeddings() if is_deterministic_provider_enabled() else JinaEmbeddings()

    return _EMBEDDINGS_CACHE


# 读取本地 JSON 向量库：这个最小版本不用 Chroma，避免 SQLite/Rust 后端在本机环境里反复报错。
def load_vector_data():
    global _VECTOR_DATA_CACHE

    if _VECTOR_DATA_CACHE is not None:
        return _VECTOR_DATA_CACHE

    vector_file, _ = resolve_current_vector_store_paths()
    if not vector_file.exists():
        raise FileNotFoundError("向量库文件不存在，请先上传知识文件或运行 vector_stores.py 重建向量库。")

    with vector_file.open("r", encoding="utf-8") as file:
        _VECTOR_DATA_CACHE = json.load(file)

    return _VECTOR_DATA_CACHE


def build_source_file_meta(knowledge_chunks):
    source_paths = {}

    for chunk in knowledge_chunks:
        file_name = chunk["file_name"]
        file_path = chunk.get("file_path")
        source_paths[file_name] = file_path

    source_files = []

    for file_name, file_path in sorted(source_paths.items()):
        path = Path(file_path) if file_path else None
        if path and path.exists():
            content = path.read_bytes()
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        else:
            content = "\n".join(
                chunk["content"]
                for chunk in knowledge_chunks
                if chunk["file_name"] == file_name
            ).encode("utf-8")
            updated_at = None

        source_files.append(
            {
                "file_name": file_name,
                "sha256": sha256(content).hexdigest(),
                "updated_at": updated_at,
            }
        )

    return source_files


def build_content_digest(knowledge_chunks):
    """Return a deterministic digest of the source identity and chunk content."""
    digest_input = [
        {
            "chunk_id": chunk["chunk_id"],
            "file_name": chunk["file_name"],
            "content": chunk["content"],
        }
        for chunk in knowledge_chunks
    ]
    encoded = json.dumps(digest_input, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_vector_store_meta(knowledge_chunks, source_task_id=None, version=None):
    built_at = datetime.now(timezone.utc).isoformat()
    return {
        "version": version or built_at,
        "source_files": build_source_file_meta(knowledge_chunks),
        "chunk_count": len(knowledge_chunks),
        "content_digest": build_content_digest(knowledge_chunks),
        "source_task_id": source_task_id,
        "embedding_provider": "deterministic" if is_deterministic_provider_enabled() else "jina",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "built_at": built_at,
    }


def resolve_current_vector_store_paths():
    """Resolve the active version, falling back to the legacy fixed files."""
    if not VECTOR_STORE_POINTER_FILE.exists():
        return VECTOR_STORE_FILE, VECTOR_STORE_META_FILE

    with VECTOR_STORE_POINTER_FILE.open("r", encoding="utf-8") as file:
        pointer = json.load(file)
    index_version = pointer.get("index_version") if isinstance(pointer, dict) else None
    if not isinstance(index_version, str) or not index_version:
        raise ValueError("vector store pointer has no index_version")
    if Path(index_version).name != index_version:
        raise ValueError("vector store pointer contains an invalid index_version")

    version_dir = VECTOR_STORE_VERSIONS_DIR / index_version
    return version_dir / VERSION_VECTOR_FILE_NAME, version_dir / VERSION_META_FILE_NAME


def load_vector_store_meta():
    _, meta_file = resolve_current_vector_store_paths()
    if not meta_file.exists():
        return {}

    with meta_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_vector_store_status(ready, reason, meta=None):
    meta = meta or {}
    return {
        "ready": ready,
        "reason": reason,
        "chunk_count": meta.get("chunk_count", 0),
        "version": meta.get("version"),
        "built_at": meta.get("built_at"),
        "source_task_id": meta.get("source_task_id"),
    }


def _source_hashes(source_files):
    return {
        item.get("file_name"): item.get("sha256")
        for item in source_files
        if item.get("file_name") and item.get("sha256")
    }


def get_vector_store_status():
    """Return safe readiness metadata without exposing vectors or knowledge text."""
    try:
        vector_file, meta_file = resolve_current_vector_store_paths()
    except (OSError, ValueError):
        return _build_vector_store_status(False, "invalid_vector_store_meta")

    if not vector_file.exists():
        return _build_vector_store_status(False, "missing_vector_store")

    if not meta_file.exists():
        return _build_vector_store_status(False, "missing_vector_store_meta")

    try:
        meta = load_vector_store_meta()
    except (OSError, ValueError):
        return _build_vector_store_status(False, "invalid_vector_store_meta")

    if not isinstance(meta, dict):
        return _build_vector_store_status(False, "invalid_vector_store_meta")

    try:
        with vector_file.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except (OSError, ValueError):
        return _build_vector_store_status(False, "invalid_vector_store", meta)

    if not isinstance(records, list):
        return _build_vector_store_status(False, "invalid_vector_store", meta)

    chunk_count = meta.get("chunk_count")
    if not isinstance(chunk_count, int) or len(records) != chunk_count:
        return _build_vector_store_status(False, "chunk_count_mismatch", meta)

    meta_sources = meta.get("source_files")
    if not isinstance(meta_sources, list):
        return _build_vector_store_status(False, "invalid_vector_store_meta", meta)

    try:
        current_chunks = build_knowledge_chunks(load_knowledge_files())
        current_sources = build_source_file_meta(current_chunks)
    except (OSError, ValueError):
        return _build_vector_store_status(False, "source_files_changed", meta)

    current_hashes = _source_hashes(current_sources)
    stored_hashes = _source_hashes(meta_sources)
    if len(current_chunks) != chunk_count or current_hashes != stored_hashes:
        return _build_vector_store_status(False, "source_files_changed", meta)

    return _build_vector_store_status(True, "ready", meta)


# 计算两个向量的余弦距离：距离越小，说明用户问题和知识块越相似。
def cosine_distance(query_vector, chunk_vector):
    dot_value = sum(left * right for left, right in zip(query_vector, chunk_vector))
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    chunk_norm = math.sqrt(sum(value * value for value in chunk_vector))

    if query_norm == 0 or chunk_norm == 0:
        return 1.0

    cosine_similarity = dot_value / (query_norm * chunk_norm)
    return 1 - cosine_similarity


# 把我们自己的 chunk 结构转成本地向量库记录，每条记录包含原文、元数据和 embedding。
def build_vector_records_from_chunks(knowledge_chunks):
    embeddings = get_embeddings()
    chunk_texts = [chunk["content"] for chunk in knowledge_chunks]
    chunk_vectors = embeddings.embed_documents(chunk_texts)
    vector_records = []

    for chunk, chunk_vector in zip(knowledge_chunks, chunk_vectors):
        vector_records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "file_name": chunk["file_name"],
                "file_path": chunk["file_path"],
                # 将 domain 随向量持久化，避免检索后丢失知识的业务边界。
                "domain": chunk.get("domain", "general"),
                "content": chunk["content"],
                "embedding": chunk_vector,
            }
        )

    return vector_records


# 重建向量库：当前最小版本每次都以当前知识文件为准，全量生成 embedding 后写入本地 JSON 文件。
def rebuild_vector_store(knowledge_chunks):
    global _VECTOR_DATA_CACHE

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    vector_records = build_vector_records_from_chunks(knowledge_chunks)
    vector_store_meta = build_vector_store_meta(knowledge_chunks)

    # meta 是索引就绪标记，最后替换；重建失败时旧 meta 仍会阻止半成品被当成可用索引。
    write_json_atomically(VECTOR_STORE_FILE, vector_records)
    write_json_atomically(VECTOR_STORE_META_FILE, vector_store_meta)

    _VECTOR_DATA_CACHE = vector_records

    return vector_records


def _validate_version_files(vector_file, meta_file):
    with Path(vector_file).open("r", encoding="utf-8") as file:
        records = json.load(file)
    with Path(meta_file).open("r", encoding="utf-8") as file:
        meta = json.load(file)

    if not isinstance(records, list) or not isinstance(meta, dict):
        raise ValueError("staged vector store has an invalid shape")
    if meta.get("chunk_count") != len(records):
        raise ValueError("staged vector store chunk count does not match metadata")
    if not meta.get("content_digest") or not meta.get("version"):
        raise ValueError("staged vector store metadata is incomplete")
    return records, meta


def _build_rebuild_result(meta, replayed):
    return {
        "task_id": meta["source_task_id"],
        "index_version": meta["version"],
        "file_count": len(meta["source_files"]),
        "chunk_count": meta["chunk_count"],
        "content_digest": meta["content_digest"],
        "replayed": replayed,
    }


def _rebuild_versioned_vector_store(knowledge_docs, knowledge_chunks, task_id):
    global _VECTOR_DATA_CACHE

    try:
        current_vector_file, current_meta_file = resolve_current_vector_store_paths()
        if current_vector_file.exists() and current_meta_file.exists():
            _, current_meta = _validate_version_files(current_vector_file, current_meta_file)
            if current_meta.get("source_task_id") == task_id:
                return _build_rebuild_result(current_meta, replayed=True)
    except (OSError, ValueError, KeyError):
        pass

    vector_records = build_vector_records_from_chunks(knowledge_chunks)
    content_digest = build_content_digest(knowledge_chunks)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    index_version = f"{timestamp}-{content_digest[:12]}"
    version_dir = VECTOR_STORE_VERSIONS_DIR / index_version
    vector_file = version_dir / VERSION_VECTOR_FILE_NAME
    meta_file = version_dir / VERSION_META_FILE_NAME
    meta = build_vector_store_meta(
        knowledge_chunks,
        source_task_id=task_id,
        version=index_version,
    )

    try:
        write_json_atomically(vector_file, vector_records)
        write_json_atomically(meta_file, meta)
        validated_records, validated_meta = _validate_version_files(vector_file, meta_file)
        write_json_atomically(
            VECTOR_STORE_POINTER_FILE,
            {"index_version": index_version},
        )
    except Exception:
        shutil.rmtree(version_dir, ignore_errors=True)
        raise

    _VECTOR_DATA_CACHE = validated_records
    return _build_rebuild_result(validated_meta, replayed=False)


# 根据用户问题检索最相关的知识块，并把结果整理成更容易打印和后续使用的结构。
def search_similar_chunks(query, top_k=DEFAULT_TOP_K, metadata_filter=None):
    vector_records = load_vector_data()
    query_vector = get_embeddings().embed_query(query)
    matched_chunks = []

    for record in vector_records:
        if metadata_filter:
            is_matched = all(record.get(key) == value for key, value in metadata_filter.items())
            if not is_matched:
                continue

        score = cosine_distance(query_vector, record["embedding"])
        matched_chunks.append(
            {
                "chunk_id": record["chunk_id"],
                "file_name": record["file_name"],
                "file_path": record["file_path"],
                "domain": record.get("domain", "general"),
                "content": record["content"],
                "score": score,
            }
        )

    matched_chunks.sort(key=lambda chunk: chunk["score"])
    return matched_chunks[:top_k]


# 从本地知识文件直接完成“读取 -> 切块 -> 重建向量库”，方便上传页在文件变化时直接调用。
def rebuild_vector_store_from_local_knowledge(task_id=None):
    knowledge_docs = load_knowledge_files()
    knowledge_chunks = build_knowledge_chunks(knowledge_docs)

    if task_id is not None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must not be blank")
        return _rebuild_versioned_vector_store(knowledge_docs, knowledge_chunks, task_id)

    vector_store = rebuild_vector_store(knowledge_chunks)

    return vector_store, knowledge_docs, knowledge_chunks


def main():
    _, knowledge_docs, knowledge_chunks = rebuild_vector_store_from_local_knowledge()
    matched_chunks = search_similar_chunks(DEFAULT_TEST_QUERY)

    print(f"已完成向量入库，共写入 {len(knowledge_chunks)} 个文本块。")
    print(f"测试问题：{DEFAULT_TEST_QUERY}")
    print("检索结果：")

    for index, chunk in enumerate(matched_chunks, start=1):
        print(
            f"[{index}] 文件: {chunk['file_name']} | chunk: {chunk['chunk_id']} | "
            f"score: {chunk['score']:.4f}"
        )
        print(chunk["content"])
        print("-" * 60)


if __name__ == "__main__":
    main()
