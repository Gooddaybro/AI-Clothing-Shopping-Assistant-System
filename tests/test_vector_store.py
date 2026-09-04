import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from clothing_assistant.infrastructure import vector_store
from clothing_assistant.infrastructure.knowledge_base import build_knowledge_chunks


@contextmanager
def temporary_vector_store_paths(temp_dir):
    temp_path = Path(temp_dir)
    with patch.object(vector_store, "VECTOR_STORE_FILE", temp_path / "vectors.json"), patch.object(
        vector_store,
        "VECTOR_STORE_META_FILE",
        temp_path / "meta.json",
    ), patch.object(
        vector_store,
        "VECTOR_STORE_POINTER_FILE",
        temp_path / "current.json",
        create=True,
    ), patch.object(
        vector_store,
        "VECTOR_STORE_VERSIONS_DIR",
        temp_path / "versions",
        create=True,
    ):
        yield


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class VectorStoreStatusTests(unittest.TestCase):
    def get_status(self):
        status_loader = getattr(vector_store, "get_vector_store_status", None)
        self.assertIsNotNone(status_loader)
        return status_loader()

    def test_missing_vector_store_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir, temporary_vector_store_paths(temp_dir):
            status = self.get_status()

        self.assertEqual(status["ready"], False)
        self.assertEqual(status["reason"], "missing_vector_store")

    def test_missing_meta_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir, temporary_vector_store_paths(temp_dir):
            write_json(vector_store.VECTOR_STORE_FILE, [])
            status = self.get_status()

        self.assertEqual(status["ready"], False)
        self.assertEqual(status["reason"], "missing_vector_store_meta")

    def test_broken_meta_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir, temporary_vector_store_paths(temp_dir):
            write_json(vector_store.VECTOR_STORE_FILE, [])
            vector_store.VECTOR_STORE_META_FILE.write_text("{broken", encoding="utf-8")
            status = self.get_status()

        self.assertEqual(status["ready"], False)
        self.assertEqual(status["reason"], "invalid_vector_store_meta")

    def test_broken_vector_store_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir, temporary_vector_store_paths(temp_dir):
            vector_store.VECTOR_STORE_FILE.write_text("[broken", encoding="utf-8")
            write_json(vector_store.VECTOR_STORE_META_FILE, {"chunk_count": 0})
            status = self.get_status()

        self.assertEqual(status["ready"], False)
        self.assertEqual(status["reason"], "invalid_vector_store")

    def test_chunk_count_mismatch_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir, temporary_vector_store_paths(temp_dir):
            write_json(vector_store.VECTOR_STORE_FILE, [{"chunk_id": "one"}])
            write_json(vector_store.VECTOR_STORE_META_FILE, {"chunk_count": 2})
            status = self.get_status()

        self.assertEqual(status["ready"], False)
        self.assertEqual(status["reason"], "chunk_count_mismatch")

    def test_changed_source_files_are_not_ready(self):
        knowledge_docs = [
            {
                "file_name": "测试知识.txt",
                "file_path": "/missing/test.txt",
                "content": "新知识",
            }
        ]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            temporary_vector_store_paths(temp_dir),
            patch.object(vector_store, "load_knowledge_files", return_value=knowledge_docs),
        ):
            write_json(vector_store.VECTOR_STORE_FILE, [{"chunk_id": "one"}])
            write_json(
                vector_store.VECTOR_STORE_META_FILE,
                {
                    "chunk_count": 1,
                    "source_files": [{"file_name": "测试知识.txt", "sha256": "old"}],
                },
            )
            status = self.get_status()

        self.assertEqual(status["ready"], False)
        self.assertEqual(status["reason"], "source_files_changed")

    def test_matching_store_meta_and_sources_are_ready(self):
        knowledge_docs = [
            {
                "file_name": "测试知识.txt",
                "file_path": "/missing/test.txt",
                "content": "第一条\n第二条",
            }
        ]
        chunks = build_knowledge_chunks(knowledge_docs)
        source_files = vector_store.build_source_file_meta(chunks)
        meta = {
            "version": "test-version",
            "built_at": "2026-07-10T00:00:00+00:00",
            "chunk_count": 2,
            "source_task_id": "task-health-check",
            "source_files": source_files,
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            temporary_vector_store_paths(temp_dir),
            patch.object(vector_store, "load_knowledge_files", return_value=knowledge_docs),
        ):
            write_json(
                vector_store.VECTOR_STORE_FILE,
                [{"chunk_id": "one"}, {"chunk_id": "two"}],
            )
            write_json(vector_store.VECTOR_STORE_META_FILE, meta)
            status = self.get_status()

        self.assertEqual(
            status,
            {
                "ready": True,
                "reason": "ready",
                "chunk_count": 2,
                "version": "test-version",
                "built_at": "2026-07-10T00:00:00+00:00",
                "source_task_id": "task-health-check",
            },
        )


class JinaEmbeddingsTests(unittest.TestCase):
    def get_embeddings_client(self, api_key="test-jina-key"):
        client_type = getattr(vector_store, "JinaEmbeddings", None)
        self.assertIsNotNone(client_type)
        return client_type(api_key=api_key)

    def test_integration_runtime_uses_deterministic_embeddings_without_jina(self):
        vector_store._EMBEDDINGS_CACHE = None
        with patch.dict(
            os.environ,
            {"AI_RUNTIME_ENV": "integration", "AI_DETERMINISTIC_PROVIDER": "true"},
            clear=True,
        ), patch("clothing_assistant.infrastructure.vector_store.httpx.post") as post:
            embeddings = vector_store.get_embeddings()
            first = embeddings.embed_query("通勤")
            second = embeddings.embed_query("通勤")
            other = embeddings.embed_query("休闲")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 16)
        post.assert_not_called()

    def test_document_embeddings_use_passage_task_and_response_indexes(self):
        client = self.get_embeddings_client()
        response = Mock()
        response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.2, 0.3]},
                {"index": 0, "embedding": [0.0, 0.1]},
            ]
        }
        with patch("clothing_assistant.infrastructure.vector_store.httpx.post", create=True) as post:
            post.return_value = response

            embeddings = client.embed_documents(["文档 A", "文档 B"])

        self.assertEqual(embeddings, [[0.0, 0.1], [0.2, 0.3]])
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["task"], "retrieval.passage")
        self.assertEqual(post.call_args.kwargs["json"]["input"], ["文档 A", "文档 B"])

    def test_query_embedding_uses_query_task(self):
        client = self.get_embeddings_client()
        response = Mock()
        response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
        with patch("clothing_assistant.infrastructure.vector_store.httpx.post", create=True) as post:
            post.return_value = response

            embedding = client.embed_query("通勤适合什么颜色？")

        self.assertEqual(embedding, [0.1, 0.2])
        self.assertEqual(post.call_args.kwargs["json"]["task"], "retrieval.query")

    def test_embedding_request_uses_configured_rag_timeout(self):
        client = self.get_embeddings_client()
        response = Mock()
        response.json.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
        with patch(
            "clothing_assistant.infrastructure.vector_store.get_rag_timeout_seconds",
            return_value=7.25,
            create=True,
        ), patch("clothing_assistant.infrastructure.vector_store.httpx.post") as post:
            post.return_value = response

            client.embed_query("测试")

        self.assertEqual(post.call_args.kwargs["timeout"], 7.25)

    def test_embedding_client_requires_jina_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            client_type = getattr(vector_store, "JinaEmbeddings", None)
            self.assertIsNotNone(client_type)
            with self.assertRaisesRegex(RuntimeError, "JINA_API_KEY"):
                client_type()


class VectorRecordMetadataTests(unittest.TestCase):
    def test_vector_records_keep_knowledge_domain(self):
        chunks = [
            {
                "chunk_id": "场景穿搭.txt-001",
                "file_name": "场景穿搭.txt",
                "file_path": "/tmp/scene.txt",
                "domain": "scene",
                "content": "1. 通勤\n选择简洁基础款。",
            }
        ]

        with patch.object(vector_store, "get_embeddings") as get_embeddings:
            get_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2]]
            records = vector_store.build_vector_records_from_chunks(chunks)

        self.assertEqual(records[0]["domain"], "scene")


class AtomicJsonWriteTests(unittest.TestCase):
    def get_writer(self):
        writer = getattr(vector_store, "write_json_atomically", None)
        self.assertIsNotNone(writer)
        return writer

    def test_atomic_write_replaces_existing_target_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "vectors.json"
            target.write_text('{"version": "old"}', encoding="utf-8")

            self.get_writer()(target, {"version": "new"})

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"version": "new"})
            self.assertFalse(target.with_suffix(".json.tmp").exists())

    def test_atomic_write_keeps_existing_target_when_serialization_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "vectors.json"
            target.write_text('{"version": "old"}', encoding="utf-8")

            with self.assertRaises(TypeError):
                self.get_writer()(target, {"bad": {"not-json"}})

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"version": "old"})
            self.assertFalse(target.with_suffix(".json.tmp").exists())


class VersionedVectorStoreRebuildTests(unittest.TestCase):
    def setUp(self):
        vector_store._VECTOR_DATA_CACHE = None

    def tearDown(self):
        vector_store._VECTOR_DATA_CACHE = None

    def test_rebuild_records_source_task_id_and_replays_same_task(self):
        knowledge_docs = [
            {
                "file_name": "guide.txt",
                "file_path": "/missing/guide.txt",
                "content": "通勤穿搭建议",
            }
        ]
        chunks = build_knowledge_chunks(knowledge_docs)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            temporary_vector_store_paths(temp_dir),
            patch.object(vector_store, "load_knowledge_files", return_value=knowledge_docs),
            patch.object(vector_store, "get_embeddings") as get_embeddings,
        ):
            get_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2]]

            first = vector_store.rebuild_vector_store_from_local_knowledge(task_id="task-1")
            second = vector_store.rebuild_vector_store_from_local_knowledge(task_id="task-1")

            self.assertFalse(first["replayed"])
            self.assertTrue(second["replayed"])
            self.assertEqual(first["index_version"], second["index_version"])
            self.assertEqual(first["chunk_count"], len(chunks))
            get_embeddings.return_value.embed_documents.assert_called_once()

    def test_failed_staged_build_keeps_current_store(self):
        knowledge_docs = [
            {
                "file_name": "guide.txt",
                "file_path": "/missing/guide.txt",
                "content": "通勤穿搭建议",
            }
        ]

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            temporary_vector_store_paths(temp_dir),
            patch.object(vector_store, "load_knowledge_files", return_value=knowledge_docs),
            patch.object(vector_store, "get_embeddings") as get_embeddings,
        ):
            get_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2]]
            vector_store.rebuild_vector_store_from_local_knowledge(task_id="task-1")
            old_pointer = json.loads(
                vector_store.VECTOR_STORE_POINTER_FILE.read_text(encoding="utf-8")
            )

            with patch.object(
                vector_store,
                "write_json_atomically",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    vector_store.rebuild_vector_store_from_local_knowledge(task_id="task-2")

            self.assertEqual(
                json.loads(vector_store.VECTOR_STORE_POINTER_FILE.read_text(encoding="utf-8")),
                old_pointer,
            )

if __name__ == "__main__":
    unittest.main()
