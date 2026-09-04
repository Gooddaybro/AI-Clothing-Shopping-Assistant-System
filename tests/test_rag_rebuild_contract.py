import json
import unittest

from tests.contract_support import resolve_contract_root


CONTRACT_SCHEMAS = resolve_contract_root() / "rag-rebuild" / "schemas"


def load_schema(filename: str) -> dict:
    return json.loads((CONTRACT_SCHEMAS / filename).read_text(encoding="utf-8"))


class RagRebuildContractTests(unittest.TestCase):
    def test_request_and_response_fields_match_shared_contract(self):
        request_schema = load_schema("rag-rebuild-request.schema.json")
        response_schema = load_schema("rag-rebuild-response.schema.json")

        self.assertEqual(set(request_schema["required"]), {"taskId", "source"})
        self.assertFalse(request_schema["additionalProperties"])
        self.assertEqual(set(request_schema["properties"]), set(request_schema["required"]))
        self.assertEqual(
            set(response_schema["required"]),
            {
                "taskId",
                "indexVersion",
                "fileCount",
                "chunkCount",
                "contentDigest",
                "replayed",
            },
        )
        self.assertFalse(response_schema["additionalProperties"])
        self.assertEqual(set(response_schema["properties"]), set(response_schema["required"]))

    def test_requested_event_fields_match_shared_contract(self):
        event_schema = load_schema("ai-task-requested.schema.json")

        self.assertEqual(
            set(event_schema["required"]),
            {
                "eventId",
                "eventType",
                "schemaVersion",
                "taskId",
                "taskType",
                "occurredAt",
                "correlationId",
                "traceparent",
            },
        )
        self.assertFalse(event_schema["additionalProperties"])
        self.assertEqual(set(event_schema["properties"]), set(event_schema["required"]))


if __name__ == "__main__":
    unittest.main()
