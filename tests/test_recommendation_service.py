import unittest
from unittest.mock import patch

from clothing_assistant.agent.router import intent_router
from clothing_assistant.agent.langgraph_executor import run_langgraph_agent
from clothing_assistant.application.recommendation_service import build_product_refs, build_product_rerank_result
from clothing_assistant.tools.size_tool import normalize_measurement_query, run_size_tool
from tests.fakes import FakeEmbeddings


class RecommendationServiceTests(unittest.TestCase):
    def test_build_product_rerank_result_exposes_preferences_scores_and_source(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "基础通勤夹克",
                "category": "外套",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["commute", "casual", "basic"],
                "attribute_tags": ["适用场景:通勤", "风格:基础款"],
                "sale_price": 269,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "正式商务西装",
                "category": "西装",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["formal"],
                "attribute_tags": ["风格:商务正装"],
                "sale_price": 899,
            },
        ]

        result = build_product_rerank_result(
            candidates,
            {"intent": "recommendation"},
            "推荐一件300以内适合学生党通勤、不要太正式的外套",
            {},
            {},
        )

        self.assertEqual(result["recommendation_source"], "java_candidates_with_ai_rerank")
        self.assertEqual(result["semantic_preferences"]["budget_max"], 300)
        self.assertIn("student", result["semantic_preferences"]["persona_tags"])
        self.assertIn("commute", result["semantic_preferences"]["scene"])
        self.assertIn("overly_formal", result["semantic_preferences"]["avoid_tags"])
        self.assertEqual(result["product_refs"][0]["spu_id"], 1001)
        self.assertEqual(result["candidate_scores"][0]["spu_id"], 1001)
        self.assertGreater(result["candidate_scores"][0]["rank_score"], result["candidate_scores"][1]["rank_score"])

    def test_rerank_result_marks_empty_candidate_source(self):
        result = build_product_rerank_result(
            [],
            {"intent": "recommendation"},
            "推荐一件通勤外套",
            {},
            {},
        )

        self.assertEqual(result["product_refs"], [])
        self.assertEqual(result["candidate_scores"], [])
        self.assertEqual(result["recommendation_source"], "java_candidates_empty")

    def test_rerank_result_marks_non_recommendable_intent_source(self):
        result = build_product_rerank_result(
            [{"spu_id": 1001, "sku_id": 2001, "name": "通勤外套"}],
            {"intent": "chat"},
            "你是谁？",
            {},
            {},
        )

        self.assertEqual(result["product_refs"], [])
        self.assertEqual(result["candidate_scores"], [])
        self.assertEqual(result["recommendation_source"], "not_recommendable_intent")

    def test_bare_height_weight_pair_is_auxiliary_to_outfit_task(self):
        result = intent_router("明天面试想要显瘦 177 130 该怎么选")

        self.assertEqual(result["intent"], "recommendation")
        self.assertEqual(result["request_type"], "OUTFIT_ADVICE")
        self.assertEqual(
            result["requested_capabilities"],
            ["OUTFIT_PLAN", "PRODUCT_SELECTION"],
        )
        self.assertIn("辅助", result["reason"])

    def test_router_keeps_greeting_from_short_circuiting_outfit_task(self):
        greeting = intent_router("你好")
        outfit = intent_router("你好，夏天怎么穿")

        self.assertEqual(greeting["request_type"], "CHAT")
        self.assertEqual(outfit["intent"], "recommendation")
        self.assertEqual(outfit["request_type"], "OUTFIT_ADVICE")

    def test_router_keeps_size_as_capability_when_outfit_is_main_task(self):
        result = intent_router("177 130 夏天怎么穿，顺便看看穿什么码")

        self.assertEqual(result["request_type"], "OUTFIT_ADVICE")
        self.assertEqual(
            result["requested_capabilities"],
            ["OUTFIT_PLAN", "PRODUCT_SELECTION", "SIZE_GUIDANCE"],
        )

    def test_router_routes_explicit_size_question_as_size_main_task(self):
        result = intent_router("177 130 穿什么码")

        self.assertEqual(result["intent"], "size_recommendation")
        self.assertEqual(result["request_type"], "SIZE_RECOMMENDATION")
        self.assertEqual(result["requested_capabilities"], ["SIZE_GUIDANCE"])

    def test_router_uses_v2_java_main_task_as_authority(self):
        result = intent_router(
            "你好，我177 130",
            {
                "version": "demand-intent-v2",
                "requestType": "OUTFIT_ADVICE",
                "requestedCapabilities": ["OUTFIT_PLAN", "PRODUCT_SELECTION"],
            },
        )

        self.assertEqual(result["intent"], "recommendation")
        self.assertEqual(result["request_type"], "OUTFIT_ADVICE")
        self.assertEqual(
            result["requested_capabilities"],
            ["OUTFIT_PLAN", "PRODUCT_SELECTION"],
        )

    def test_router_uses_v3_java_main_task_as_authority(self):
        result = intent_router(
            "你好",
            {
                "version": "demand-intent-v3",
                "requestType": "OUTFIT_ADVICE",
                "requestedCapabilities": ["OUTFIT_PLAN", "PRODUCT_SELECTION"],
                "hardFilters": [],
                "softPreferences": [],
            },
        )

        self.assertEqual(result["intent"], "recommendation")
        self.assertEqual(result["request_type"], "OUTFIT_ADVICE")
        self.assertEqual(result["requested_capabilities"], ["OUTFIT_PLAN", "PRODUCT_SELECTION"])

    def test_router_keeps_v1_product_recommendation_compatibility(self):
        result = intent_router(
            "男性，想买夏季 T 恤",
            {
                "version": "demand-intent-v1",
                "targetGender": "male",
                "category": "T恤",
            },
        )

        self.assertEqual(result["intent"], "recommendation")
        self.assertEqual(result["request_type"], "PRODUCT_RECOMMENDATION")
        self.assertEqual(result["requested_capabilities"], ["PRODUCT_SELECTION"])

    def test_stock_and_base_score_do_not_create_product_ref_without_evidence(self):
        result = build_product_rerank_result(
            [
                {
                    "spu_id": 1001,
                    "sku_id": 2001,
                    "name": "库存充足的基础款",
                    "category": "T恤",
                    "stock_status": "in_stock",
                    "available_stock": 100,
                    "sale_price": 139,
                }
            ],
            {"intent": "recommendation", "request_type": "OUTFIT_ADVICE"},
            "夏天怎么穿",
            {},
            {},
            demand_intent={
                "version": "demand-intent-v2",
                "requestType": "OUTFIT_ADVICE",
                "season": "summer",
            },
        )

        self.assertEqual(result["product_refs"], [])
        self.assertGreaterEqual(result["candidate_scores"][0]["rank_score"], 0)
        self.assertEqual(result["candidate_scores"][0]["matched_dimensions"], [])

    def test_product_ref_contains_structured_explicit_match_evidence(self):
        result = build_product_rerank_result(
            [
                {
                    "spu_id": 1001,
                    "sku_id": 2001,
                    "name": "夏季休闲T恤",
                    "category": "T恤",
                    "season": ["summer"],
                    "style_tags": ["casual"],
                    "fit_type": "relaxed",
                    "stock_status": "in_stock",
                    "sale_price": 139,
                }
            ],
            {"intent": "recommendation", "request_type": "OUTFIT_ADVICE"},
            "夏季休闲宽松T恤怎么穿",
            {},
            {},
            demand_intent={
                "version": "demand-intent-v2",
                "requestType": "OUTFIT_ADVICE",
                "category": "T恤",
                "season": "summer",
                "style": ["casual"],
                "fitPreferences": ["relaxed"],
            },
        )

        self.assertEqual(len(result["product_refs"]), 1)
        evidence = result["product_refs"][0]["matched_dimensions"]
        self.assertEqual(
            {(item["dimension"], item["evidence_source"]) for item in evidence},
            {
                ("category", "PRODUCT_CATEGORY"),
                ("season", "PRODUCT_SEASON"),
                ("style", "PRODUCT_STYLE_TAG"),
                ("fitPreferences", "PRODUCT_FIT"),
            },
        )

    def test_v3_constraints_drive_filtering_ranking_and_match_evidence(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "夏季休闲通勤外套",
                "category": "外套",
                "season": ["summer"],
                "style_tags": ["casual"],
                "stock_status": "in_stock",
                "sale_price": 299,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "高价夏季休闲外套",
                "category": "外套",
                "season": ["summer"],
                "style_tags": ["casual"],
                "stock_status": "in_stock",
                "sale_price": 899,
            },
        ]
        demand_intent = {
            "version": "demand-intent-v3",
            "requestType": "OUTFIT_ADVICE",
            "requestedCapabilities": ["PRODUCT_SELECTION"],
            "hardFilters": [
                self.v3_constraint("category", "EQUALS", ["外套"], "HARD"),
                self.v3_constraint("season", "EQUALS", ["SUMMER"], "HARD"),
                self.v3_constraint("budgetMax", "MAX", ["500"], "HARD"),
            ],
            "softPreferences": [
                self.v3_constraint("style", "CONTAINS", ["CASUAL"], "SOFT", weight=0.8),
            ],
        }

        result = build_product_rerank_result(
            candidates,
            {"intent": "recommendation", "request_type": "OUTFIT_ADVICE"},
            "帮我看看",
            {},
            {},
            demand_intent=demand_intent,
        )

        self.assertEqual([ref["spu_id"] for ref in result["product_refs"]], [1001])
        self.assertIsNone(result["semantic_preferences"]["budget_max"])
        self.assertEqual(result["semantic_preferences"]["season"], [])
        self.assertIn("casual", result["semantic_preferences"]["style_tags"])
        evidence = result["product_refs"][0]["matched_dimensions"]
        self.assertEqual(
            {item["dimension"] for item in evidence},
            {"category", "season", "style", "budgetMax"},
        )

    def test_soft_budget_and_season_never_hard_reject_a_candidate(self):
        demand_intent = {
            "version": "demand-intent-v3",
            "requestType": "OUTFIT_ADVICE",
            "requestedCapabilities": ["PRODUCT_SELECTION"],
            "hardFilters": [
                self.v3_constraint("category", "EQUALS", ["外套"], "HARD"),
            ],
            "softPreferences": [
                self.v3_constraint("budgetMax", "CONTAINS", ["100"], "SOFT", weight=0.7),
                self.v3_constraint("season", "CONTAINS", ["WINTER"], "SOFT", weight=0.6),
            ],
        }

        result = build_product_rerank_result(
            [
                {
                    "spu_id": 1001,
                    "sku_id": 2001,
                    "name": "夏季通勤外套",
                    "category": "外套",
                    "season": ["summer"],
                    "stock_status": "in_stock",
                    "sale_price": 299,
                }
            ],
            {"intent": "recommendation"},
            "帮我看看",
            {},
            {},
            demand_intent=demand_intent,
        )

        self.assertEqual([ref["spu_id"] for ref in result["product_refs"]], [1001])
        self.assertEqual(result["semantic_preferences"]["budget_max"], 100)
        self.assertIn("winter", result["semantic_preferences"]["season"])
        self.assertEqual(result["rejected_reasons"], {})

    def test_soft_budget_does_not_tighten_hard_budget_match_evidence(self):
        demand_intent = {
            "version": "demand-intent-v3",
            "hardFilters": [
                self.v3_constraint("budgetMax", "MAX", ["500"], "HARD"),
            ],
            "softPreferences": [
                self.v3_constraint("budgetMax", "MAX", ["300"], "SOFT", weight=0.7),
            ],
        }

        result = build_product_rerank_result(
            [
                {
                    "spu_id": 1001,
                    "sku_id": 2001,
                    "name": "400 元外套",
                    "sale_price": 400,
                }
            ],
            {"intent": "recommendation"},
            "帮我看看",
            {},
            {},
            demand_intent=demand_intent,
        )

        self.assertEqual([ref["spu_id"] for ref in result["product_refs"]], [1001])
        self.assertEqual(result["rejected_reasons"], {})
        self.assertEqual(result["semantic_preferences"]["budget_max"], 300)
        self.assertEqual(
            result["product_refs"][0]["matched_dimensions"],
            [
                {
                    "dimension": "budgetMax",
                    "requested_value": "500",
                    "candidate_value": "400",
                    "evidence_source": "PRODUCT_PRICE",
                }
            ],
        )

    def test_soft_only_budget_can_supply_match_evidence(self):
        demand_intent = {
            "version": "demand-intent-v3",
            "hardFilters": [],
            "softPreferences": [
                self.v3_constraint("budgetMax", "MAX", ["300"], "SOFT", weight=0.7),
            ],
        }

        result = build_product_rerank_result(
            [{"spu_id": 1001, "sku_id": 2001, "name": "250 元外套", "sale_price": 250}],
            {"intent": "recommendation"},
            "帮我看看",
            {},
            {},
            demand_intent=demand_intent,
        )

        self.assertEqual([ref["spu_id"] for ref in result["product_refs"]], [1001])
        self.assertEqual(
            result["product_refs"][0]["matched_dimensions"][0],
            {
                "dimension": "budgetMax",
                "requested_value": "300",
                "candidate_value": "250",
                "evidence_source": "PRODUCT_PRICE",
            },
        )

    def test_rerank_result_counts_stable_java_rejection_reasons(self):
        hard_category = {
            "version": "demand-intent-v3",
            "hardFilters": [self.v3_constraint("category", "EQUALS", ["外套"], "HARD")],
            "softPreferences": [],
        }
        mismatch = build_product_rerank_result(
            [{"spu_id": 1001, "sku_id": 2001, "name": "T恤", "category": "T恤"}],
            {"intent": "recommendation"},
            "帮我看看",
            {},
            {},
            demand_intent=hard_category,
        )
        size_and_evidence = build_product_rerank_result(
            [
                {"spu_id": 1002, "sku_id": 2002, "name": "M 码外套", "size": "M"},
                {"spu_id": 1003, "sku_id": 2003, "name": "L 码外套", "size": "L"},
            ],
            {"intent": "recommendation"},
            "帮我看看",
            {},
            {"size_tool": {"recommended_size": "L"}},
            demand_intent={"version": "demand-intent-v3", "hardFilters": [], "softPreferences": []},
        )
        limited = build_product_rerank_result(
            [
                {"spu_id": 1004, "sku_id": 2004, "name": "外套 A", "category": "外套"},
                {"spu_id": 1005, "sku_id": 2005, "name": "外套 B", "category": "外套"},
            ],
            {"intent": "recommendation"},
            "外套",
            {},
            {},
            limit=1,
            demand_intent=hard_category,
        )

        self.assertEqual(mismatch["rejected_reasons"], {"HARD_FILTER_MISMATCH": 1})
        self.assertEqual(
            size_and_evidence["rejected_reasons"],
            {"SIZE_MISMATCH": 1, "MISSING_REQUIRED_EVIDENCE": 1},
        )
        self.assertEqual(limited["rejected_reasons"], {"LOW_STYLE_SCORE": 1})

    def test_matched_size_does_not_mislabel_another_rejection_as_size_mismatch(self):
        result = build_product_rerank_result(
            [
                {
                    "spu_id": 1001,
                    "sku_id": 2001,
                    "name": "M 码外套",
                    "category": "外套",
                    "size": "M",
                    "sale_price": 300,
                }
            ],
            {"intent": "recommendation"},
            "推荐外套",
            {"budget_max": 100},
            {"size_tool": {"recommended_size": "M"}},
        )

        self.assertEqual(result["rejected_reasons"], {"LOW_STYLE_SCORE": 1})

    @staticmethod
    def v3_constraint(field, operator, values, strength, weight=None):
        return {
            "id": f"constraint-{field}",
            "field": field,
            "operator": operator,
            "values": values,
            "strength": strength,
            "origin": "USER_EXPLICIT",
            "originTurnId": "turn-7",
            "derivedFromConstraintId": None,
            "scope": "ACTIVE_DEMAND",
            "weight": weight,
        }

    def test_bare_height_weight_pair_is_normalized_for_size_tool(self):
        normalized = normalize_measurement_query("明天面试想要显瘦 177 130 该怎么选")

        self.assertIn("身高177cm", normalized)
        self.assertIn("体重130斤", normalized)

        result = run_size_tool("明天面试想要显瘦 177 130 该怎么选")

        self.assertEqual(result["recommended_size"], "XL")
        self.assertEqual(result["measurements"]["height_cm"], 177.0)
        self.assertEqual(result["measurements"]["weight_jin"], 130.0)

    def test_height_with_kg_pair_overrides_previous_measurements(self):
        history = [
            {
                "user_query": "177 130 怎么穿？",
                "assistant_answer": "建议优先看 XL 码。",
            }
        ]

        normalized = normalize_measurement_query("160 60kg呢？")
        self.assertIn("身高160cm", normalized)
        self.assertIn("体重60kg", normalized)

        result = run_size_tool("160 60kg呢？", chat_history=history)

        self.assertEqual(result["recommended_size"], "M")
        self.assertEqual(result["measurements"]["height_cm"], 160.0)
        self.assertEqual(result["measurements"]["weight_jin"], 120.0)

    def test_partial_weight_question_only_fills_missing_height_from_history(self):
        history = [
            {
                "user_query": "177 130 怎么穿？",
                "assistant_answer": "建议优先看 XL 码。",
            }
        ]

        result = run_size_tool("60kg呢？", chat_history=history)

        self.assertEqual(result["measurements"]["height_cm"], 177.0)
        self.assertEqual(result["measurements"]["weight_jin"], 120.0)

    def test_same_thread_does_not_reuse_previous_size_state(self):
        thread_id = "size-switch-thread"
        embeddings = FakeEmbeddings()
        # CI has no developer-generated vector index; keep this state test independent of disk.
        with patch(
            "clothing_assistant.infrastructure.vector_store.get_embeddings",
            return_value=embeddings,
        ), patch(
            "clothing_assistant.infrastructure.vector_store.load_vector_data",
            return_value=[],
        ):
            first = run_langgraph_agent(
                "177 130 怎么穿？",
                thread_id=thread_id,
                request_id="req-size-switch-1",
                session_id="session-size-switch",
            )
            second = run_langgraph_agent(
                "160 60kg穿什么码呢？",
                chat_history=[
                    {
                        "user_query": "177 130 怎么穿？",
                        "assistant_answer": first["answer"],
                    }
                ],
                thread_id=thread_id,
                request_id="req-size-switch-2",
                session_id="session-size-switch",
            )

        size_result = second["debug"]["tool_results"]["size_tool"]
        self.assertEqual(size_result["recommended_size"], "M")
        self.assertEqual(size_result["measurements"]["height_cm"], 160.0)
        self.assertEqual(size_result["measurements"]["weight_jin"], 120.0)
        self.assertTrue(embeddings.queries)

    def test_build_product_refs_selects_only_java_candidates_matching_size(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "面试通勤西装外套",
                "category": "外套",
                "size": "XL",
                "stock_status": "in_stock",
                "style_tags": ["通勤", "显瘦"],
                "sale_price": 399,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "日常休闲衬衫",
                "category": "衬衫",
                "size": "M",
                "stock_status": "in_stock",
                "style_tags": ["日常"],
                "sale_price": 199,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "size_recommendation"},
            "明天面试想要显瘦的外套，177 130 穿什么码",
            {},
            {"size_tool": {"recommended_size": "XL"}},
            demand_intent={"version": "demand-intent-v2", "category": "外套"},
        )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["spu_id"], 1001)
        self.assertEqual(refs[0]["sku_id"], 2001)
        self.assertIn("XL 码匹配", refs[0]["reason"])

    def test_build_product_refs_uses_alternative_size_after_primary_size(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "基础款纯棉T恤",
                "category": "T恤",
                "size": "L",
                "stock_status": "in_stock",
                "style_tags": ["日常"],
                "sale_price": 99,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "修身短袖T恤",
                "category": "T恤",
                "size": "M",
                "stock_status": "in_stock",
                "style_tags": ["日常"],
                "sale_price": 109,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "size_recommendation"},
            "160 60kg，T恤穿什么码？",
            {},
            {"size_tool": {"recommended_size": "M", "alternative": "L"}},
            demand_intent={"version": "demand-intent-v2", "category": "T恤"},
        )

        self.assertEqual([ref["spu_id"] for ref in refs], [1002, 1001])
        self.assertIn("M 码匹配", refs[0]["reason"])
        self.assertIn("L 码是当前尺码建议的备选范围", refs[1]["reason"])

    def test_build_product_refs_does_not_recommend_for_chat_intent(self):
        refs = build_product_refs(
            [{"spu_id": 1001, "sku_id": 2001, "name": "通勤外套"}],
            {"intent": "chat"},
            "你是谁？",
            {},
            {},
        )

        self.assertEqual(refs, [])

    def test_build_product_refs_skips_invalid_and_duplicate_candidate_ids(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "通勤轻薄外套",
                "category": "外套",
                "stock_status": "in_stock",
                "style_tags": ["通勤"],
                "sale_price": 299,
            },
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "重复 SKU",
                "category": "外套",
                "stock_status": "in_stock",
                "style_tags": ["通勤"],
                "sale_price": 299,
            },
            {
                "spu_id": 1002,
                "name": "缺少 SKU 的候选",
                "category": "外套",
                "stock_status": "in_stock",
                "style_tags": ["通勤"],
                "sale_price": 199,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "推荐一件通勤外套",
            {},
            {},
        )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["spu_id"], 1001)
        self.assertEqual(refs[0]["sku_id"], 2001)

    def test_build_product_refs_explains_budget_color_and_scene_matches(self):
        candidates = [
            {
                "spu_id": 1002,
                "sku_id": 2101,
                "name": "通勤轻薄外套",
                "category": "外套",
                "color": "黑色",
                "stock_status": "in_stock",
                "season": ["autumn"],
                "style_tags": ["commute"],
                "sale_price": 299,
            }
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "秋季通勤，想要黑色外套",
            {"budget_max": 300, "preferred_colors": ["黑色"]},
            {},
        )

        self.assertEqual(len(refs), 1)
        reason = refs[0]["reason"]
        self.assertIn("当前候选显示有库存", reason)
        self.assertIn("风格、季节或场景标签匹配", reason)
        self.assertIn("价格 299 在预算 300 内", reason)
        self.assertIn("黑色匹配颜色偏好", reason)

    def test_rag_chunks_enrich_reason_without_overriding_candidate_facts(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "通勤轻薄外套",
                "category": "外套",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["commute"],
                "sale_price": 299,
            }
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "推荐一件通勤外套",
            {"budget_max": 300},
            {
                "rag_tool": {
                    "retrieved_chunks": [
                        {
                            "file_name": "颜色选择.txt",
                            "content": "通勤场景优先选择黑色、灰色、藏青色等基础色，更容易搭配。",
                            "score": 0.1,
                        }
                    ]
                }
            },
        )

        self.assertIn("RAG 知识提示", refs[0]["reason"])
        self.assertIn("通勤场景优先选择黑色", refs[0]["reason"])
        self.assertIn("价格 299 在预算 300 内", refs[0]["reason"])

    def test_rerank_uses_java_contract_field_aliases(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "秋季T恤",
                "category_name": "T恤",
                "color": "白色",
                "stock_status": "in_stock",
                "seasons": ["summer"],
                "sale_price": 129,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "轻薄夹克",
                "category_name": "外套",
                "color": "黑色",
                "stock_status": "in_stock",
                "seasons": ["autumn"],
                "sale_price": 299,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "推荐一件秋季外套",
            {},
            {},
        )

        self.assertEqual(refs[0]["spu_id"], 1002)
        self.assertIn("风格、季节或场景标签匹配", refs[0]["reason"])

    def test_behavior_context_boosts_recent_interest_candidate(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "通勤轻薄外套",
                "category": "外套",
                "stock_status": "in_stock",
                "style_tags": ["commute"],
                "sale_price": 299,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "休闲卫衣",
                "category": "卫衣",
                "stock_status": "in_stock",
                "style_tags": ["casual"],
                "sale_price": 199,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "推荐一件外套",
            {
                "recent_interest_spu_ids": [1001],
                "behavior_preferred_categories": ["外套"],
                "behavior_preferred_styles": ["commute"],
            },
            {},
        )

        self.assertEqual(refs[0]["spu_id"], 1001)
        self.assertIn("近期行为显示你关注过类似商品", refs[0]["reason"])

    def test_fuzzy_student_budget_query_prefers_matching_candidate(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "基础百搭卫衣",
                "category": "卫衣",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["casual", "basic"],
                "sale_price": 199,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "正式商务西装",
                "category": "西装",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["formal"],
                "sale_price": 899,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "学生党想要平价百搭，预算500以内",
            {},
            {},
        )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["spu_id"], 1001)
        self.assertIn("风格、季节或场景标签匹配", refs[0]["reason"])
        self.assertIn("价格 199 在预算 500 内", refs[0]["reason"])
        self.assertIn("符合平价优先", refs[0]["reason"])

    def test_router_treats_fuzzy_profile_as_recommendation(self):
        result = intent_router("学生党平价百搭")

        self.assertEqual(result["intent"], "recommendation")

    def test_router_treats_synonym_profile_as_recommendation(self):
        result = intent_router("大学生日常上课，别太贵，还要遮肉显腿长")

        self.assertEqual(result["intent"], "recommendation")

    def test_router_treats_gender_outfit_and_skirt_queries_as_recommendation(self):
        for query in ["男性穿搭", "女性穿搭", "女生通勤半裙", "裙子"]:
            with self.subTest(query=query):
                result = intent_router(query)

                self.assertEqual(result["intent"], "recommendation")

    def test_skirt_query_prefers_skirt_candidate(self):
        candidates = [
            {
                "spu_id": 1120,
                "sku_id": 11201,
                "name": "通勤百褶半裙",
                "category": "半裙",
                "stock_status": "in_stock",
                "style_tags": ["commute"],
                "attribute_tags": ["场景:通勤", "适用性别:female"],
                "sale_price": 189,
            },
            {
                "spu_id": 1002,
                "sku_id": 10021,
                "name": "通勤轻薄外套",
                "category": "外套",
                "stock_status": "in_stock",
                "style_tags": ["commute"],
                "attribute_tags": ["场景:通勤", "适用性别:unisex"],
                "sale_price": 299,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "女性裙子推荐",
            {},
            {},
            demand_intent={"version": "demand-intent-v2", "category": "半裙"},
        )

        self.assertEqual(refs[0]["spu_id"], 1120)
        self.assertIn("风格、季节或场景标签匹配", refs[0]["reason"])

    def test_demand_intent_terms_drive_ranking_when_query_is_generic(self):
        candidates = [
            {
                "spu_id": 1120,
                "sku_id": 11201,
                "name": "通勤百褶半裙",
                "category": "半裙",
                "stock_status": "in_stock",
                "style_tags": ["commute"],
                "attribute_tags": ["场景:通勤", "适用性别:female"],
                "sale_price": 189,
            },
            {
                "spu_id": 1002,
                "sku_id": 10021,
                "name": "通勤轻薄外套",
                "category": "外套",
                "stock_status": "in_stock",
                "style_tags": ["commute"],
                "attribute_tags": ["场景:通勤", "适用性别:unisex"],
                "sale_price": 299,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "帮我推荐一下",
            {},
            {},
            demand_intent={
                "targetGender": "female",
                "category": "半裙",
                "scene": ["commute"],
                "style": ["commute", "minimal"],
                "budgetMax": 200,
                "attributes": [],
            },
        )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["spu_id"], 1120)
        self.assertIn("风格、季节或场景标签匹配", refs[0]["reason"])
        self.assertIn("价格 189 在预算 200 内", refs[0]["reason"])

    def test_visual_goal_uses_java_attribute_tags(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "中高腰直筒牛仔裤",
                "category": "牛仔裤",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["casual"],
                "attribute_tags": ["下装版型:直筒", "腰线:中高腰"],
                "sale_price": 259,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "宽松低腰休闲裤",
                "category": "休闲裤",
                "color": "米色",
                "stock_status": "in_stock",
                "style_tags": ["casual"],
                "attribute_tags": ["下装版型:宽松", "腰线:低腰"],
                "sale_price": 229,
            },
        ]

        result = build_product_rerank_result(
            candidates,
            {"intent": "recommendation"},
            "想要显高显瘦的裤子",
            {},
            {},
        )

        self.assertEqual(result["product_refs"], [])
        self.assertEqual(result["candidate_scores"][0]["spu_id"], 1001)
        self.assertIn("版型或腰线更利于拉长比例", result["candidate_scores"][0]["score_parts"])
        self.assertIn("颜色、线条或版型更贴合显瘦需求", result["candidate_scores"][0]["score_parts"])

    def test_synonym_profile_scores_student_and_visual_reasons(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "黑色中高腰直筒裤",
                "category": "裤子",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["casual", "basic"],
                "attribute_tags": ["腰线:中高腰", "下装版型:直筒"],
                "sale_price": 199,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "米色低腰宽松裤",
                "category": "裤子",
                "color": "米色",
                "stock_status": "in_stock",
                "style_tags": ["formal"],
                "attribute_tags": ["腰线:低腰", "版型:过度宽松"],
                "sale_price": 299,
            },
        ]

        result = build_product_rerank_result(
            candidates,
            {"intent": "recommendation"},
            "适合大学生日常上课，别太贵，看起来显腿长一点，还要遮肉",
            {},
            {},
        )

        self.assertEqual(result["product_refs"], [])
        self.assertEqual(result["candidate_scores"][0]["spu_id"], 1001)
        self.assertIn("价格和风格更适合学生日常穿搭", result["candidate_scores"][0]["score_parts"])
        self.assertIn("版型或腰线更利于拉长比例", result["candidate_scores"][0]["score_parts"])
        self.assertIn("颜色、线条或版型更贴合显瘦需求", result["candidate_scores"][0]["score_parts"])

    def test_java_demand_intent_affects_rerank_when_query_is_vague(self):
        candidates = [
            {
                "spu_id": 1001,
                "sku_id": 2001,
                "name": "黑色中高腰直筒裤",
                "category": "裤子",
                "color": "黑色",
                "stock_status": "in_stock",
                "style_tags": ["casual"],
                "attribute_tags": ["腰线:中高腰", "下装版型:直筒"],
                "sale_price": 199,
            },
            {
                "spu_id": 1002,
                "sku_id": 2002,
                "name": "米色低腰商务裤",
                "category": "裤子",
                "color": "米色",
                "stock_status": "in_stock",
                "style_tags": ["formal"],
                "attribute_tags": ["腰线:低腰", "风格:商务正装"],
                "sale_price": 899,
            },
        ]

        result = build_product_rerank_result(
            candidates,
            {"intent": "recommendation"},
            "看看这个",
            {},
            {},
            demand_intent={
                "scene": ["campus", "daily"],
                "style": ["casual"],
                "attributes": ["平价", "显瘦", "显高"],
            },
        )

        self.assertEqual(result["product_refs"][0]["spu_id"], 1001)
        self.assertEqual(result["semantic_preferences"]["price_preference"], "budget")
        self.assertIn("slimmer", result["semantic_preferences"]["visual_goals"])
        self.assertIn("taller", result["semantic_preferences"]["visual_goals"])
        self.assertIn("版型或腰线更利于拉长比例", result["product_refs"][0]["reason"])

    def test_winter_warm_query_prefers_real_warm_candidates(self):
        candidates = [
            {
                "spu_id": 1119,
                "sku_id": 11192,
                "name": "极简通勤Polo衫",
                "category": "T恤",
                "stock_status": "in_stock",
                "season": ["autumn", "all_season"],
                "style_tags": ["minimal", "casual"],
                "attribute_tags": ["适用场景:通勤", "厚度:常规"],
                "sale_price": 139,
            },
            {
                "spu_id": 1110,
                "sku_id": 11102,
                "name": "约会A字半裙",
                "category": "半裙",
                "stock_status": "in_stock",
                "season": ["autumn"],
                "style_tags": ["date", "commute"],
                "attribute_tags": ["适用场景:约会社交", "厚度:常规"],
                "sale_price": 189,
            },
            {
                "spu_id": 1116,
                "sku_id": 11162,
                "name": "羊毛保暖针织衫",
                "category": "针织衫",
                "material": "羊毛",
                "stock_status": "in_stock",
                "season": ["winter"],
                "style_tags": ["minimal"],
                "attribute_tags": ["厚度:厚款", "材质特征:保暖"],
                "sale_price": 279,
            },
        ]

        refs = build_product_refs(
            candidates,
            {"intent": "recommendation"},
            "秋冬保暖的?",
            {},
            {},
        )

        self.assertEqual([ref["spu_id"] for ref in refs], [1116])
        self.assertIn("匹配冬季需求", refs[0]["reason"])
        self.assertIn("材质或厚度更适合保暖", refs[0]["reason"])

    def test_winter_warm_query_returns_no_refs_without_strong_match(self):
        refs = build_product_refs(
            [
                {
                    "spu_id": 1119,
                    "sku_id": 11192,
                    "name": "极简通勤Polo衫",
                    "category": "T恤",
                    "stock_status": "in_stock",
                    "season": ["autumn", "all_season"],
                    "style_tags": ["minimal"],
                    "attribute_tags": ["厚度:常规"],
                    "sale_price": 139,
                }
            ],
            {"intent": "recommendation"},
            "秋冬保暖的?",
            {},
            {},
        )

        self.assertEqual(refs, [])


if __name__ == "__main__":
    unittest.main()
