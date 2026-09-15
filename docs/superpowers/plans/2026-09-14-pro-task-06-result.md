# Task 6：Pro 结构化决策与动态循环

本任务基于 Task 5 提交 `3354de7`，在隔离分支 `codex/pro-task-06` 实现。范围为 Python 内部动态执行器、模型决策适配、运行预算配置及行为测试。

## 验收场景

同一句“预算内推荐有货的外套，并提供尺码和退换依据”，模型每次读取当前工具观察结果，再决定下一步。

- 候选 A 有货：可进入完成提案路径。
- 候选 A 无货：继续查询候选 B，然后根据 B 的结果重新决策。
- 关键数据不足：发起澄清并停止新工具调用。
- 临时依赖错误：只做一次有限重试，每次实际尝试均消耗预算。
- 达到决策、工具或时间上限，或者收到取消：停止新增工作，保留已有证据和未完成需求。

动态循环的完成提案仍需要 Task 7 核验。工具调用成功不自动证明商品符合全部条件，模型提供的完成建议不构成可信事实。

## 实现接口与限制

- `decision.py` 调用现有模型工厂，复用并发信号量和安全错误分类。单次等待及模型请求均使用剩余时间，保留模型消息的 usage 元数据，行动按 JSON 和既有 `parse_action` 严格解析。
- `executor.py` 提供 `run_pro_agent(query, *, decide, execute_tool, clock, limits, context)`，用 LangGraph 连接决策、执行和待校验节点。测试可注入决策器、工具和时钟，默认工具执行经过 Task 5 的六类适配器。
- 模型只收到只读的需求、明确筛选条件、筛选后的用户资料与历史、商品引用和工具观察。Java 运行标识用于内部证据关联，凭证及用户身份不进入模型快照。
- 每次真实模型调用与工具尝试都计数，默认最多 12 次决策、8 次工具调用、90 秒。`PRO_MAX_MODEL_CALLS`、`PRO_MAX_TOOL_CALLS`、`PRO_TOTAL_SECONDS` 可以降低上限。成功的重复查询使用本轮缓存，库存缓存满 15 秒可以重新查询。
- 搜索工具会补入明确筛选条件，拒绝模型扩大预算或改写用户明确的筛选条件。需求完成状态复用 Task 4 的证据关联检查；本阶段不把模型自报成功当作验证完成。
- 内部返回 `public_approved=False`；完成路径使用 `stop_reason=validation_required` 和 `proposal`，这是内部中间结果，不是可直接发送的 v2 done 响应。

原始复合问题完整保留在 `query`，模型结合它和工具结果连续选取行动。当前需求记录包含原问题和明确筛选条件；尚不声称已经把自由文本中的每个子需求独立拆分并逐项认证。注入的同步测试回调必须遵守 timeout；取消会阻止后续新调用。

质量审查用本地慢速响应复现：仅设置 HTTP 超时不能保证整个调用按截止时间返回。默认模型与工具路径因此增加整段调用的等待上限及后台并发数量限制。调用方超时后会停止等待，已经发出的同步请求可能继续到结束；模型并发名额在底层请求真正结束后才释放。这保证调用方及时停止，同时避免通过超时绕过并发限制。

## 后续 Task 7

Python 校验证据、需求状态和硬条件；Java 按本轮运行登记记录校验商品来源并重新读取价格、库存等事实。最终商品卡片由 Java 真实数据组装。价格变化、无库存或无退换政策证据时，文本、卡片与需求完成状态需要一致降级。

本任务不启用 v2 API 或前端 Pro 按钮。Python v2 API 接入与前端展示仍按后续计划实施。

## 验证

- Task 5 基线：Pro state/tools/contract 与 Lite pipeline/MVP 共 97 项通过。
- 测试先行：最初 13 项断言因执行器尚未实现而失败，再实现后转绿；统计完整性、已分类工具异常等边界修复均先补回归测试。
- 专项 `tests/test_pro_executor.py`：42 项通过，包括有货/无货分支、政策有据/无据跨工具分支、Task 5 Java MockTransport 实际适配、缓存刷新、预算、取消、JSON 修正、重试和凭证过滤，以及整段 I/O 等待截止时间、并发名额保留和过期任务禁止启动。
- 相关回归 `tests/test_pro_executor.py tests/test_pro_state.py tests/test_pro_tools.py tests/test_pro_contract.py tests/test_agent_pipeline.py tests/test_agent_mvp.py tests/test_config_data.py tests/test_llm_client.py`：155 项通过、13 个 subtests 通过。
- 全量 `python -m pytest -q`：409 项通过、1 项失败、104 个 subtests 通过。失败仍为既有 `RecommendationServiceTests.test_same_thread_does_not_reuse_previous_size_state` 的 `embeddings.queries` 空列表断言，与 Task 4/5 记录的基线失败一致。
- Ruff、compileall 通过；interrogate 文档覆盖率 35.7%，通过 30% 门槛。
- 本轮以 fake decide、fake clock、模拟模型消息和本地 HTTP transport 验证，未调用真实模型或真实 Java 服务。默认模型适配已接线，其真实服务联调仍留待后续 API 接入阶段。

后续提交继续使用 `英文类型: 中文说明`，本任务提交标题为 `feat: 增加 Pro 动态决策循环`。

2026-09-15 收尾：规格复审与质量复审均通过；本次没有调用真实模型服务，也没有合并或推送分支。
