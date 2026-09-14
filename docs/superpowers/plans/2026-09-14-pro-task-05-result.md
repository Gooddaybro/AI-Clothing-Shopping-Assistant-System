# Task 5：Pro 六类工具适配结果

本次完成计划中的 Task 5，位于隔离分支 `codex/pro-task-05`，基于 Task 4 的 `8cb3cf7`。本次只增加 Pro 工具适配层和行为测试，不修改 Lite 入口、Lite 工具或动态决策循环。

## Git 提交规范

后续提交统一采用：

```text
<英文类型>: <中文说明>
```

类型沿用 Conventional Commits 的常用集合：`feat`（功能）、`fix`（修复）、`test`（测试）、`docs`（文档）、`refactor`（重构）、`chore`（工程维护）。一个提交只对应一个任务，中文说明使用简短的动作句，不写入 token、密码或其他敏感信息。

本任务的提交信息为：

```text
feat: 增加 Pro 工具适配
```

示例：`fix: 修复库存结果映射`、`test: 补充工具超时测试`、`docs: 更新 Pro 接入说明`。

## 实现内容

- `infrastructure/java_tool_client.py` 提供固定地址、固定运行路径和双凭证请求的 Java 网关客户端，只允许 `search_products`、`get_product_detail`、`check_availability` 三个动作。模型不能指定 URL、用户 ID 或其他动作，凭证只进入请求头，不进入结果和 `repr`。
- Java HTTP 超时、连接错误、非 JSON、鉴权失败和上游错误会归一到安全的 `ToolResult`。库存为 0 的有效 SKU 保持 `ok`，不存在的 SKU 保持 `empty`；成功结果缺少有效证据时拒绝为权威事实。
- `agent/pro/tools.py` 提供六类工具的统一 `execute_tool` 入口。Java 工具经过参数模型和客户端白名单校验；RAG、政策、尺码能力通过 bounded 线程池适配，单次调用有最多 5 秒的工具期限，避免重复超时创建无界后台线程。
- 商品知识检索会附加目标 SPU 范围并保留检索证据；政策没有来源时返回 `no_evidence`；尺码输入必须明确厘米和千克，裸数字会返回 `needs_input`。尺码偏好会传给旧能力；只有真实商品尺码表证据才能使用 `product_chart`，否则返回 `generic_rule` 和限制说明。
- `ToolResult` 统一包含 `status`、`data`、`evidence_id`、`source`、`missing_fields`、`error_code`，并对结果数据做深层只读保护，避免后续决策阶段篡改事实或证据。

新增文件：

- `clothing_assistant/infrastructure/java_tool_client.py`
- `clothing_assistant/agent/pro/tools.py`
- `tests/test_pro_tools.py`

## 验证结果

- `python -m pytest tests/test_pro_tools.py -q`：18 项通过。
- `python -m pytest tests/test_pro_tools.py tests/test_agent_pipeline.py tests/test_agent_mvp.py -q`：41 项通过，Lite 相关回归通过。
- `python -m pytest -q`：367 项通过、1 项失败、104 个 subtests 通过。唯一失败是既有 Lite 测试 `RecommendationServiceTests.test_same_thread_does_not_reuse_previous_size_state` 对 `embeddings.queries` 的断言；该失败在 Task 4 基线也能复现，和本次新增适配器无关，因此不宣称全量绿色。
- `ruff check clothing_assistant tests`、`python -m compileall -q clothing_assistant tests` 通过；`interrogate -i --fail-under=30 clothing_assistant` 为 35.0%，通过 30% 门槛。
- 测试使用本地 `httpx.MockTransport` 和注入的 RAG/政策/尺码 runner，没有连接真实 Java 或 embedding 服务，也没有新增依赖。

## Task 6 下一步：结构化决策与动态循环

Task 6 将新增 `agent/pro/decision.py`、`agent/pro/executor.py`，并补充 Pro 预算配置和 `tests/test_pro_executor.py`。核心是让执行器根据每轮只读状态和工具观察结果重新决策：库存为 0 时切换候选，库存大于 0 时才进入完成路径；需要补充信息时澄清并结束；完成前再走校验路径。

它还需要接入现有模型工厂的独立 JSON-only prompt 和 `parse_action`，限制为最多 12 次决策、8 次工具调用、90 秒，支持取消、去重缓存和一次临时依赖重试，保留 usage 与证据。Task 6 完成后才具备动态工具调度；最终商品事实校验和商品卡片仍属于后续 Task 7。
