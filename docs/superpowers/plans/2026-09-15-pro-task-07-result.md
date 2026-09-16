# Task 7：最终事实校验与一致推荐结果

本任务接续 Python `a945c4b` 与 Java `143a728`，两端隔离分支均为 `codex/pro-task-07`。

## 责任边界

Python 对 Task 6 的内部提案做证据、运行关联及明确条件检查，生成 v2 done 协议结果，供后续 Task 8 接口调用。Java 以本轮运行登记记录为来源边界，重新读取商品价格、库存等事实后组装最终卡片。Python 的完成状态不代表 Java 已批准公开展示。

政策缺少证据、通用尺码规则、商品引用失效或条件无法确认时，需求状态和文字必须明确保留不确定性。当前原问题尚未被逐项拆成已认证的子需求，因此不能仅凭商品查询成功把整句复合需求标为全部满足。

## 本次实现

- Python 新增 `validation.py`：绑定 `request_id/thread_id/run_id`，只接受本轮、来源正确且可追溯的 evidence；拒绝假 SKU、跨 run 引用、SPU/SKU 错配、无库存和超预算商品；清理模型伪造的价格、图片、URL、政策承诺，并保留有证据的尺码建议及限制说明。
- Java 新增 `ProRecommendationValidator`：从 Redis run ledger 验证商品对，绕过缓存重新读取 SQL 商品快照和 MySQL 价格库存；最终卡片只使用 Java 商品事实，失败终止状态不出卡，需求缺失时保留原请求为 `unconfirmed`。
- Java `ProAssistantService` 在撤销运行凭证前完成校验，`RecommendationCandidateQueryService.findFreshCandidates` 提供不走召回缓存的最终复核入口；Lite 路径保持不变。

## 验证记录

- Python Task7 及相关 Pro 测试：`147 passed`；`ruff check clothing_assistant tests` 和 `compileall` 通过。
- Python 全量：`440 passed, 1 failed, 104 subtests passed`。唯一失败是既有 Lite 测试 `RecommendationServiceTests.test_same_thread_does_not_reuse_previous_size_state`，与本任务文件无关。
- Java Task7、Pro 服务、候选查询和控制器定向测试：`20 tests, 0 failures`。
- Java `mvnw.cmd -DskipTests verify`：构建成功，Checkstyle `0` 个违规。全量 `verify` 的 2 个错误来自当前环境没有 Docker 的 Testcontainers 测试，另有 11 个跳过，非 Task7 断言失败。

代码提交：Python `686c3f9`、Java `9b7a826`，均遵循 `feat: 中文说明` 格式。

## 完成 Task 7 后还剩什么

| 阶段 | 必须完成的工作 | 用户能看到的变化 |
| --- | --- | --- |
| Task 8 | Python v2 同步、流式 API；连接执行器和校验器；输出真实工具进度并传播取消 | Python 能通过接口运行 Pro，尚需 Java 和前端接通 |
| Task 9 | Java v2 SSE、最终校验、结果持久化和撤销凭证；防重复终结与断开处理 | Java 可安全转发进度并保存最终回答 |
| Task 10 | 前端 Lite/Pro 切换、进度状态、最终卡片和详情跳转 | 可以在现有聊天界面实际选择 Pro 并查看商品 |
| Task 11 | 固定行为场景、真实模型决策评测、跨服务联调、E2E、Lite 对比报告与回归 | 有证据判断功能完整性和实际模型效果，再在测试环境启用 |

Pro Max 多 Agent 不在这轮实现范围。Task 7 提供卡片校验和组装能力，页面上的卡片展示仍由 Task 10 完成。
