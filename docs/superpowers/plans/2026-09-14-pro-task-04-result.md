# Task 4：Python 行动协议与每轮状态

本次仅完成计划中的 Task 4，位于隔离分支 `codex/pro-task-04`，基于 Task 1 的 `4003b53`。Java Task 2、Task 3 在 Java 仓库独立分支中保留。本次不启用 Pro API，不修改 Lite，不接入模型或工具执行循环。

## 实现与使用边界

- `schemas.py` 提供 `parse_action(payload)`，按 kind 区分 tool / clarify / finish，再按 tool 区分六种工具参数。Pydantic 严格校验未知字段、参数类型和行动互斥，不执行模型生成的代码。
- `state.py` 提供 `new_state(run_id, query, hard_constraints)`，保留原始用户问题及来源。当前 hard_constraints 入参仅用于可信初始化的明确筛选条件，逐项保存来源；用户文本中的多需求提取仍属于后续执行器。
- 硬约束脱离调用方传入的可变对象并递归冻结，运行顶层只读；需求、证据、软约束集合每次新建。
- `apply_requirement_updates(state, updates)` 仅接收需求 ID、状态和证据 ID 建议，不允许模型改写需求正文、来源或硬约束。
- 升级为 satisfied 必须引用本轮、状态为 ok、关联该需求的全部证据。空证据、其他运行、无关证据以及失败/空结果均不能用来标记完成；不能验证的完成建议保持原状态。

证据与需求的关联必须由后续可信的校验流程建立。本次只定义并检查关联结构，不判定商品确实满足预算、库存或尺码条件，也不把模型提交的商品引用变成可信事实。这些业务校验仍按计划由后续任务实现。

## 验证

- 基线 `tests/test_pro_contract.py tests/test_agent_pipeline.py tests/test_agent_mvp.py`：37 项通过。
- Task 4 专项 `python -m pytest tests/test_pro_state.py -q`：42 项通过。先观察模块缺失的红灯；金额兼容与行动上限新增测试先 8 项失败，修复后通过。
- 最终相关回归 `python -m pytest tests/test_pro_state.py tests/test_pro_contract.py tests/test_agent_pipeline.py tests/test_agent_mvp.py -q`：79 项通过。
- 全量 `python -m pytest -q`：349 passed、1 failed、104 subtests passed。失败为旧 `RecommendationServiceTests.test_same_thread_does_not_reuse_previous_size_state` 的 `embeddings.queries` 为空断言；尺码及新测量值断言通过。使用 `git archive HEAD` 导出未修改的 `4003b53` 到 `.test_tmp/task04-baseline`，相同环境运行全量基线得到 307 passed、1 failed、104 subtests passed，复现同一失败。因此不将它误报为 Task 4 回归，也不声称全量绿色。
- `python -m compileall -q clothing_assistant tests` 和 `ruff check clothing_assistant tests` 通过；interrogate 文档覆盖率 34.5%，通过项目 30% 门槛。
- 测试使用项目现有 Python 3.13.12 虚拟环境；全量运行设置 `PYTHON_DOTENV_DISABLED=1`、`OUTFIT_CONTRACT_ROOT` 指向本分支 contracts。没有安装新依赖。

## Task 5 下一步：六类工具适配

新增 `clothing_assistant/infrastructure/java_tool_client.py`、`clothing_assistant/agent/pro/tools.py` 和 `tests/test_pro_tools.py`，接入：

| 工具 | 数据来源与用途 |
| --- | --- |
| search_products | Java：查询候选商品 |
| get_product_detail | Java：读取商品详情 |
| check_availability | Java：查询指定颜色、尺码的 SKU 库存 |
| search_product_knowledge | 现有知识检索：返回带来源的商品知识 |
| search_policy | 现有政策检索：返回带来源的政策依据 |
| recommend_size | 现有尺码能力：明确单位和建议依据 |

需要完成的边界：

- Java 客户端使用固定服务地址和服务/运行凭证，不接受模型指定 URL、用户 ID 或任意动作；凭证不进入模型输入或日志。
- 所有适配器统一返回 status、data、evidence_id、source、missing_fields、error_code。
- 区分库存为 0、SKU 不存在、依赖不可用和缺少证据；不把空结果当错误，也不虚构事实。
- 商品知识带目标商品范围和证据；尺码输入显式使用厘米/千克，有歧义则追问。没有真实商品尺码表时只能报告 generic_rule 及其限制。
- 使用本地测试替身验证超时、鉴权、边界与结果转换，并回归 Lite 的 pipeline / MVP 测试。

Task 5 完成后，工具才具有实际调用适配；根据中间结果连续决定下一步的动态执行循环属于 Task 6。
