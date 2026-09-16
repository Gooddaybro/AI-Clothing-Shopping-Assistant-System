# Lite / Pro Dynamic Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 Lite 行为的前提下，交付能够动态调用六类工具、处理复合需求并展示真实商品卡片的 Pro。

**Architecture:** Java 保留用户授权、数据事实、会话和最终商品校验，Python 新增独立 LangGraph 决策循环。Lite 继续 v1，Pro 使用明确版本化的 v2 协议和本轮动态候选登记；前端共享商品展示组件。

**Tech Stack:** 现有 Spring Boot / Java、FastAPI / Pydantic / LangGraph、React / TypeScript、pytest、Maven verify、Vitest、Playwright。使用现有模型客户端，不新增 Agent 框架。

---

## 0. 计划状态与执行约定

2026-09-11：用户确认设计并要求编写实施计划。本次仅新增计划与更新设计中的核实信息，未实施运行代码，未运行产品测试。

设计来源：`../specs/2026-09-10-lite-pro-dynamic-agent-design.md`。

此功能是跨 Java、Python、前端的一条纵向业务链，按可独立验证的阶段依次实施，不能只交付 Python 循环就宣布 Pro 完成。Pro Max 不实施。每项任务先写行为测试、运行确认失败，再实现、运行确认通过，最后只提交本任务文件。不要提交现有 IDE、学习代码或其他未跟踪文件。

执行前为涉及的仓库创建隔离工作区；遵循 using-git-worktrees 技能。以下根目录用于定位源文件，创建工作区后将命令目录替换为对应工作区，不能跨目录误改原检出。共享契约是真实独立目录，不是当前工作区内那个不完整同名目录。

| 缩写 | 绝对根目录 |
| --- | --- |
| P | `D:/git/推荐系统/AI Clothing Shopping Assistant System` |
| J | `D:/git/推荐系统/Intelligent Outfit Recommendation System` |
| C | `D:/git/outfit-project-contract` |
| B | `J/backend/src/main/java/com/recommendation/intelligentoutfitrecommendationsystem` |
| JT | `J/backend/src/test/java/com/recommendation/intelligentoutfitrecommendationsystem` |

下文 `P/...`、`B/...` 等均按此表展开。Java 修改前读已核实存在的 `J/docs/commenting-guidelines.md`。新增 Java 类型写清职责和权限边界的 Javadoc。

## 1. 核实结论与实现决策

1. 找到并读取 `C/AGENTS.md`、`docs/business-rules.md`、`docs/coding-boundary.md`、`docs/dev-checklist.md`、`contracts/assistant-streaming-chat/v1.md`。共享规则要求跨项目变更先定义契约；责任或事件语义变化使用新版本。
2. 当前 Python `/chat`、`/chat/stream` 在 `P/clothing_assistant/api/app.py` 直接调用现有 LangGraph。新增 Pro 入口分支，旧测试和旧入口保留。
3. Java `AssistantContextService.buildContext` 包含旧意图解析，`AssistantService` 会在澄清问题存在时提前返回。Pro 必须在该流程之前分流，不能只替换 Python router。
4. Java `RecommendationDecisionService` 接收初始候选。Pro 新增专用结果校验入口，候选来自 Java 登记的本轮查询结果，不能把 Python 上报的候选当事实。
5. 内部接口已有 `X-Internal-Token`，由 `common/internal/InternalApiInterceptor.java` 检查。沿用服务鉴权，并增加本轮 run token 绑定上下文；模型不可见服务密钥。
6. `RecommendationCandidateQuery` 支持 category/style/season/material/fit/budgetMax/gender/recallText，没有颜色、尺码筛选字段。第一版允许搜索后按 Java SKU 事实筛选，不声称颜色、尺码已在搜索阶段过滤。
7. `ProductDetail` 有图片、属性、minPrice/maxPrice，没有专用尺码表。详情价格区间不等于选中 SKU 售价；最终卡片以真实 SKU 数据为准。通用尺码结果必须标记依据限制。
8. 当前 Python LLM 包装返回文本，不能直接当作已支持 tool calling。第一版使用结构化 JSON 行动协议 + Pydantic 校验，通过现有模型工厂注入；不依赖供应商未验证的原生工具调用功能。
9. 前端已有 `ChatPanel`、SSE parser、`ProductCard` 与测试，可扩展，不重建聊天页面。

## 2. v2 数据协议草案（Task 1 固化为权威契约）

### 模式与版本

前端新增 `agentMode: "lite" | "pro"`，缺省 lite；非法值拒绝。Pro 使用新公共端点 `/api/assistant/v2/chat`、`/api/assistant/v2/chat/stream`；Java 调 Python `/v2/chat`、`/v2/chat/stream`。旧公共端点与 Python 旧入口继续 v1/Lite，不向旧消费者发 progress。新入口只接收 pro，避免版本和模式含义冲突。

前端切换只改变下一轮请求。内部状态按 user/session/mode/run 隔离，历史消息仍由 Java 提供。首期 Pro 不依赖跨请求 checkpoint 恢复；每轮重新装配可信历史，不复用上轮库存。用户明确要求继续的需求可从历史解析，模型推断保留来源。

Java→Python 示例（字段值为示例，不能作为测试外真实身份）：

```json
{
  "contract_version": "assistant-v2",
  "agent_mode": "pro",
  "request_id": "req-1",
  "run_id": "run-1",
  "thread_id": "thread-1",
  "query": "300元内的通勤外套，175cm、70kg，要有货，能退换",
  "chat_history": [],
  "user_context": {"user_id": 1, "height_cm": 175, "weight_kg": 70},
  "explicit_filters": {"budget_max": "300.00"},
  "current_product_refs": [],
  "tool_run_token": "server-generated-opaque-value"
}
```

`tool_run_token`、用户标识及鉴权不进入模型 prompt、trace 或前端。模型只看到与需求相关的最少上下文。金额 v2 用十进制字符串和 CNY；Java 适配使用 BigDecimal，不能靠猜测乘除 100。Task 2 用当前数据库/Mapper 测试固定 299.90 与 300.00 边界，旧 v1 单位不变。

### 模型行动

每次返回一个 JSON 对象；未知字段拒绝。`tool` 动作选择白名单工具，参数按工具 schema 校验；`clarify` 与 `finish` 不携带工具调用。

```json
{"kind":"tool","tool":"check_availability","arguments":{"spu_id":101,"color":"black","size":"L"}}
```

```json
{"kind":"clarify","question":"你希望这件外套合身还是宽松？"}
```

```json
{"kind":"finish","answer":"找到一件符合预算且L码有货的外套；退换条件暂未确认。","product_refs":[{"spu_id":102,"sku_id":202,"reason":"符合预算和通勤需求"}]}
```

需求清单在初始化时提取，包含 id/text/status/source；source 指向原消息或明确筛选条件。约束分 HARD/SOFT，HARD 不允许工具后改写。需求完成建议可随行动返回，但只有对应证据存在才升级为已验证；模型自报完成不等于事实成立。商品描述和检索文本一律作为数据，不作为调度指令。

### 工具结果与预算

```json
{"status":"ok","data":{"spu_id":101,"sku_id":201,"available_stock":0},"evidence_id":"ev-1","source":"java","missing_fields":[],"error_code":null}
```

status 枚举：ok / empty / needs_input / unavailable / no_evidence / forbidden。库存为 0 是 ok，超时是 unavailable。失败不能伪装为空结果。

首期服务端可配置默认：最多 12 次决策、8 次逻辑工具调用、总时限 90 秒、每个暂时失败最多重试 1 次、一次 JSON 修正机会。每次尝试均记录；重试也计入总预算。内部 HTTP 调用设独立 5 秒上限并受剩余总时限约束，模型调用上限 20 秒并受剩余总时限约束；没有剩余预算就返回部分结果。库存工具包含的两个 HTTP 请求都记入 trace。以上是待实测调优的起始配置，不是性能承诺。Java v2 SSE 120 秒超时覆盖 Python 90 秒执行和收尾。

同工具同参数成功结果在本轮缓存；短暂失败可重试一次；库存成功结果超过 15 秒可在预算内复查。拒绝无意义重复调用但将原因反馈模型。前端断开后传播取消，不启动新工具，未完成答案不入库。

### 动态候选可信登记

Java 在 Pro 请求开始创建 runId 和随机 run token，绑定 userId/threadId/requestId、10 分钟 TTL。使用现有 Redis 能力存储，支持多 Java 实例；Redis 不可用时 Pro 明确失败，不回退成不校验。

新增 `/internal/assistant/runs/{runId}/tools/{toolName}`，只支持 search_products/get_product_detail/check_availability。同时验证内部服务 token 与 run token，工具参数不接受 userId、URL、SQL。网关复用现有商品服务；实际返回的 SPU/SKU 及价格快照写入 Java run ledger。每轮最多登记 200 个 SKU，每次搜索最多返回 20 个，避免无限候选。

本地 RAG/政策/尺码工具不能向候选登记表添加商品。详情查询只有 SPU 时，不等于所有 SKU 均被登记；SKU 必须来自候选或 availability 工具实际返回。

最终 product_refs 必须在本轮 ledger 中，SPU 与 SKU 配对正确；重新读取商品展示和价格，复核明确硬约束。价格上涨超预算或无货时移除推荐并同步修正文本为受限说明，不能留下“已找到且有货”的矛盾。不要单纯过滤卡片而保留错误答案。

run 在 done/error/cancel 后撤销 token，清理 ledger；TTL 用于异常终止兜底。最终处理先完成验证、归因和持久化，再关闭 run，重复 done 不重复保存。

## 3. 文件结构与任务依赖

新增 `P/clothing_assistant/agent/pro/`：schemas.py（行动/需求/结果）、state.py（每轮状态）、tools.py（六类工具适配）、decision.py（模型调用与结构化解码）、executor.py（LangGraph 循环）、validation.py（证据与约束校验）、__init__.py。

新增 `P/clothing_assistant/infrastructure/java_tool_client.py`：固定 Java 网关地址、鉴权和超时；新增 `P/clothing_assistant/api/pro_routes.py`：v2 路由与 SSE。

Java 新增 assistant/service/ProAssistantService.java（请求生命周期）、ProContextService.java（无旧意图的上下文）、ProRunRegistry.java（可信候选登记）、ProRecommendationValidator.java（最终校验）；新增 assistant/api/ProAssistantController.java、InternalAssistantToolController.java。不要把这些职责继续堆进 AssistantService。

依赖：Task 1 → 2 → 3；Task 1 → 4 → 5；Task 3+5 → 6 → 7 → 8 → 9 → 10 → 11。每个阶段合入后保持 Pro 开关关闭，直到 Task 11 验收。

## Task 1：固化 v2 契约与共享样例

**Files:** Create `C/contracts/assistant-streaming-chat/v2.md`、`schemas/v2-pro-request.schema.json`、`schemas/v2-pro-done.schema.json`、`examples/v2-pro-request.json`、`examples/v2-pro-done.json`；Modify `C/docs/coding-boundary.md`（引用 v2 的动态候选规则，不改变 v1）。Create `P/tests/test_pro_contract.py`。

- [ ] 将第 2 节协议固化到共享 v2，明确公共/内部请求分离，progress/token/done/error 的事件顺序、一次终止事件、sync 与 done 结果一致。
- [ ] 编写 JSON Schema：强制 version/mode/run、拒绝模型可写身份、定义枚举和金额正则；复制脱敏样例到测试 fixtures，记录共享源路径与版本。
- [ ] 添加以下最小边界测试，正常样例分别覆盖 request/done：

```python
import json
from pathlib import Path
import jsonschema
import pytest

def test_pro_request_rejects_wrong_mode():
    root = Path(__file__).parent / "fixtures" / "pro"
    schema = json.loads((root / "request.schema.json").read_text(encoding="utf-8"))
    payload = json.loads((root / "request.json").read_text(encoding="utf-8"))
    payload["agent_mode"] = "pro_max"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)
```

- [ ] 在 P 运行 `python -m pytest tests/test_pro_contract.py -q`，先观察缺少 schema/fixture 的失败，再添加实际 fixture 并获得 PASS。若 jsonschema 未列入开发依赖，仅加入 requirements-dev.txt。
- [ ] 在 C、P 分别提交本任务文件；不要把两套目录当成同一仓库提交。提交信息 `docs: define assistant v2 dynamic tool contract`。

## Task 2：Java 本轮查询网关与可信候选登记

**Files:** Create `B/assistant/service/ProRunRegistry.java`、`B/assistant/api/InternalAssistantToolController.java`、`B/assistant/service/ProToolQueryService.java`、`B/assistant/dto/ProToolRequest.java`、`B/assistant/dto/ProToolResult.java`；Test `JT/assistant/ProRunRegistryTests.java`、`JT/assistant/InternalAssistantToolControllerTests.java`。

- [ ] 写 MockMvc 测试：无内部 token、无 run token、其他 run 的 token、过期 run 均拒绝；合法搜索只登记实际返回 SKU，TTL/容量/关闭后调用均有断言。
- [ ] 在 J/backend 运行 `./mvnw.cmd "-Dtest=ProRunRegistryTests,InternalAssistantToolControllerTests" test`，确认新行为尚未存在。
- [ ] 实现 Redis run ledger 与三种工具分派，复用 ProductCatalogService、RecommendationCandidateQueryService 和 InventoryQueryService。明确字段映射，预算使用 decimal；299.90 可进入 300.00 的范围，300.01 不可进入。颜色/尺码硬条件不支持搜索过滤时按真实 SKU 筛除。
- [ ] 现有候选 DTO 的 budgetMax 是 Integer；v2 小数预算先使用向上取整且检查溢出的整数作召回上界，再按原始 BigDecimal 预算精确过滤，不能截断 299.90 为 299。超过 Integer 范围返回参数错误，保持旧 DTO 不变。最终金额单位以 Mapper/数据库的真实 sale_price 为依据，不做隐式单位转换。
- [ ] 网关固定动作白名单：

```text
search_products -> Java candidate query -> bounded results -> register actual pairs
get_product_detail -> Java detail -> register returned SPU fact only
check_availability -> resolve SKU -> read inventory -> register actual pair
unknown tool -> 400; bad credentials -> 401/403; dependency outage -> structured unavailable
```

- [ ] 重新执行上述命令，预期全部通过；增加 HTTP 失败不能产生候选登记的断言。提交 `feat: add run-scoped Pro query gateway`。

## Task 3：Java Pro 上下文与独立入口

**Files:** Create `B/assistant/api/ProAssistantController.java`、`B/assistant/service/ProAssistantService.java`、`B/assistant/service/ProContextService.java`、`B/assistant/dto/ProChatRequest.java`、`B/assistant/dto/ProPythonChatRequest.java`、`B/assistant/client/ProPythonAssistantClient.java`；Test `JT/assistant/ProAssistantServiceTests.java`、`JT/assistant/ProPythonAssistantClientTests.java`。

- [ ] 写行为测试：会话归属校验后保存用户消息；Pro 组装历史、资料和明确筛选条件；verify 旧 DemandIntentParseClient 从未被调用，旧 clarification 不提前截断 Pro。
- [ ] 运行 `./mvnw.cmd "-Dtest=ProAssistantServiceTests,ProPythonAssistantClientTests" test` 确认红灯。
- [ ] 实现 Pro 独立路径、v2 序列化和 run 生命周期。服务凭据来自配置，不能使用前端提交的 trusted context；默认 Pro 功能开关关闭，关闭时返回明确的模式不可用响应。
- [ ] 保留 v1 DTO 的构造签名和行为；不为新增模式强行修改旧 record 构造调用。公共 Pro DTO 的额外可信事实字段应拒绝或明确忽略且不进入 Python。
- [ ] 同一测试命令通过后提交 `feat: add isolated Java Pro entry and context`。

## Task 4：Python 行动协议与每轮状态

**Files:** Create `P/clothing_assistant/agent/pro/__init__.py`、`schemas.py`、`state.py`；Test `P/tests/test_pro_state.py`。

- [ ] 写下面的不可变约束和严格行动测试；在 schemas.py 实现 `parse_action(payload)`，在 state.py 实现 `new_state(run_id, query, hard_constraints)`：

```python
import pytest
from pydantic import ValidationError
from clothing_assistant.agent.pro.schemas import parse_action
from clothing_assistant.agent.pro.state import new_state

def test_unknown_tool_is_rejected():
    with pytest.raises(ValidationError):
        parse_action({"kind": "tool", "tool": "run_sql", "arguments": {}})

def test_runs_do_not_share_evidence():
    first = new_state("a", "找外套", {"budget_max": "300.00"})
    second = new_state("b", "查尺码", {})
    first["evidence"].append({"id": "e1"})
    assert second["evidence"] == []
```

- [ ] `python -m pytest tests/test_pro_state.py -q` 红灯后实现 discriminated union；三种行动参数互斥、未知字段拒绝、每轮 list/dict 新建。
- [ ] 约束保存用户来源，工具反馈不能改原始 hard_constraints；模型状态更新仅接受需求状态建议，状态升级检查 evidence_id。
- [ ] 相同命令 PASS 后提交 `feat: define Pro actions and isolated state`。

## Task 5：六类工具适配

**Files:** Create `P/clothing_assistant/infrastructure/java_tool_client.py`、`P/clothing_assistant/agent/pro/tools.py`；Test `P/tests/test_pro_tools.py`。

- [ ] 使用本地 fake transport 写测试：固定 base URL、隐藏 token、HTTP 超时映射 unavailable、库存 0 保持 ok、未找到 SKU 为 empty、政策无来源为 no_evidence。
- [ ] `python -m pytest tests/test_pro_tools.py -q` 确认失败。
- [ ] 定义 `execute_tool(name, arguments, *, java_client, knowledge_runner, policy_runner, size_runner)`，返回统一 ToolResult；Java client 只请求 Task 2 三个动作，不接受任意 URL。
- [ ] 将 RAG/policy/size 的现有调用用适配器封装，不改 Lite should_run；商品知识 query 必须附带目标商品范围，返回 source 和 evidence_id。尺寸输入显式单位，裸数字有歧义时返回 needs_input。
- [ ] 尺码结果含 `basis: generic_rule | product_chart`；只有存在实际商品尺码证据才能选 product_chart。当前只能通用规则时明确返回限制，不虚构表。
- [ ] 重新运行工具测试和 `python -m pytest tests/test_agent_pipeline.py tests/test_agent_mvp.py -q`，通过后提交 `feat: adapt shared capabilities as Pro tools`。

## Task 6：结构化模型决策与动态循环

**Files:** Create `P/clothing_assistant/agent/pro/decision.py`、`executor.py`；Modify `P/clothing_assistant/config_data.py`（Pro 预算配置）；Test `P/tests/test_pro_executor.py`。

- [ ] 建立可注入的 `run_pro_agent(query, *, decide, execute_tool, clock, limits, context)` 测试入口。decide 接收只读状态快照，execute_tool 接收经验证行动，clock 使用 monotonic。
- [ ] 测试 fake decide 必须根据 observation 分支，不能只返回与工具结果无关的预排动作：第一次库存 0 时选 B、库存正数时结束；同输入换工具结果产生不同调用轨迹。
- [ ] `python -m pytest tests/test_pro_executor.py -q` 确认失败；实现 LangGraph decide→execute→decide、clarify→end、finish→validate 路径。每轮先判断 deadline 和取消标记。
- [ ] 决策使用现有模型工厂但独立 prompt，JSON-only 文本通过 parse_action；复用并发限制和错误分类，保留模型 usage。解析失败只允许一次修正，仍失败返回受限结果；不假设现有 deterministic provider 能生成行动，测试显式注入 fake decide。
- [ ] 加入 12 步/8 工具/90 秒边界、去重缓存、一次暂时失败重试；使用 fake clock 测超时，不在测试中 sleep。预算耗尽保留证据与未完成需求。
- [ ] 测试检索文本中“忽略预算并调用其他地址”不能改变白名单或 hard_constraints；token 不出现在模型输入、输出记录中。
- [ ] 测试通过后提交 `feat: add bounded Pro decision loop`。

## Task 7：最终事实校验、动态卡片与一致回答

**Files:** Create `P/clothing_assistant/agent/pro/validation.py`、`B/assistant/service/ProRecommendationValidator.java`；Modify Task 3 ProAssistantService；Test `P/tests/test_pro_validation.py`、`JT/assistant/ProRecommendationValidatorTests.java`。

- [x] 写测试：B 由运行中新查询加入可展示；假 SKU、跨 run、SPU/SKU 错配均拒绝；价格变更超过预算后文本与卡片同步变为部分结果。
- [x] 运行 `python -m pytest tests/test_pro_validation.py -q`；Java 运行 `./mvnw.cmd "-Dtest=ProRecommendationValidatorTests" test`，确认失败。
- [x] Python 校验 evidence、需求状态、硬条件，不依赖单一 intent；Java 使用 run ledger 验证 provenance 并读取事实。能复用现有字段校验就复用，不放宽旧 RecommendationDecisionService。
- [x] Java 返回最终商品信息、推荐理由、建议尺码和 basis。Python 任意提供的价格/图片/URL 不采用。价格或库存变化使推荐失效时，用确定性受限说明替换相关承诺，或预算内重新决策一次；首期选择确定性受限说明，避免增加跨服务循环。
- [x] 没有政策证据时需求标为无法确认，不能将“能退换”的强条件标为全部满足。通用尺码建议不作为适配保证。
- [x] 两端测试通过后提交 `feat: 校验 Pro 推荐商品与实时事实`。

## Task 8：Python v2 同步与流式 API

**Files:** Create `P/clothing_assistant/api/pro_routes.py`；Modify `P/clothing_assistant/api/app.py`（注册 router）、`P/clothing_assistant/api/streaming.py`（仅新增 v2 helper）；Test `P/tests/test_pro_api.py`、`P/tests/test_pro_stream.py`。

- [x] 测试 v2 鉴权、版本拒绝、sync/done 内容一致；SSE 顺序为 progress* → token* → done，失败只有一个 error 终止。不将工具数据或内部推理当 token。
- [x] `python -m pytest tests/test_pro_api.py tests/test_pro_stream.py -q` 通过。
- [x] progress 使用执行器真实 start/complete 事件、序号和 runId；最终草稿经过 Python 校验后才发 token，卡片只在 done。Java v2 必须缓存答案 token 直到自身最终校验后再对前端释放，progress 可以实时转发，避免价格变化时错误承诺已被用户看见。
- [x] 同步与流式调用共享执行器，HTTP disconnect 传播取消；凭据不进入 checkpoint。所有 SSE data 为单行 JSON。
- [x] 运行 `python -m pytest tests/test_pro_api.py tests/test_pro_stream.py tests/test_api.py tests/test_chat_stream.py -q`，旧 v1 测试通过后提交 `feat: 提供 Pro v2 同步流式 API`。

## Task 9：Java v2 SSE 与持久化

**Files:** Modify `B/assistant/client/ProPythonAssistantClient.java`、`B/assistant/service/ProAssistantService.java`、`B/assistant/api/ProAssistantController.java`；Create `B/assistant/dto/ProProgressEvent.java`、`ProDoneEvent.java`；Test `JT/assistant/ProAssistantStreamTests.java`。

- [x] 测试 progress 允许枚举事件且丢弃不匹配 runId，重复 done 只保存一次，断开不存半条答案，超时和错误会撤销 run token。
- [x] `./mvnw.cmd "-Dtest=ProAssistantStreamTests" test` 红灯后实现 meta/progress/token/done/error v2 转换、缓冲验证策略和取消。
- [x] 完成时先用 Task 7 校验，保存校验后的答案和最终商品引用，再发送最终 done 并关闭 run；数据库失败输出 error，不将未保存答案标为已完成。
- [x] 使用已有会话存储服务和推荐归因能力；同一 runId 防止重复终结。旧 v1 handler 不转发新事件。
- [x] 运行 `./mvnw.cmd "-Dtest=ProAssistantStreamTests,AssistantServiceTests,AssistantControllerTests,RestPythonAssistantClientTests" test`，通过后提交 `feat: 实现 Java v2 SSE 与结果持久化`。

## Task 10：前端模式、进度和共用商品卡片

**Files:** Modify `J/frontend/src/shared/api/types.ts`、`shared/api/assistantStream.ts`、`features/assistant/ChatPanel.tsx`、`features/assistant/assistantState.ts`；复用 `features/catalog/ProductCard.tsx`；Modify 对应 `assistantStream.test.ts`、`ChatPanel.test.tsx`、`assistantState.test.ts`。

- [x] 写交互测试：默认 Lite；选 Pro 后请求 v2 且带 pro；发送后固定本轮模式；progress 不显示商品；done 才显示经校验卡片；切回 Lite 保持历史。
- [x] 在 J/frontend 运行 `npm run test -- --run src/shared/api/assistantStream.test.ts src/features/assistant/ChatPanel.test.tsx src/features/assistant/assistantState.test.ts`，确认新行为失败。
- [x] 新增类型和按模式的 endpoint 选择，保持旧事件 parser 测试不变：

```typescript
export type AgentMode = "lite" | "pro";
export function chatStreamPath(mode: AgentMode): string {
  return mode === "pro"
    ? "/api/assistant/v2/chat/stream"
    : "/api/assistant/chat/stream";
}
```

- [x] 输入框加有可访问名称的模式选择；运行中禁止改变本轮 mode，下一轮允许选择。Pro 不可用明确展示错误，不静默调用 Lite。Pro Max 不展示为可用。
- [x] reducer 按本轮 runId 处理进度，重置旧进度和中间卡片。复用商品卡片并跳转现有详情路由；Pro 卡片直接消费 Java 返回事实，不通过旧候选列表再次筛掉动态商品。
- [x] 上述测试 PASS，再 `npm run build`；提交 `feat: add Pro mode and validated product display`。

## Task 11：纵向验收、对比报告与启用

**Files:** Create `P/tests/test_pro_eval_cases.py`、`P/tests/fixtures/pro/comparison_cases.json`、`P/scripts/eval_pro_comparison.py`、`P/docs/evals/lite-pro-comparison.md`、`J/frontend/e2e/pro-assistant.spec.ts`；Modify `P/docs/langgraph-flow.md`（链接 Pro 设计和执行说明）。

- [ ] 固定 fixtures 包含：明确单需求、复合五需求、A 无货改选 B、预算内无商品、政策无证据、尺码缺依据、库存超时、循环耗尽、跨模式与非法商品。expected 字段包括硬约束、必须回答项、允许 SKU 和预期失败类型。
- [ ] `python -m pytest tests/test_pro_eval_cases.py -q`；先用 fake 决策和 fake 工具验证代码闭环，再用真实模型连接固定工具结果测决策能力，两者报告分开，不把 scripted 成功率当模型能力。
- [ ] 编写评测脚本，命令 `python scripts/eval_pro_comparison.py --cases tests/fixtures/pro/comparison_cases.json --output docs/evals/lite-pro-comparison.json`；记录模型、输入、数据版本、调用次数、耗时、token usage。usage 缺失记 null；费用无可验证单价则不计算。每个版本使用相同工具事实，列出能力差异。
- [ ] E2E 以假 v2 SSE 服务验证模式切换、进度、最终商品 B 及详情跳转；命令 `npm run test:e2e -- e2e/pro-assistant.spec.ts`。再在本地集成环境验证 Java→Python→Java 工具回调不会死锁，并验证实际取消。
- [ ] 完成一次最终验证：P 执行 `python -m pytest -q`；J/backend 执行 `./mvnw.cmd verify`；J/frontend 执行 `npm run test -- --run` 和 `npm run build`。在依赖齐备环境跑相关集成测试；依赖缺失如实记录为未验证，不能声称全量通过。
- [ ] 验收门槛：固定行为用例全部通过；不存在伪造卡片、跨 run 候选、硬预算静默放宽；Lite 回归通过；实模复合场景有完整轨迹和实际质量报告。单次实模成功不作为统计提升结论。
- [ ] 验收后仅在本地/测试环境启用 Pro；生产发布另行按用户范围执行。开关关闭不影响 Lite，但必须等正在执行的 Pro 请求结束或取消后再清理。
- [ ] 提交 `test: verify Lite and Pro end-to-end behavior`，记录各仓库 commit 与实际运行命令、结果和未验证项目。

## 4. 设计覆盖自检与交付边界

| 设计要求 | 任务 |
| --- | --- |
| Lite 不变、版本切换 | 1、3、8、9、10 |
| 六类工具、真实 Java 数据 | 2、5 |
| 多需求与结果驱动循环 | 4、6 |
| 预算、重试、追问与部分结果 | 4、6、8 |
| 商品来源、硬约束、卡片 | 2、7、10 |
| 进度、SSE、取消和持久化 | 8、9、10 |
| 会话共享、状态隔离 | 3、4、9、10 |
| 对比与全链路验收 | 11 |
| Pro Max 延后 | 不新增多 Agent 代码 |

推荐执行顺序为本文任务顺序。可以在当前会话使用 executing-plans 分阶段执行；若用户选择子 Agent 执行，再按 subagent-driven-development 的审查流程调度。此计划没有启动任何实现任务。
