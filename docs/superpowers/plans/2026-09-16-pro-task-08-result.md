# Task 8：Python v2 同步与流式 API

本任务在 Python 分支 `codex/pro-task-07` 完成，代码提交为 `67a4cc0`。

## 本次实现

- 新增 [pro_routes.py](<C:/Users/14417/.config/superpowers/worktrees/outfit-python/pro-task-01/clothing_assistant/api/pro_routes.py>)，提供内部 `/v2/chat` 同步接口和 `/v2/chat/stream` SSE 接口。请求边界严格限制为 `assistant-v2` 与 `agent_mode=pro`，并校验运行、线程、请求和工具令牌字段。
- 同步和流式接口共用 `run_pro_agent`、Task 7 `validate_pro_result` 及安全 done 组装流程。商品引用只出现在 done，token 只来自已校验的最终回答。
- 工具执行通过服务端白名单适配器发出 `progress` 的 started/completed 事件，事件带递增 sequence、runId 和固定文案；工具数据、凭据和模型内部推理不会进入 SSE。
- 流式 worker 轮询 HTTP 断开并传播停止信号，失败只发一个安全 error 终止事件；所有 SSE data 均为单行 JSON。
- 修改 [app.py](<C:/Users/14417/.config/superpowers/worktrees/outfit-python/pro-task-01/clothing_assistant/api/app.py>) 注册 v2 路由并纳入现有内部鉴权和请求体大小限制；[streaming.py](<C:/Users/14417/.config/superpowers/worktrees/outfit-python/pro-task-01/clothing_assistant/api/streaming.py>) 仅新增 v2 SSE helper，v1 行为保持不变。

## 验证记录

- `python -m pytest tests/test_pro_api.py tests/test_pro_stream.py tests/test_api.py tests/test_chat_stream.py -q`：`80 passed`。
- 加上 Task 7 Pro 测试：`227 passed`（含 23 个 subtests）。
- `ruff check clothing_assistant tests`：通过。
- `python -m compileall -q clothing_assistant`：通过。
- `git diff --check`：通过。

## 完成 Task 8 后还剩什么

| 阶段 | 必须完成的工作 | 用户能看到的变化 |
| --- | --- | --- |
| Task 9 | Java v2 SSE、最终事实复核、答案持久化、重复终结保护、断开取消和运行凭证撤销 | Java 能安全转发 Python 进度，并只保存和发送最终校验结果 |
| Task 10 | 前端 Lite/Pro 模式按钮、进度状态、按 runId 更新、done 后共用商品卡片和详情跳转 | 用户可以在聊天界面选择 Pro 并看到对应商品 |
| Task 11 | 固定场景评测、Lite/Pro 对比、Java→Python→Java 联调、E2E 和测试环境启用 | 有完整证据判断功能正确性和实际模型效果 |

Pro Max 多 Agent 仍不在本轮实现范围。当前 Python 接口已经就绪，但商品卡片真正出现在现有页面仍需 Task 9 的 Java 结果转换和 Task 10 的前端接入。
