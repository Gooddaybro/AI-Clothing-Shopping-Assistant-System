# Pro Task 1：v2 契约与边界测试

用户要求按最小 task 依次交付。本次仅完成 Task 1，不启用 Pro，不改变 Lite，不包含 Python 工具、Java 网关或前端模式按钮。

## 查看入口

- [v2 协议](../../../contracts/assistant-streaming-chat/v2.md)：Java/Python/前端边界、工具请求、事件和最终结果。
- [请求示例](../../../contracts/assistant-streaming-chat/examples/v2-pro-request.json)：Java 交给 Python 的数据。
- [结果示例](../../../contracts/assistant-streaming-chat/examples/v2-pro-done.json)：Python 返回 Java 的数据。
- [可执行测试](../../../tests/test_pro_contract.py)：正常数据可通过，非法模式、金额、状态和事实字段被拒绝。

JSON Schema 是协议验证基础，不代表运行接口已接入验证；鉴权、候选真实来源、金额单位和事实校验仍需后续任务实现。`user_context` 仅属于 Java 内部可信请求，其合法性必须在 Java 验证。

## 验证记录

先运行测试：缺少契约文件时 6 项失败；补齐初版后 6 项通过。再新增金额、状态等用例，观察 7 项预期失败，收紧 schema 后最终 14 项全部通过。

在本项目虚拟环境执行：

```powershell
python -m pytest tests/test_pro_contract.py -q
```

预期输出：`14 passed`。`requirements-dev.txt` 已声明 jsonschema 开发依赖。

运行代码未修改，因此未把后续尚未实现的 Pro 执行器测试加入本分支，也未声称完整 Pro 或全量集成测试通过。

## 隔离与下一步

本次分支：`codex/pro-task-01`。此前在 `codex/pro-dynamic` 工作区产生的后续草稿保留，不属于这次交付；原始检出的用户文件没有被覆盖。

共享契约权威源是 `D:/git/outfit-project-contract`，该目录目前没有 Git 仓库，因此通过项目内版本化副本保存本次可审阅提交。

下一项是 Task 2：Java 本轮查询网关与可信候选登记。它不属于本次完成范围，需按用户下一步指令继续。
