# Lite / Pro 固定场景评测

Task11 用同一组脱敏工具事实比较 Lite 与 Pro。Lite 是现有固定流程，评测报告把它记录为 `reference_only`，不把固定流程的一次输出冒充模型成功率；Pro 分为两种来源：

- `scripted_fake`：固定决策和固定工具返回，用来验证约束、事实校验、取消和预算边界的安全门槛。
- `configured_model`：读取当前模型配置，由模型决定下一步；仍使用同一份 fixture 工具事实，结果单独记录。

固定用例覆盖明确单需求、复合五需求、A 无货改选 B、预算内无商品、政策无证据、尺码缺依据、库存超时、循环耗尽、跨模式非法商品和取消请求。每个用例声明硬约束、必须回答项、允许的 SPU/SKU 以及失败类型。

## 运行

在 Python 仓库根目录执行：

```text
uv run --with-requirements requirements.txt --with-requirements requirements-dev.txt \
  pytest tests/test_pro_eval_cases.py -q

uv run --with-requirements requirements.txt --with-requirements requirements-dev.txt \
  python scripts/eval_pro_comparison.py \
  --cases tests/fixtures/pro/comparison_cases.json \
  --output docs/evals/lite-pro-comparison.json
```

默认 `--mode fake` 只执行安全门槛，不需要模型密钥。需要真实模型时，在测试环境配置模型凭据后显式执行：

```text
uv run --with-requirements requirements.txt --with-requirements requirements-dev.txt \
  python scripts/eval_pro_comparison.py --mode both \
  --cases tests/fixtures/pro/comparison_cases.json \
  --output docs/evals/lite-pro-comparison.json
```

JSON 报告记录生成时间、模型来源和名称、输入用例、工具事实版本、每轮决策数、工具调用数、耗时和 token usage。provider 没有返回 usage 时写 `null`；没有可验证的单价时不计算费用。Lite 与 Pro 的 `source`、能力开关和结果状态分开保存，禁止把 scripted 成功率解释为真实模型能力。

## 当前安全门槛结果

本轮默认 fake 评测应满足 `pro_fake_passed == case_count`，并且每个 Pro 用例的候选商品都来自本轮 Java 事实。预算、跨 run、非法 SKU、库存变化和取消都必须先经过执行器/最终校验，不能通过文字或前端 payload 绕过。

真实模型评测需要在具备模型服务和固定工具回放环境的测试机执行。单次真实成功只能作为联调证据，不能作为统计提升结论；正式比较应保存多轮报告和对应数据版本。

## 环境边界

Pro 开关默认关闭。Docker demo 的本地/测试环境可以在 `.env` 中设置 `APP_AI_PRO_ENABLED=true`，并让 Python 使用 `JAVA_ASSISTANT_BASE_URL=http://backend-web:8080` 与同一内部 token。生产环境保持关闭，发布和用户范围另行审批。

前端假 SSE 验收命令：

```text
cd frontend
npm run test:e2e -- e2e/pro-assistant.spec.ts
```

真实 Java → Python → Java 还需要启动 Docker demo 的 MySQL、Redis、RabbitMQ、Elasticsearch、Postgres、Java、Python 和索引初始化服务；未启动或依赖缺失时，只能记录为未验证，不能声称端到端通过。
