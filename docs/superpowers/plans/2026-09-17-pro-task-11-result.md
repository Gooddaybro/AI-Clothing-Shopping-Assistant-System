# Task 11：Lite / Pro 固定场景评测与纵向验收

本任务为 Python Pro v2 增加固定场景评测、安全门槛报告和测试环境说明，并与 Java 前端的假 SSE/可选真实集成验收对齐。

## 已交付

- `tests/fixtures/pro/comparison_cases.json` 固定 10 个场景：单需求、复合五需求、A 无货改选 B、预算内无商品、政策无证据、尺码无商品表、库存超时、循环耗尽、跨模式非法商品和取消。
- `scripts/eval_pro_comparison.py` 支持 `fake`、`real`、`both` 三种模式。fake 决策和 Java 工具事实完全脚本化；real 模式只替换决策来源，仍回放同一工具事实，结果单独保存。
- `docs/evals/lite-pro-comparison.json` 记录输入、数据版本、模型来源、能力差异、每轮调用数、耗时和 nullable token usage。Lite 行标记为 `reference_only`，不冒充模型成功率。
- `frontend/e2e/pro-assistant.spec.ts` 覆盖 Lite → Pro 切换、Pro 进度事件、B 商品最终卡片和详情跳转；`frontend/e2e/integration/pro-assistant.spec.ts` 提供显式环境开关控制的 Java → Python → Java 取消验收。
- Docker demo 增加 Python 回调 Java 的内部地址/token、Pro 预算和本地/测试开关；默认仍关闭 Pro。

## 验证命令与结果

| 命令 | 结果 |
| --- | --- |
| `uv run --with-requirements requirements.txt --with-requirements requirements-dev.txt pytest tests/test_pro_eval_cases.py -q` | 3 passed |
| `uv run --with-requirements requirements.txt --with-requirements requirements-dev.txt python scripts/eval_pro_comparison.py --cases tests/fixtures/pro/comparison_cases.json --output docs/evals/lite-pro-comparison.json` | fake Pro 10/10 passed |
| `uv run --with-requirements requirements.txt --with-requirements requirements-dev.txt ruff check scripts/eval_pro_comparison.py tests/test_pro_eval_cases.py` | passed |
| `uv run --with-requirements requirements.txt --with-requirements requirements-dev.txt python -m compileall -q clothing_assistant scripts tests` | passed |
| `npm run test -- --run` | 30 files / 87 tests passed |
| `npm run build` | passed |
| `npm run test:e2e -- e2e/pro-assistant.spec.ts` | 1 passed |
| `npm run test:e2e:integration -- e2e/integration/pro-assistant.spec.ts` | 1 skipped（未设置 `RUN_PRO_INTEGRATION=true`） |

Python 全量 `pytest -q` 为 454 passed、1 failed。唯一失败是既有 Lite 测试 `RecommendationServiceTests.test_same_thread_does_not_reuse_previous_size_state` 的 `embeddings.queries` 断言；在 Task 4/5/6 记录的基线中同样复现，原因是干净 checkout 没有被 Git 跟踪的派生 `clothing_assistant/chroma_db/` 向量索引。尺码状态断言本身通过，本任务没有改动该路径，因此不把它报告为 Task11 回归，也不声称 Python 全量绿色。

Java backend `./mvnw.cmd verify` 与真实 Docker 集成需要在依赖齐备环境继续执行。本机 Docker daemon 当前不可用（`dockerDesktopLinuxEngine` named pipe 不存在），所以 Java → Python → Java 正向链路、真实取消和真实模型决策均记录为未验证；可选 Playwright 集成测试只有在显式设置 `RUN_PRO_INTEGRATION=true` 后才会访问真实服务。

真实模型没有在本轮执行；报告中的 `real_model_executed=false` 是事实，不把 fake 成功率解释为模型能力或统计提升。
