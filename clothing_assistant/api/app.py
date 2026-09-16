import asyncio
import hmac
import logging
import re
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from clothing_assistant.api.schemas import (
    DemandIntentParseRequest,
    FeedbackRequest,
    LegacyChatRequest,
    PythonChatRequest,
    PythonChatResponse,
    RagRebuildRequest,
    RagRebuildResponse,
)
from clothing_assistant.application.demand_intent_parse_service import (
    DemandIntentParseError,
    DemandIntentParseService,
)
from clothing_assistant.domain.demand_intent_models import DemandIntentParseCandidate
from clothing_assistant.api.streaming import (
    build_error_event,
    build_stream_done_payload,
    format_sse_event,
)
from clothing_assistant.api.pro_routes import router as pro_router
from clothing_assistant.agent.agent_executor import run_agent
from clothing_assistant.agent.langgraph_executor import (
    close_runtime_checkpointer,
    initialize_runtime_checkpointer,
    run_langgraph_agent,
    stream_langgraph_agent,
)
from clothing_assistant.config_data import (
    PROJECT_API_TITLE,
    get_internal_api_token,
    get_llm_max_concurrency,
    get_llm_max_retries,
    get_llm_timeout_seconds,
    get_max_chat_request_bytes,
    get_rag_timeout_seconds,
    get_runtime_environment,
    get_stream_safety_tail_chars,
    is_debug_response_enabled,
    is_deterministic_provider_enabled,
    is_internal_auth_required,
)
from clothing_assistant.infrastructure.vector_store import (
    get_vector_store_status,
    rebuild_vector_store_from_local_knowledge,
)

logger = logging.getLogger(__name__)
INTERNAL_ERROR_MESSAGE = "AI service failed to process the request."
INTERNAL_TOKEN_HEADER = "X-Internal-Token"
INTERNAL_AUTH_ERROR = {
    "error": "internal_auth_required",
    "message": "python assistant internal authentication failed",
}
CHAT_REQUEST_PATHS = frozenset(
    {
        "/chat",
        "/chat/stream",
        "/chat/pipeline",
        "/chat/langgraph",
        "/v2/chat",
        "/v2/chat/stream",
    }
)
REQUEST_TOO_LARGE_ERROR = {
    "error": "request_too_large",
    "message": "python assistant request exceeds the configured size limit",
}
SAFE_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_runtime_checkpointer()
    try:
        get_llm_timeout_seconds()
        get_llm_max_retries()
        get_llm_max_concurrency()
        get_rag_timeout_seconds()
        get_stream_safety_tail_chars()
        is_deterministic_provider_enabled()
        if get_runtime_environment() == "production" and not get_internal_api_token():
            raise RuntimeError("APP_INTERNAL_API_TOKEN is required in production")
        yield
    finally:
        close_runtime_checkpointer()


app = FastAPI(
    title=PROJECT_API_TITLE,
    description="服装尺寸与商品问答助手的 API 入口。",
    version="0.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def reject_oversized_declared_chat_requests(request: Request, call_next):
    if request.url.path in CHAT_REQUEST_PATHS:
        raw_content_length = request.headers.get("Content-Length")
        try:
            content_length = int(raw_content_length) if raw_content_length is not None else None
        except ValueError:
            content_length = None

        if content_length is not None and content_length >= 0:
            if content_length > get_max_chat_request_bytes():
                return JSONResponse(status_code=413, content=REQUEST_TOO_LARGE_ERROR)

    return await call_next(request)


async def extract_safe_request_id(request: Request) -> str | None:
    try:
        body = await request.json()
    except Exception:
        return None

    if not isinstance(body, dict):
        return None

    request_id = body.get("request_id")
    if not isinstance(request_id, str):
        return None

    return request_id if SAFE_REQUEST_ID_PATTERN.fullmatch(request_id) else None


async def require_internal_auth(request: Request):
    if not is_internal_auth_required():
        return

    expected = get_internal_api_token()
    supplied = request.headers.get(INTERNAL_TOKEN_HEADER, "")
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail=INTERNAL_AUTH_ERROR)


# v2 is an internal Java-to-Python boundary.  Keep the dependency here so the
# existing authentication implementation remains the single source of truth,
# while pro_routes stays importable without an app-module cycle.
app.include_router(pro_router, dependencies=[Depends(require_internal_auth)])


def get_demand_intent_parse_service() -> DemandIntentParseService:
    return DemandIntentParseService()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器：捕获所有未被处理的系统内部异常，防止服务直接崩溃。

    统一返回 500 错误格式，并在日志中记录详细错误栈。
    """
    logger.exception("处理请求时发生未捕获的异常 %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "request_id": await extract_safe_request_id(request),
            "message": INTERNAL_ERROR_MESSAGE,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    if exc.status_code == 401 and exc.detail == INTERNAL_AUTH_ERROR:
        return JSONResponse(status_code=401, content=INTERNAL_AUTH_ERROR)

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


from fastapi.exceptions import RequestValidationError


async def extract_safe_request_body(request: Request):
    request_id = await extract_safe_request_id(request)
    if request_id is None:
        return None

    return {"request_id": request_id}


def sanitize_validation_errors(errors):
    sanitized = []
    for error in errors:
        sanitized.append(
            {
                key: value
                for key, value in error.items()
                if key not in {"input", "ctx", "url"}
            }
        )

    return sanitized


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数校验异常处理器：当客户端(Java端)发送的请求数据格式不符合 Pydantic 模型（如缺少必填字段）时触发。

    返回 422 错误，并详细提示哪个字段校验失败。
    """
    safe_body = await extract_safe_request_body(request)
    safe_errors = sanitize_validation_errors(exc.errors())
    logger.warning(
        "422 validation error request_id=%s method=%s path=%s errors=%s",
        (safe_body or {}).get("request_id"),
        request.method,
        request.url.path,
        safe_errors,
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": safe_errors, "body": safe_body}),
    )


def build_legacy_chat_response(agent_result, include_debug):
    """构建旧版本聊天响应：根据是否需要调试信息，返回简化版的问答结果。"""
    include_debug = include_debug and is_debug_response_enabled()
    if include_debug:
        return agent_result

    return {"answer": agent_result["answer"]}


def get_agent_intent(agent_result):
    """提取意图：从 AI 代理的执行结果中提取用户的聊天意图（如：商品查询、尺寸建议等）。"""
    debug = agent_result.get("debug", {})
    return debug.get("intent_result", {}).get("intent") or "unknown"


def build_suggested_actions(agent_result):
    """构建建议动作：如果 AI 发现缺少必要信息（如未提供身高体重），

    则向前端/Java后端下发“请求追加信息 (ask_follow_up)”的动作指令。
    """
    debug = agent_result.get("debug", {})

    if debug.get("stop_reason") == "missing_info":
        return [{"type": "ask_follow_up"}]

    return []


def build_contract_chat_response(agent_result, request_id, include_debug):
    """构建符合最新契约的聊天响应：将 AI 执行的结果包装成 Java 侧所期望的标准 JSON 格式。

    包含了答案、意图、推荐商品列表以及建议动作。
    """
    include_debug = include_debug and is_debug_response_enabled()
    response = PythonChatResponse(
        request_id=request_id,
        answer=agent_result["answer"],
        intent=get_agent_intent(agent_result),
        product_refs=agent_result.get("product_refs", []),
        rejected_reasons=agent_result.get("rejected_reasons", {}),
        suggested_actions=build_suggested_actions(agent_result),
        debug=agent_result.get("debug") if include_debug else None,
    )
    return response.model_dump(exclude_none=True)


@app.get("/health")
def health():
    """健康检查接口：用于 Kubernetes 等容器服务或负载均衡器检查当前 Python 服务是否正常存活。"""
    return {"status": "ok"}


@app.post(
    "/internal/demand-intent/parse",
    response_model=DemandIntentParseCandidate,
    response_model_by_alias=True,
)
def parse_demand_intent(
    parse_request: DemandIntentParseRequest,
    _: None = Depends(require_internal_auth),
    service: DemandIntentParseService = Depends(get_demand_intent_parse_service),
):
    """Return an untrusted but strictly shaped LLM candidate for Java validation."""
    try:
        return service.parse(parse_request.service_dict())
    except DemandIntentParseError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "demand_intent_parse_unavailable",
                "message": "semantic demand parsing is temporarily unavailable",
            },
        ) from error


@app.get("/health/rag")
def rag_health():
    """Return RAG index readiness separately from service liveness."""
    return get_vector_store_status()


@app.post("/internal/rag/rebuild", response_model=RagRebuildResponse)
async def rebuild_rag_index(
    rebuild_request: RagRebuildRequest,
    _: None = Depends(require_internal_auth),
):
    """Rebuild the local global-knowledge index without blocking the event loop."""
    result = await asyncio.to_thread(
        rebuild_vector_store_from_local_knowledge,
        task_id=rebuild_request.task_id,
    )
    return RagRebuildResponse.model_validate(result)


@app.post("/chat")
async def chat(
    chat_request: PythonChatRequest,
    request: Request,
    _: None = Depends(require_internal_auth),
):
    """主聊天接口（阻塞式）：接收用户的提问，调用 LangGraph 主工作流进行处理。

    这个接口会等待模型完全生成结果后，一次性返回完整的对话 JSON 响应。
    """
    try:
        result = run_langgraph_agent(
            chat_request.query.strip(),
            chat_history=chat_request.chat_history_dicts(),
            thread_id=chat_request.thread_id or chat_request.session_id,
            request_id=chat_request.request_id,
            session_id=chat_request.session_id,
            user_context=chat_request.user_context_dict(),
            candidates=chat_request.candidate_dicts(),
            demand_intent=(
                chat_request.demand_intent.model_dump(exclude_none=True, exclude_unset=True)
                if chat_request.demand_intent
                else None
            ),
        )
    except Exception:
        logger.exception("POST /chat 接口发生未捕获异常")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "request_id": await extract_safe_request_id(request),
                "message": INTERNAL_ERROR_MESSAGE,
            },
        )

    return build_contract_chat_response(result, chat_request.request_id, chat_request.debug)


_STREAM_END = object()
STREAM_ERROR_MESSAGES = {
    "timeout": ("dependency_timeout", "AI dependency timed out."),
    "rate_limited": ("dependency_unavailable", "AI dependency is temporarily unavailable."),
    "connection_error": ("dependency_unavailable", "AI dependency is temporarily unavailable."),
    "upstream_5xx": ("dependency_unavailable", "AI dependency is temporarily unavailable."),
    "upstream_4xx": ("dependency_rejected", "AI dependency rejected the request."),
    "invalid_response": ("dependency_invalid_response", "AI dependency returned an invalid response."),
    "validation_failed": ("unsafe_model_output", "AI output failed safety validation."),
    "internal_error": ("internal_error", INTERNAL_ERROR_MESSAGE),
}


def _next_stream_event(events):
    try:
        return next(events)
    except StopIteration:
        return _STREAM_END


async def generate_chat_stream(chat_request: PythonChatRequest, http_request: Request):
    """Translate internal graph events to v1 SSE with disconnect cancellation."""
    events = stream_langgraph_agent(
        chat_request.query.strip(),
        chat_history=chat_request.chat_history_dicts(),
        thread_id=chat_request.thread_id or chat_request.session_id,
        request_id=chat_request.request_id,
        session_id=chat_request.session_id,
        user_context=chat_request.user_context_dict(),
        candidates=chat_request.candidate_dicts(),
        demand_intent=(
            chat_request.demand_intent.model_dump(exclude_none=True, exclude_unset=True)
            if chat_request.demand_intent
            else None
        ),
    )
    try:
        while not await http_request.is_disconnected():
            event = await asyncio.to_thread(_next_stream_event, events)
            if event is _STREAM_END:
                return
            if event.kind == "heartbeat":
                continue
            if event.kind == "token":
                yield format_sse_event("token", {"content": event.content})
                continue
            if event.kind == "result":
                yield format_sse_event(
                    "done",
                    build_stream_done_payload(event.result or {}, chat_request.request_id),
                )
                return
            if event.kind == "error":
                code, message = STREAM_ERROR_MESSAGES.get(
                    event.code,
                    STREAM_ERROR_MESSAGES["internal_error"],
                )
                yield build_error_event(code, message)
                return
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            await asyncio.to_thread(close)


@app.post("/chat/stream")
async def chat_stream(
    chat_request: PythonChatRequest,
    request: Request,
    _: None = Depends(require_internal_auth),
):
    """流式聊天接口：前端和 Java 端用于获取“打字机效果”回复的主要接口。

    返回一个持续推送文本的 StreamingResponse，实现一边思考一边输出的功能。
    """
    return StreamingResponse(
        generate_chat_stream(chat_request, request),
        media_type="text/event-stream;charset=utf-8",
        headers={
            "Cache-Control":"no-cache",
            "Connection":"keep-alive",
        },
    )


@app.post("/chat/pipeline")
def chat_pipeline(request: LegacyChatRequest, _: None = Depends(require_internal_auth)):
    """旧版聊天接口：保留了旧的手写 pipeline 逻辑，主要用于开发和测试阶段，

    方便开发者比对与验证新旧工作流（LangGraph与Pipeline）的差异。
    """
    result = run_agent(request.query.strip(), chat_history=request.chat_history)
    return build_legacy_chat_response(result, request.debug)


@app.post("/chat/langgraph")
def chat_langgraph(request: LegacyChatRequest, _: None = Depends(require_internal_auth)):
    result = run_langgraph_agent(
        request.query.strip(),
        chat_history=request.chat_history,
        thread_id=request.thread_id,
    )
    return build_legacy_chat_response(result, request.debug)


@app.post("/chat/feedback")
def receive_feedback(request: FeedbackRequest):
    """反馈收集接口：接收 Java 层转发过来的用户“点赞/踩”反馈。

    通过记录日志落盘，用于后期对 AI 推荐的准确度进行评估和模型微调。
    """
    # 根据契约：第一步先做简单落盘，把收到的点赞/踩信息打印到日志里
    logger.info(f"[Feedback Log] User {request.userId} rated {request.feedbackType} for message {request.messageId}")

    # 返回契约规定的 JSON 格式
    return {"status": "success", "message": "Feedback recorded"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
