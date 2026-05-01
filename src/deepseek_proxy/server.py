"""
FastAPI HTTP 服务器 — OpenAI 兼容端点

端点:
- GET  /v1/models
- POST /v1/chat/completions
- GET  /health
"""

from __future__ import annotations

import json
import asyncio
from typing import Optional, Any

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import ProxyConfig
from .client import DsClient
from .auth import create_auth_provider, AuthProvider
from .pow_solver import create_solver, PowSolver
from .sessions import (
    create_session_strategy, SessionStrategy,
    ReuseSessionStrategy, cleanup_session,
)
from .openai_adapter import OpenAIAdapter
from .sse_parser import full_sse_pipeline


# ═══════════════════════════════════════════════════════════
# 全局状态 (应用生命周期内)
# ═══════════════════════════════════════════════════════════

class AppState:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.client: Optional[DsClient] = None
        self.solver: Optional[PowSolver] = None
        self.auth: Optional[AuthProvider] = None
        self.token: Optional[str] = None
        self.session_strategy: Optional[SessionStrategy] = None
        self.adapter: Optional[OpenAIAdapter] = None

    async def init(self):
        """初始化所有组件"""
        import logging
        logger = logging.getLogger("deepseek_proxy")

        # 1. HTTP 客户端
        logger.info("初始化 HTTP 客户端...")
        self.client = DsClient(self.config)

        # 2. 认证
        logger.info("初始化认证 (mode=%s)...", self.config.auth_mode.value)
        self.auth = create_auth_provider(self.config, self.client)
        self.token = await self.auth.authenticate()
        logger.info("认证成功, token: %s...", self.token[:20])

        # 3. PoW 求解器
        logger.info("下载 WASM 并初始化 PoW 求解器...")
        self.solver = await create_solver(self.config.wasm_url)
        logger.info("PoW 求解器初始化完成")

        # 4. 会话策略
        logger.info("初始化会话策略 (mode=%s)...", self.config.session_mode.value)
        self.session_strategy = create_session_strategy(
            self.config, self.client, self.solver, self.token,
            self.config.model_types,
        )
        if isinstance(self.session_strategy, ReuseSessionStrategy):
            await self.session_strategy.init(self.config.model_types)
            logger.info("会话池初始化完成")

        # 5. 适配器
        self.adapter = OpenAIAdapter(self.config, self.session_strategy)

        logger.info("DeepSeek Proxy 初始化完成")

    async def shutdown(self):
        """清理资源"""
        if self.session_strategy:
            await self.session_strategy.cleanup()
        if self.client:
            await self.client.close()


_app_state: Optional[AppState] = None


def get_state() -> AppState:
    if _app_state is None:
        raise RuntimeError("AppState not initialized")
    return _app_state


# ═══════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════

def create_app(config: ProxyConfig) -> FastAPI:
    global _app_state

    app = FastAPI(
        title="DeepSeek Proxy",
        description="OpenAI-compatible API proxy for DeepSeek chat",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key 验证中间件
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # 跳过 health endpoint
        if request.url.path == "/health":
            return await call_next(request)

        # 如果有配置 api_tokens 则验证
        state = _app_state
        if state and state.config.api_tokens:
            auth_header = request.headers.get("Authorization", "")
            token = auth_header.replace("Bearer ", "").strip()

            if not token or token not in state.config.api_tokens:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid API key", "type": "auth_error"}},
                )

        return await call_next(request)

    # ── 端点 ─────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        """返回可用模型列表"""
        models = [
            {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"},
        ]
        # 根据配置添加搜索/思考变体
        for mt in _app_state.config.model_types if _app_state else ["default", "expert"]:
            if mt == "expert":
                models.append({"id": "deepseek-r1", "object": "model", "owned_by": "deepseek"})
            if mt == "default":
                models.append({"id": "deepseek-v3", "object": "model", "owned_by": "deepseek"})

        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI 兼容的聊天补全端点"""
        state = get_state()

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        messages = body.get("messages", [])
        model = body.get("model", state.config.default_model.value)
        stream = body.get("stream", True)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        response_format = body.get("response_format")
        web_search = body.get("web_search")
        reasoning_effort = body.get("reasoning_effort")

        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        try:
            resp = await state.adapter.chat(
                messages=messages,
                model=model,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                web_search=web_search,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            import logging
            logging.getLogger("deepseek_proxy").error("Chat failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        if not stream:
            # 非流式: 累积内容后返回
            full_content = ""
            async for chunk in full_sse_pipeline(
                resp.aiter_bytes(),
                model=model,
            ):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    full_content += delta["content"]
                elif delta.get("reasoning_content"):
                    pass  # 非流式暂不处理 reasoning

            return {
                "id": f"chatcmpl-{id(full_content)}",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        # 流式: SSE 响应
        async def event_stream():
            try:
                async for chunk in full_sse_pipeline(
                    resp.aiter_bytes(),
                    model=model,
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                import traceback
                traceback.print_exc()
                error_chunk = {
                    "error": {"message": "Stream error", "type": "server_error"}
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # 清理 session (NEW 模式)
                session_id = getattr(resp, "_session_id", None)
                if session_id:
                    state = get_state()
                    await cleanup_session(state.client, state.token, session_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# ═══════════════════════════════════════════════════════════
# 初始化入口 (在 main.py 中调用)
# ═══════════════════════════════════════════════════════════

async def init_server(config: ProxyConfig):
    global _app_state
    _app_state = AppState(config)
    await _app_state.init()
    return _app_state


async def shutdown_server():
    if _app_state:
        await _app_state.shutdown()