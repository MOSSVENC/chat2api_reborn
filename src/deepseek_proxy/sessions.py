"""
会话管理 — 兼容两种模式: 复用 (REUSE) 和 新建 (NEW)

对应:
- ds-free-api: ds_core/accounts.rs (session 复用 + health_check)
- Chat2API: src/main/proxy/adapters/deepseek.ts (每次 createSession)
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Callable, Awaitable

from .config import ProxyConfig, SessionMode
from .client import DsClient, CompletionPayload, EditMessagePayload, ChallengeData
from .pow_solver import PowSolver, PowResult


class SessionStrategy:
    """会话策略抽象"""

    def __init__(self, config: ProxyConfig, client: DsClient, solver: PowSolver,
                 token: str):
        self.config = config
        self.client = client
        self.solver = solver
        self.token = token

    async def prepare_pow(self, target_path: str = "/api/v0/chat/completion") -> str:
        """获取 PoW 挑战并求解, 返回 X-Ds-Pow-Response header 值"""
        challenge = await self.client.create_pow_challenge(self.token, target_path)
        result = self.solver.solve(challenge)
        return result.to_header()

    async def execute(self, prompt: str, thinking_enabled: bool = False,
                      search_enabled: bool = False, model_type: str = "default"):
        """
        执行对话请求。

        Returns:
            流式 httpx.Response（字节流）和可选的清理回调
        """
        raise NotImplementedError

    async def cleanup(self) -> None:
        """清理资源 (会话删除等)"""
        pass


class ReuseSessionStrategy(SessionStrategy):
    """ds-free-api 模式: 初始化时创建 session + health_check, 永久复用

    会话绑定生命周期, 通过 edit_message 编辑 message 1。
    """

    def __init__(self, config: ProxyConfig, client: DsClient, solver: PowSolver,
                 token: str, model_types: list[str]):
        super().__init__(config, client, solver, token)
        self._sessions: dict[str, str] = {}  # model_type → session_id

    async def init(self, model_types: list[str]) -> None:
        """初始化所有 model_type 的 session"""
        for mt in model_types:
            await self._init_model_type(mt)

    async def _init_model_type(self, model_type: str) -> None:
        # 1. 创建 session
        session_id = await self.client.create_session(self.token)

        # 2. health_check: 发送一条测试消息 (message 0)
        pow_header = await self.prepare_pow("/api/v0/chat/edit_message")
        payload = CompletionPayload(
            chat_session_id=session_id,
            prompt="只回复`Hello, world!`",
            thinking_enabled=False,
            search_enabled=False,
        )
        # 用 completion 做 health check, 让它写入 message 0
        resp = await self.client.completion(self.token, pow_header, payload)
        # 消费流确保消息写入
        async for _ in resp.aiter_bytes():
            pass

        self._sessions[model_type] = session_id

    async def execute(self, prompt: str, thinking_enabled: bool = False,
                      search_enabled: bool = False, model_type: str = "default"):
        session_id = self._sessions.get(model_type)
        if not session_id:
            raise RuntimeError(f"No session for model_type={model_type}. Call init() first.")

        pow_header = await self.prepare_pow("/api/v0/chat/edit_message")
        payload = EditMessagePayload(
            chat_session_id=session_id,
            message_id=1,  # message 0 是 health_check
            prompt=prompt,
            search_enabled=search_enabled,
            thinking_enabled=thinking_enabled,
            model_type=model_type,
        )

        resp = await self.client.edit_message(self.token, pow_header, payload)
        return resp  # 调用方负责消费流

    async def cleanup(self) -> None:
        for session_id in self._sessions.values():
            try:
                await self.client.delete_session(self.token, session_id)
            except Exception:
                pass
        self._sessions.clear()


class NewSessionStrategy(SessionStrategy):
    """Chat2API 模式: 每次请求创建新 session, 用 completion 端点"""

    def __init__(self, config: ProxyConfig, client: DsClient, solver: PowSolver,
                 token: str):
        super().__init__(config, client, solver, token)

    async def execute(self, prompt: str, thinking_enabled: bool = False,
                      search_enabled: bool = False, model_type: str = "default"):
        # 1. 创建 session
        session_id = await self.client.create_session(self.token)

        # 2. PoW
        pow_header = await self.prepare_pow("/api/v0/chat/completion")

        # 3. Completion
        payload = CompletionPayload(
            chat_session_id=session_id,
            prompt=prompt,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
        )

        resp = await self.client.completion(self.token, pow_header, payload)

        # 绑定 session_id 到响应对象, 调用方消费完流后清理
        resp._session_id = session_id  # type: ignore
        return resp

    async def cleanup(self) -> None:
        pass  # NEW 模式下由调用方按 session 清理


async def cleanup_session(client: DsClient, token: str, session_id: str) -> None:
    """清理单个 session"""
    try:
        await client.delete_session(token, session_id)
    except Exception:
        pass


def create_session_strategy(config: ProxyConfig, client: DsClient,
                            solver: PowSolver, token: str,
                            model_types: list[str]) -> SessionStrategy:
    """工厂函数"""
    if config.session_mode == SessionMode.REUSE:
        return ReuseSessionStrategy(config, client, solver, token, model_types)
    elif config.session_mode == SessionMode.NEW:
        return NewSessionStrategy(config, client, solver, token)
    else:
        raise ValueError(f"Unsupported session_mode: {config.session_mode}")