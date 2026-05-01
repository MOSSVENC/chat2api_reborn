"""
OpenAI 兼容请求适配 — 接收 OpenAI 格式请求, 转换为 DeepSeek 调用

对应 ds-free-api: openai_adapter/ + openai_adapter/request/
"""

from __future__ import annotations

from typing import Optional, Any

from .config import ProxyConfig, DeepSeekModel
from .client import DsClient
from .prompt import build_chatml_prompt
from .sessions import SessionStrategy


class OpenAIAdapter:
    """将 OpenAI /v1/chat/completions 请求适配到 DeepSeek 网页 API"""

    def __init__(self, config: ProxyConfig, session: SessionStrategy):
        self.config = config
        self.session = session

    def _resolve_model(self, model: str) -> tuple[str, bool, bool]:
        """解析模型名 → (model_type, thinking_enabled, search_enabled)"""
        model_lower = model.lower()

        # 精确匹配
        if model_lower in ("deepseek-v4-flash", "deepseek-v3", "deepseek-v3.2", "default"):
            return "default", False, False
        if model_lower == "deepseek-v4-pro":
            return "default", True, True
        # deepseek-chat — 将于 2026/07/24 弃用，请迁移至 deepseek-v4-flash
        if model_lower == "deepseek-chat":
            return "default", False, False
        # deepseek-reasoner — 将于 2026/07/24 弃用，实际等同于 deepseek-v4-flash 的思考模式
        if model_lower == "deepseek-reasoner":
            return "default", True, False
        if "search" in model_lower:
            return "default", False, True
        if "think" in model_lower or "reasoner" in model_lower or "r1" in model_lower:
            return "default", True, False

        # 默认
        return "default", False, False

    def build_prompt(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        response_format: Optional[dict] = None,
        web_search: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """构建 ChatML prompt"""

        # 如果 request 中显式指定了搜索/推理, 但不改变 prompt 内容 (仅影响 adapter 参数)
        # prompt 构建只用 messages + tools + response_format
        return build_chatml_prompt(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    async def chat(
        self,
        messages: list[dict],
        model: str = "deepseek-v4-flash",
        stream: bool = True,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        response_format: Optional[dict] = None,
        web_search: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """
        执行一次对话请求。

        Returns:
            如果是 stream=True: httpx 流式响应 (可通过 full_sse_pipeline 解析)
        """
        # 解析模型参数
        model_type, default_thinking, default_search = self._resolve_model(model)

        # 显式参数覆盖
        thinking_enabled = default_thinking
        search_enabled = default_search

        if reasoning_effort:
            thinking_enabled = True
        if web_search:
            search_enabled = True

        # 构建 prompt
        prompt = self.build_prompt(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            web_search=web_search,
            reasoning_effort=reasoning_effort,
        )

        # 执行
        resp = await self.session.execute(
            prompt=prompt,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            model_type=model_type,
        )

        return resp