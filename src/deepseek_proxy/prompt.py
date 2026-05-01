"""
ChatML Prompt 构建 — 将 OpenAI messages 转换为 ChatML 格式字符串

对应 ds-free-api: openai_adapter/request/prompt.rs
使用 <|im_start|> / <|im_end|> 分隔符。
"""

from __future__ import annotations

import json
from typing import Optional, Any, Dict, List

from .client import EditMessagePayload


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_chatml_prompt(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
    response_format: Optional[dict] = None,
) -> str:
    """
    将 OpenAI chat messages 转换为 ChatML prompt 字符串。

    Args:
        messages: [{"role": "user", "content": "..."}, ...]
        tools: OpenAI tools 定义
        tool_choice: 工具选择策略
        response_format: {"type": "json_object"} 等

    Returns:
        ChatML 格式字符串, 末尾不闭合 assistant 块
    """
    parts: list[str] = []

    for msg in messages:
        parts.append(format_message(msg))

    # 工具定义和指令注入 (reminder 块)
    extra_blocks: list[str] = []

    if tools:
        tools_text = _format_tools(tools, tool_choice)
        if tools_text:
            extra_blocks.append(tools_text)

    if response_format:
        rf_text = _format_response_format(response_format)
        if rf_text:
            extra_blocks.append(rf_text)

    if extra_blocks:
        reminder = "\n\n".join(extra_blocks)
        parts.append(f"{IM_START}reminder\n{reminder}\n{IM_END}")

    # 末尾: <|im_start|>assistant (不闭合, 让模型生成)
    parts.append(f"{IM_START}assistant")

    return "\n".join(parts)


def format_message(msg: dict) -> str:
    role = msg.get("role", "user")
    body = _format_body(msg, role)
    return f"{IM_START}{role}\n{body}\n{IM_END}"


def _format_body(msg: dict, role: str) -> str:
    parts = []

    if role == "assistant":
        if msg.get("content"):
            parts.append(_format_content(msg["content"]))
        if msg.get("tool_calls"):
            parts.append(_format_tool_calls(msg["tool_calls"]))
        if msg.get("function_call"):
            parts.append(_format_function_call(msg["function_call"]))
        if msg.get("refusal"):
            parts.append(f"(refusal: {msg['refusal']})")

    elif role == "tool":
        if msg.get("tool_call_id"):
            parts.append(f"(tool_call_id: {msg['tool_call_id']})")
        if msg.get("content"):
            parts.append(_format_content(msg["content"]))

    elif role == "function":
        if msg.get("name"):
            parts.append(f"(name: {msg['name']})")
        if msg.get("content"):
            parts.append(_format_content(msg["content"]))

    else:  # user, system
        if msg.get("name"):
            parts.append(f"(name: {msg['name']})")
        if msg.get("content"):
            parts.append(_format_content(msg["content"]))

    return "\n".join(parts)


def _format_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    detail = part.get("image_url", {}).get("detail", "auto")
                    texts.append(f"[图片: detail={detail}]")
                elif part.get("type") == "input_audio":
                    fmt = part.get("input_audio", {}).get("format", "unknown")
                    texts.append(f"[音频: format={fmt}]")
                elif part.get("type") == "file":
                    filename = part.get("file", {}).get("filename", "unknown")
                    texts.append(f"[文件: filename={filename}]")
                else:
                    texts.append(f"[未支持的内容类型: {part.get('type')}]")
            else:
                texts.append(str(part))
        return "\n".join(texts)
    return str(content)


def _format_tool_calls(tool_calls: list[dict]) -> str:
    items = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = func.get("arguments", "{}")
        items.append(json.dumps({"name": name, "arguments": args}, ensure_ascii=False))
    return f"<tool_calls>\n[{', '.join(items)}]\n</tool_calls>"


def _format_function_call(fc: dict) -> str:
    name = fc.get("name", "")
    try:
        args = json.loads(fc.get("arguments", "{}"))
    except (json.JSONDecodeError, TypeError):
        args = fc.get("arguments", "{}")
    item = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"<tool_calls>\n[{item}]\n</tool_calls>"


# ═══════════════════════════════════════════════════════════
# 工具定义格式化
# ═══════════════════════════════════════════════════════════

def _format_tools(tools: list[dict], tool_choice: Any = None) -> str:
    """格式化工具定义 + 使用规范"""
    lines = ["# Tools", ""]

    for i, tool in enumerate(tools):
        func = tool.get("function", tool)
        name = func.get("name", f"function_{i}")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"You may call one or more functions to assist with user query.")
        lines.append("You may use the following functions:")
        lines.append(f"```json")
        lines.append(json.dumps({name: {
            "description": desc,
            "parameters": params,
        }}, ensure_ascii=False, indent=2))
        lines.append("```")

    lines.append("")
    lines.append("For function call returns, you MUST use the following format:")
    lines.append("```")
    lines.append("<tool_calls>")
    lines.append('[{"name": "function_name", "arguments": {"arg1": "value1", ...}}]')
    lines.append("</tool_calls>")
    lines.append("```")

    if tool_choice == "required":
        lines.append("")
        lines.append("注意：你必须调用至少一个工具函数来完成用户的请求。")

    return "\n".join(lines)


def _format_response_format(rf: dict) -> Optional[str]:
    ty = rf.get("type", "text")
    if ty == "json_object":
        return "请直接输出合法的 JSON 对象，不要包含任何 markdown 代码块标记或其他解释性文字。"
    elif ty == "json_schema":
        schema = rf.get("json_schema", {})
        if schema:
            schema_text = json.dumps(schema, ensure_ascii=False)
            return f"请严格遵循以下 JSON Schema 输出，不要包含 schema 之外的内容，不要添加 markdown 代码块。\nSchema: {schema_text}"
        else:
            return "请严格遵循 JSON Schema 输出，不要添加 markdown 代码块。"
    elif ty == "text":
        return None
    else:
        return f"请以 {ty} 格式输出。"