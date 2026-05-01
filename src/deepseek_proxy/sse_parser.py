"""
SSE 流解析 Pipeline — 三层转换: Byte → SSE Event → DsFrame → OpenAI Chunk

对应 ds-free-api:
- openai_adapter/response/sse_parser.rs  (SseStream)
- openai_adapter/response/state.rs       (StateStream → DsFrame)
- openai_adapter/response/converter.rs   (ConverterStream → ChatCompletionChunk)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, AsyncIterator, Any, Callable


# ═══════════════════════════════════════════════════════════
# Layer 1: SSE Event
# ═══════════════════════════════════════════════════════════

@dataclass
class SseEvent:
    event: Optional[str] = None
    data: str = ""


async def parse_sse_stream(byte_stream) -> AsyncIterator[SseEvent]:
    """
    Layer 1: 原始字节流 → SSE 事件.

    以 \\n\\n 为事件分隔符，解析 event: 和 data: 字段。
    """
    buf = b""

    async for chunk in byte_stream:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        buf += chunk

        while b"\n\n" in buf:
            raw_event, buf = buf.split(b"\n\n", 1)
            evt = _parse_raw_event(raw_event.decode("utf-8", errors="replace"))
            if evt:
                yield evt

    # 末尾残留
    if buf.strip():
        evt = _parse_raw_event(buf.decode("utf-8", errors="replace"))
        if evt:
            yield evt


def _parse_raw_event(text: str) -> Optional[SseEvent]:
    """解析单个 SSE 事件块"""
    event = None
    data_lines = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if not data_lines:
        return None

    return SseEvent(event=event, data="\n".join(data_lines))


# ═══════════════════════════════════════════════════════════
# Layer 2: DsFrame (DeepSeek Frame)
# ═══════════════════════════════════════════════════════════

class DsFrameType(Enum):
    ROLE = "role"
    THINK_DELTA = "think_delta"
    CONTENT_DELTA = "content_delta"
    STATUS = "status"
    USAGE = "usage"
    FINISH = "finish"


@dataclass
class DsFrame:
    type: DsFrameType
    value: Any = None  # str for THINK/CONTENT/STATUS, int for USAGE


FRAG_THINK = "THINK"
FRAG_RESPONSE = "RESPONSE"
FRAG_ANSWER = "ANSWER"


class DsState:
    """DeepSeek Patch 状态机 — 维护 p/o/v 路径操作状态"""

    def __init__(self):
        self.current_path: Optional[str] = None
        self.fragments: list[dict] = []  # [{type, content}]
        self.status: Optional[str] = None
        self.accumulated_token_usage: Optional[int] = None

    def apply_event(self, evt: SseEvent) -> list[DsFrame]:
        frames: list[DsFrame] = []

        # 事件类型
        if evt.event == "ready":
            frames.append(DsFrame(type=DsFrameType.ROLE))
        elif evt.event == "finish":
            frames.append(DsFrame(type=DsFrameType.FINISH))

        # 解析 data
        if evt.data and evt.data.strip():
            try:
                val = json.loads(evt.data)
            except json.JSONDecodeError:
                return frames
            frames.extend(self._apply_patch_value(val))

        return frames

    def _apply_patch_value(self, val: dict) -> list[DsFrame]:
        frames: list[DsFrame] = []
        has_p = "p" in val
        op = val.get("o")

        if has_p:
            p = val["p"]
            if isinstance(p, str):
                self.current_path = p

        v = val.get("v")
        if v is None:
            return frames

        if has_p or op is not None:
            path = self.current_path
            if not path:
                return frames

            if path == "response" and op == "BATCH":
                if isinstance(v, list):
                    for item in v:
                        frames.extend(self._apply_patch_value(item))
            else:
                frames.extend(self._apply_path(path, op, v))

        elif self.current_path:
            # 继承 current_path 的纯 v 增量, 隐含 APPEND
            frames.extend(self._apply_path(self.current_path, "APPEND", v))

        else:
            # 初始 snapshot: 无 current_path
            self._apply_snapshot(v, frames)

        return frames

    def _apply_snapshot(self, v: Any, frames: list[DsFrame]) -> None:
        """处理初始快照: {v: {response: {thinking_enabled, fragments: [...]}}}"""
        if not isinstance(v, dict):
            return

        response = v.get("response")
        if not isinstance(response, dict):
            return

        thinking_enabled = response.get("thinking_enabled")
        # 不直接输出, 仅用于 context

        fragments = response.get("fragments")
        if not isinstance(fragments, list):
            return

        self.fragments = []
        for frag in fragments:
            if not isinstance(frag, dict):
                continue
            ty = frag.get("type", "")
            content = frag.get("content", "")
            if not isinstance(content, str):
                content = ""

            self.fragments.append({"type": ty, "content": content})

            if content:
                if ty == FRAG_THINK:
                    frames.append(DsFrame(type=DsFrameType.THINK_DELTA, value=content))
                elif ty in (FRAG_RESPONSE, FRAG_ANSWER):
                    frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=content))

    def _apply_path(self, path: str, op: Optional[str], v: Any) -> list[DsFrame]:
        frames: list[DsFrame] = []

        if path in ("response/status",):
            if isinstance(v, str):
                self.status = v
                frames.append(DsFrame(type=DsFrameType.STATUS, value=v))

        elif path in ("response/accumulated_token_usage", "accumulated_token_usage"):
            if isinstance(v, (int, float)):
                u = int(v)
                self.accumulated_token_usage = u
                frames.append(DsFrame(type=DsFrameType.USAGE, value=u))

        elif path in ("response/search_status", "response/search_results"):
            pass  # 忽略搜索相关

        elif path == "response/fragments/-1/content":
            if isinstance(v, str) and self.fragments:
                last = self.fragments[-1]
                ty = last.get("type", "")
                last["content"] += v
                if ty == FRAG_THINK:
                    frames.append(DsFrame(type=DsFrameType.THINK_DELTA, value=v))
                elif ty in (FRAG_RESPONSE, FRAG_ANSWER):
                    frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=v))

        elif path == "response/fragments/-1/elapsed_secs":
            pass  # 忽略耗时

        elif path == "response/fragments" and op == "APPEND":
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        ty = item.get("type", "")
                        content = item.get("content", "")
                        if not isinstance(content, str):
                            content = ""
                        self.fragments.append({"type": ty, "content": content})
                        if content:
                            if ty == FRAG_THINK:
                                frames.append(DsFrame(type=DsFrameType.THINK_DELTA, value=content))
                            elif ty in (FRAG_RESPONSE, FRAG_ANSWER):
                                frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=content))

        return frames


async def parse_dsframe_stream(sse_stream) -> AsyncIterator[DsFrame]:
    """
    Layer 2: SSE 事件流 → DsFrame 增量帧.

    应用 patch 状态机, 将 p/o/v 操作转为语义化帧。
    """
    state = DsState()

    async for evt in sse_stream:
        frames = state.apply_event(evt)
        for frame in frames:
            yield frame


# ═══════════════════════════════════════════════════════════
# Layer 3: OpenAI Chunk
# ═══════════════════════════════════════════════════════════

@dataclass
class Delta:
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


@dataclass
class ChunkChoice:
    index: int = 0
    delta: Delta = field(default_factory=Delta)
    finish_reason: Optional[str] = None


# Tool call 累积状态
@dataclass
class ToolCallAccum:
    id: str = ""
    name: str = ""
    arguments: str = ""


async def convert_to_openai_chunks(
    ds_frame_stream,
    model: str = "deepseek-chat",
    include_usage: bool = False,
    prompt_tokens: int = 0,
) -> AsyncIterator[dict]:
    """
    Layer 3: DsFrame 增量帧 → OpenAI ChatCompletionChunk (dict).

    输出标准 OpenAI 流式响应格式:
    {"id": "...", "object": "chat.completion.chunk", "choices": [...], ...}
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    finished = False
    usage_value: Optional[int] = None

    # 工具调用解析状态 (简化: 从 content 中提取 <tool_calls> XML)
    tool_buffer: str = ""
    tool_calls_active = False
    tool_call_accums: list[ToolCallAccum] = []

    def make_chunk(delta: Delta, finish_reason: Optional[str] = None) -> dict:
        c = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        }
        if delta.role:
            c["choices"][0]["delta"]["role"] = delta.role
        if delta.content is not None:
            c["choices"][0]["delta"]["content"] = delta.content
        if delta.reasoning_content is not None:
            c["choices"][0]["delta"]["reasoning_content"] = delta.reasoning_content
        if delta.tool_calls:
            c["choices"][0]["delta"]["tool_calls"] = delta.tool_calls
        return c

    def make_usage_chunk(prompt: int, completion: int) -> dict:
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }

    try:
        async for frame in ds_frame_stream:
            if frame.type == DsFrameType.ROLE:
                yield make_chunk(Delta(role="assistant"))

            elif frame.type == DsFrameType.THINK_DELTA:
                text = _clean_text(frame.value)
                if text:
                    yield make_chunk(Delta(reasoning_content=text))

            elif frame.type == DsFrameType.CONTENT_DELTA:
                text = _clean_text(frame.value)
                if not text:
                    continue

                # 检测工具调用开始
                if not tool_calls_active and "<tool_calls>" in text:
                    tool_calls_active = True
                    # 取 <tool_calls> 之前的内容 (如果有)
                    before, after = text.split("<tool_calls>", 1)
                    if before.strip():
                        yield make_chunk(Delta(content=before))
                    tool_buffer = after
                    continue

                if tool_calls_active:
                    tool_buffer += text

                    # 检测工具调用结束
                    if "</tool_calls>" in tool_buffer:
                        tool_calls_active = False
                        parsed = _parse_tool_calls_buffer(tool_buffer)
                        if parsed:
                            # 转为 OpenAI tool_calls delta 格式
                            tc_deltas = []
                            for i, tc in enumerate(parsed):
                                tc_deltas.append({
                                    "index": i,
                                    "id": f"call_{uuid.uuid4().hex[:24]}",
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", ""),
                                        "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                                    },
                                })
                            yield make_chunk(Delta(tool_calls=tc_deltas))
                        tool_buffer = ""
                    continue

                yield make_chunk(Delta(content=text))

            elif frame.type == DsFrameType.STATUS:
                status = str(frame.value) if frame.value else ""
                if status == "FINISHED" and not finished:
                    finished = True
                    yield make_chunk(Delta(), finish_reason="stop")

            elif frame.type == DsFrameType.USAGE:
                usage_value = int(frame.value) if frame.value else 0
                if finished and include_usage:
                    yield make_usage_chunk(prompt_tokens, usage_value)

            elif frame.type == DsFrameType.FINISH:
                if not finished:
                    finished = True
                    yield make_chunk(Delta(), finish_reason="stop")

    finally:
        # 确保结束时发送 usage (如果启用)
        if finished and include_usage and usage_value is not None:
            not_sent = True  # 简单处理: 如果没发过就补发
            if not_sent:
                yield make_usage_chunk(prompt_tokens, usage_value)


def _clean_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    # 移除 FINISHED 标记
    text = text.replace("FINISHED", "")
    # 移除搜索关键词前缀
    import re
    text = re.sub(r'^(SEARCH|WEB_SEARCH|SEARCHING)\s*', '', text, flags=re.IGNORECASE)
    return text


def _parse_tool_calls_buffer(buf: str) -> list[dict]:
    """从 tool_buffer 中提取工具调用 JSON 数组"""
    buf = buf.replace("</tool_calls>", "")
    buf = buf.strip()
    try:
        parsed = json.loads(buf)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    return []


# ═══════════════════════════════════════════════════════════
# 完整 Pipeline (便捷函数)
# ═══════════════════════════════════════════════════════════

async def full_sse_pipeline(
    byte_stream,
    model: str = "deepseek-chat",
    include_usage: bool = False,
    prompt_tokens: int = 0,
) -> AsyncIterator[dict]:
    """
    完整 SSE Pipeline: 字节流 → OpenAI Chunks.

    Layer 1: bytes → SseEvent
    Layer 2: SseEvent → DsFrame (patch 状态机)
    Layer 3: DsFrame → OpenAI Chunk
    """
    sse_stream = parse_sse_stream(byte_stream)
    ds_frame_stream = parse_dsframe_stream(sse_stream)

    async for chunk in convert_to_openai_chunks(
        ds_frame_stream,
        model=model,
        include_usage=include_usage,
        prompt_tokens=prompt_tokens,
    ):
        yield chunk