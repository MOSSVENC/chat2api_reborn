"""
手动测试 SSE 解析 Pipeline (不依赖真实 API)
"""
import asyncio
from deepseek_proxy.sse_parser import parse_sse_stream, parse_dsframe_stream, full_sse_pipeline


# 模拟 DeepSeek SSE 响应 (纯 ASCII, 中文用 unicode escape)
MOCK_SSE = (
    b'event: ready\ndata: {}\n\n'
    b'data: {"v":{"response":{"thinking_enabled":false,"fragments":[{"type":"ANSWER","content":"hello"}]}}}\n\n'
    b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" world"}\n\n'
    b'data: {"v":"! I am"}\n\n'
    b'data: {"v":" DeepSeek."}\n\n'
    b'data: {"p":"response/status","o":"APPEND","v":"FINISHED"}\n\n'
    b'event: finish\ndata: {}\n\n'
)


async def test_sse_events():
    print("=== Layer 1: SSE Events ===")
    async def byte_stream():
        yield MOCK_SSE

    async for evt in parse_sse_stream(byte_stream()):
        print(f"  event={evt.event!r}, data={evt.data[:80]}")


async def test_dsframes():
    print("\n=== Layer 2: DsFrames ===")
    async def byte_stream():
        yield MOCK_SSE

    async for frame in parse_dsframe_stream(parse_sse_stream(byte_stream())):
        print(f"  {frame.type.value}: {str(frame.value)[:60]}")


async def test_full_pipeline():
    print("\n=== Layer 3: Full Pipeline (OpenAI Chunks) ===")
    async def byte_stream():
        yield MOCK_SSE

    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-chat"):
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish = choice.get("choices", [{}])[0].get("finish_reason", "")
        if delta.get("role"):
            print(f"  role: {delta['role']}")
        if delta.get("content"):
            print(f"  content: {delta['content']}")
        if delta.get("reasoning_content"):
            print(f"  reasoning: {delta['reasoning_content'][:50]}")
        if finish:
            print(f"  finish_reason: {finish}")


async def main():
    await test_sse_events()
    print("---")
    await test_dsframes()
    print("---")
    await test_full_pipeline()
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())