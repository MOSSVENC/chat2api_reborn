"""
DeepSeek Proxy — Python 模块化复刻

基于 Chat2API (TypeScript/Electron) 和 ds-free-api (Rust) 的 DeepSeek 部分分析，
用 Python 实现的无需 Electron 的 DeepSeek 网页 API 反代。

模块:
- config:     配置 (硬编码)
- client:     HTTP 客户端 (对应 ds_core/client.rs)
- auth:       双通道认证 (密码登录 + Token)
- pow_solver: PoW 求解 (wasmtime-py)
- sessions:   会话管理 (复用/新建)
- prompt:     ChatML prompt 构建
- sse_parser: SSE 流解析 Pipeline
- openai_adapter: OpenAI 请求适配
- server:     FastAPI HTTP 服务器
- main:       入口
"""

from .config import ProxyConfig, AuthMode, SessionMode, DeepSeekModel, CONFIG
from .client import DsClient, ClientError, BusinessError, HTTPStatusError

__version__ = "0.1.0"
__all__ = [
    "ProxyConfig", "AuthMode", "SessionMode", "DeepSeekModel", "CONFIG",
    "DsClient", "ClientError", "BusinessError", "HTTPStatusError",
]