"""
DeepSeek Proxy — 入口

Usage:
    python -m deepseek_proxy.main
    # 或
    python main.py
"""

import sys
import logging
import asyncio

from .config import CONFIG
from .server import init_server, shutdown_server, create_app


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 降低 httpx 日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main():
    setup_logging()
    logger = logging.getLogger("deepseek_proxy")

    logger.info("启动 DeepSeek Proxy...")
    logger.info("认证模式: %s", CONFIG.auth_mode.value)
    logger.info("会话模式: %s", CONFIG.session_mode.value)
    logger.info("服务器: %s:%d", CONFIG.server_host, CONFIG.server_port)

    # 初始化
    state = await init_server(CONFIG)

    # 创建 FastAPI app
    app = create_app(CONFIG)

    # 启动 uvicorn
    import uvicorn
    config = uvicorn.Config(
        app,
        host=CONFIG.server_host,
        port=CONFIG.server_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("收到关闭信号...")
    finally:
        await shutdown_server()
        logger.info("DeepSeek Proxy 已关闭")


if __name__ == "__main__":
    asyncio.run(main())


def cli():
    """命令行入口 (供 console_scripts 使用)"""
    asyncio.run(main())