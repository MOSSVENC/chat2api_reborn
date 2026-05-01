"""
认证模块 — 双通道: 账号密码登录 + Bearer Token 验证

对应:
- Chat2API: src/main/oauth/adapters/deepseek.ts
- ds-free-api: ds_core/accounts.rs (登录部分)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from .config import ProxyConfig, AuthMode
from .client import DsClient, LoginPayload, UserInfo, LoginData, ClientError


class AuthProvider(ABC):
    """认证提供者抽象基类"""

    @abstractmethod
    async def authenticate(self) -> str:
        """执行认证, 返回 Bearer token"""
        ...

    @abstractmethod
    async def validate(self, token: str) -> bool:
        """验证 token 是否有效"""
        ...

    @abstractmethod
    async def refresh(self, token: str) -> Optional[str]:
        """刷新 token (如果支持), 返回新 token 或 None"""
        ...


class PasswordAuthProvider(AuthProvider):
    """通过邮箱/手机号 + 密码登录获取 token

    对应 ds-free-api 的 init_account() → client.login()
    """

    def __init__(self, config: ProxyConfig, client: DsClient):
        self.config = config
        self.client = client
        self._login_data: Optional[LoginData] = None

    async def authenticate(self) -> str:
        payload = LoginPayload(
            email=self.config.account_email or None,
            mobile=self.config.account_mobile or None,
            password=self.config.account_password,
            area_code=self.config.account_area_code or None,
            device_id="",
            os="web",
        )

        if not payload.email and not payload.mobile:
            raise ValueError("email 和 mobile 不能同时为空")
        if not payload.password:
            raise ValueError("password 不能为空")

        # 3 次重试
        last_error = None
        for attempt in range(1, 4):
            try:
                self._login_data = await self.client.login(payload)
                return self._login_data.user.token
            except ClientError as e:
                last_error = e
                if attempt < 3:
                    import asyncio
                    await asyncio.sleep(2)

        raise last_error or ClientError("login failed after 3 attempts")

    async def validate(self, token: str) -> bool:
        try:
            await self.client.get_current_user(token)
            return True
        except ClientError:
            return False

    async def refresh(self, token: str) -> Optional[str]:
        # 密码登录不能 refresh token, 需要重新登录
        try:
            return await self.authenticate()
        except ClientError:
            return None

    @property
    def account_info(self) -> Optional[dict]:
        if self._login_data:
            return {
                "id": self._login_data.user.id,
                "email": self._login_data.user.email,
                "mobile": self._login_data.user.mobile_number,
            }
        return None


class TokenAuthProvider(AuthProvider):
    """通过手动提供的 Bearer Token 认证

    对应 Chat2API 的 DeepSeekAdapter.validateToken()
    """

    def __init__(self, config: ProxyConfig, client: DsClient):
        self.config = config
        self.client = client
        self._user_info: Optional[dict] = None
        self._last_validated: float = 0

    async def authenticate(self) -> str:
        token = self.config.user_token
        if not token:
            raise ValueError("user_token 不能为空 (在 config.py 中设置)")

        # 验证
        if not await self.validate(token):
            raise ClientError("Token 无效或已过期")

        return token

    async def validate(self, token: str) -> bool:
        try:
            self._user_info = await self.client.get_current_user(token)
            self._last_validated = time.time()
            return True
        except ClientError:
            self._user_info = None
            return False

    async def refresh(self, token: str) -> Optional[str]:
        # Token 模式下直接重新验证, 如果失败就是过期了
        if await self.validate(token):
            return token
        return None

    @property
    def account_info(self) -> Optional[dict]:
        return self._user_info

    @property
    def last_validated(self) -> float:
        return self._last_validated


def create_auth_provider(config: ProxyConfig, client: DsClient) -> AuthProvider:
    """工厂函数: 根据配置创建对应的认证提供者"""
    if config.auth_mode == AuthMode.PASSWORD:
        return PasswordAuthProvider(config, client)
    elif config.auth_mode == AuthMode.TOKEN:
        return TokenAuthProvider(config, client)
    else:
        raise ValueError(f"Unsupported auth_mode: {config.auth_mode}")