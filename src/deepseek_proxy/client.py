"""
DeepSeek HTTP 客户端 —— 纯 HTTP 调用层, 无状态, 无缓存, 无重试。

对应 ds-free-api 的 ds_core/client.rs。
每个方法对应一个 REST 端点, 返回原始数据或流。
"""

from __future__ import annotations

import json
import base64
from typing import Optional, AsyncIterator, Any
from dataclasses import dataclass

import httpx

from .config import ProxyConfig


# ═══════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════

@dataclass
class LoginPayload:
    email: Optional[str] = None
    mobile: Optional[str] = None
    password: str = ""
    area_code: Optional[str] = None
    device_id: str = ""
    os: str = "web"

    def to_dict(self) -> dict:
        d = {"password": self.password, "device_id": self.device_id, "os": self.os}
        if self.email:
            d["email"] = self.email
        else:
            d["email"] = ""
        if self.mobile:
            d["mobile"] = self.mobile
        else:
            d["mobile"] = ""
        if self.area_code:
            d["area_code"] = self.area_code
        return d


@dataclass
class UserInfo:
    id: str
    token: str
    email: Optional[str] = None
    mobile_number: Optional[str] = None


@dataclass
class LoginData:
    code: int
    msg: str
    user: UserInfo


@dataclass
class ChallengeData:
    algorithm: str
    challenge: str
    salt: str
    signature: str
    difficulty: int
    expire_after: int
    expire_at: int
    target_path: str


@dataclass
class CompletionPayload:
    chat_session_id: str
    prompt: str
    parent_message_id: Optional[int] = None
    ref_file_ids: Optional[list[str]] = None
    thinking_enabled: bool = False
    search_enabled: bool = False

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "chat_session_id": self.chat_session_id,
            "prompt": self.prompt,
            "ref_file_ids": self.ref_file_ids or [],
            "thinking_enabled": self.thinking_enabled,
            "search_enabled": self.search_enabled,
        }
        if self.parent_message_id is not None:
            d["parent_message_id"] = self.parent_message_id
        return d


@dataclass
class EditMessagePayload:
    chat_session_id: str
    message_id: int
    prompt: str
    search_enabled: bool = False
    thinking_enabled: bool = False
    model_type: str = "default"

    def to_dict(self) -> dict:
        return {
            "chat_session_id": self.chat_session_id,
            "message_id": self.message_id,
            "prompt": self.prompt,
            "search_enabled": self.search_enabled,
            "thinking_enabled": self.thinking_enabled,
            "model_type": self.model_type,
        }


# ═══════════════════════════════════════════════════════════
# API 端点常量
# ═══════════════════════════════════════════════════════════

ENDPOINT_USERS_LOGIN = "/users/login"
ENDPOINT_USERS_CURRENT = "/users/current"
ENDPOINT_CHAT_SESSION_CREATE = "/chat_session/create"
ENDPOINT_CHAT_SESSION_DELETE = "/chat_session/delete"
ENDPOINT_CHAT_CREATE_POW_CHALLENGE = "/chat/create_pow_challenge"
ENDPOINT_CHAT_COMPLETION = "/chat/completion"
ENDPOINT_CHAT_EDIT_MESSAGE = "/chat/edit_message"


# ═══════════════════════════════════════════════════════════
# 错误
# ═══════════════════════════════════════════════════════════

class ClientError(Exception):
    """客户端层错误"""
    pass


class HTTPStatusError(ClientError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class BusinessError(ClientError):
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"Business error: code={code}, msg={msg}")


# ═══════════════════════════════════════════════════════════
# DeepSeek Client
# ═══════════════════════════════════════════════════════════

class DsClient:
    """DeepSeek HTTP 客户端 — 无状态, 每个方法对应一个端点"""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self._http = httpx.AsyncClient(timeout=config.http_timeout)

    # ── 请求头构建 ──────────────────────────────────────

    def _base_headers(self, token: Optional[str] = None,
                      pow_response: Optional[str] = None) -> dict[str, str]:
        h = {
            "User-Agent": self.config.user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/",
            "X-App-Version": self.config.app_version,
            "X-Client-Locale": self.config.client_locale,
            "X-Client-Platform": self.config.client_platform,
            "X-Client-Version": self.config.client_version,
            "Content-Type": "application/json",
        }
        if token:
            h["Authorization"] = f"Bearer {token}"
        if pow_response:
            h["X-Ds-Pow-Response"] = pow_response
        return h

    # ── 信封解析 ────────────────────────────────────────

    async def _parse_envelope(self, resp: httpx.Response) -> dict:
        """解析 Envelope: {code, msg, data: {biz_code, biz_msg, biz_data}}"""
        if resp.status_code != 200:
            body = resp.text
            raise HTTPStatusError(resp.status_code, body)

        envelope = resp.json()
        code = envelope.get("code", -1)
        if code != 0:
            raise BusinessError(code, envelope.get("msg", "unknown"))

        data = envelope.get("data")
        if not data:
            raise BusinessError(-1, "missing data")

        biz_code = data.get("biz_code", -1)
        if biz_code != 0:
            raise BusinessError(biz_code, data.get("biz_msg", "unknown"))

        biz_data = data.get("biz_data")
        if biz_data is None:
            raise BusinessError(-1, "missing biz_data")

        return biz_data

    # ── API 方法 ────────────────────────────────────────

    async def login(self, payload: LoginPayload) -> LoginData:
        """邮箱/手机号 + 密码登录"""
        resp = await self._http.post(
            f"{self.config.api_base}{ENDPOINT_USERS_LOGIN}",
            headers=self._base_headers(),
            json=payload.to_dict(),
        )
        biz_data = await self._parse_envelope(resp)

        user = biz_data.get("user", {})
        return LoginData(
            code=biz_data.get("code", 0),
            msg=biz_data.get("msg", ""),
            user=UserInfo(
                id=user.get("id", ""),
                token=user.get("token", ""),
                email=user.get("email"),
                mobile_number=user.get("mobile_number"),
            ),
        )

    async def get_current_user(self, token: str) -> dict:
        """验证 token 并获取用户信息"""
        resp = await self._http.get(
            f"{self.config.api_base}{ENDPOINT_USERS_CURRENT}",
            headers=self._base_headers(token=token),
        )
        return await self._parse_envelope(resp)

    async def create_session(self, token: str) -> str:
        """创建聊天会话, 返回 session_id"""
        resp = await self._http.post(
            f"{self.config.api_base}{ENDPOINT_CHAT_SESSION_CREATE}",
            headers=self._base_headers(token=token),
            json={"character_id": None},
        )
        biz_data = await self._parse_envelope(resp)
        # biz_data 可能是 {"chat_session": {"id": "..."}} 或 {"id": "..."}
        if "chat_session" in biz_data:
            return biz_data["chat_session"]["id"]
        return biz_data["id"]

    async def delete_session(self, token: str, session_id: str) -> None:
        """删除会话"""
        resp = await self._http.post(
            f"{self.config.api_base}{ENDPOINT_CHAT_SESSION_DELETE}",
            headers=self._base_headers(token=token),
            json={"chat_session_id": session_id},
        )
        await self._parse_envelope(resp)  # 验证成功

    async def create_pow_challenge(self, token: str,
                                   target_path: str = "/api/v0/chat/completion") -> ChallengeData:
        """获取 PoW 挑战参数"""
        resp = await self._http.post(
            f"{self.config.api_base}{ENDPOINT_CHAT_CREATE_POW_CHALLENGE}",
            headers=self._base_headers(token=token),
            json={"target_path": target_path},
        )
        biz_data = await self._parse_envelope(resp)
        # biz_data 可能是 {"challenge": {...}} 或直接是 challenge
        challenge = biz_data.get("challenge", biz_data)
        return ChallengeData(
            algorithm=challenge["algorithm"],
            challenge=challenge["challenge"],
            salt=challenge["salt"],
            signature=challenge["signature"],
            difficulty=challenge["difficulty"],
            expire_after=challenge.get("expire_after", 0),
            expire_at=challenge["expire_at"],
            target_path=challenge.get("target_path", target_path),
        )

    async def completion(self, token: str, pow_response: str,
                         payload: CompletionPayload) -> httpx.Response:
        """发起对话 completion (Chat2API 方式), 返回流式响应"""
        resp = await self._http.send(
            self._http.build_request(
                "POST",
                f"{self.config.api_base}{ENDPOINT_CHAT_COMPLETION}",
                headers=self._base_headers(token=token, pow_response=pow_response),
                json=payload.to_dict(),
            ),
            stream=True,
        )
        if resp.status_code != 200:
            body = await resp.aread()
            raise HTTPStatusError(resp.status_code, body.decode())
        return resp

    async def edit_message(self, token: str, pow_response: str,
                           payload: EditMessagePayload) -> httpx.Response:
        """编辑已有消息 (ds-free-api 方式), 返回流式响应"""
        resp = await self._http.send(
            self._http.build_request(
                "POST",
                f"{self.config.api_base}{ENDPOINT_CHAT_EDIT_MESSAGE}",
                headers=self._base_headers(token=token, pow_response=pow_response),
                json=payload.to_dict(),
            ),
            stream=True,
        )
        if resp.status_code != 200:
            body = await resp.aread()
            raise HTTPStatusError(resp.status_code, body.decode())
        return resp

    async def close(self) -> None:
        await self._http.aclose()


# ═══════════════════════════════════════════════════════════
# PoW Header 构造工具
# ═══════════════════════════════════════════════════════════

def build_pow_header(algorithm: str, challenge: str, salt: str,
                     answer: int, signature: str, target_path: str) -> str:
    """构造 X-Ds-Pow-Response header 值 (base64)"""
    payload = {
        "algorithm": algorithm,
        "challenge": challenge,
        "salt": salt,
        "answer": answer,
        "signature": signature,
        "target_path": target_path,
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()