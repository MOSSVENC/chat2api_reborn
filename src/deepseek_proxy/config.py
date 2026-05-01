"""
DeepSeek Proxy — 配置文件 (硬编码, 后续迁移到 config.toml)
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AuthMode(str, Enum):
    PASSWORD = "password"       # 邮箱/手机号 + 密码自动登录
    TOKEN = "token"              # 手动提供 Bearer Token


class SessionMode(str, Enum):
    REUSE = "reuse"              # 初始化时创建 session, 永久复用 edit_message
    NEW = "new"                  # 每次请求新建 session + completion


class DeepSeekModel(str, Enum):
    DEFAULT = "default"          # DeepSeek-V3.2
    EXPERT = "expert"            # DeepSeek-R1
    DEFAULT_SEARCH = "default_search"   # V3.2 + 搜索
    EXPERT_SEARCH = "expert_search"     # R1 + 搜索
    DEFAULT_THINK = "default_think"     # V3.2 + 深度思考
    EXPERT_THINK = "expert_think"       # R1 + 深度思考

    @property
    def model_type(self) -> str:
        """映射到 DeepSeek API 的 model_type 字段"""
        mapping = {
            DeepSeekModel.DEFAULT: "default",
            DeepSeekModel.EXPERT: "expert",
            DeepSeekModel.DEFAULT_SEARCH: "default",
            DeepSeekModel.EXPERT_SEARCH: "expert",
            DeepSeekModel.DEFAULT_THINK: "default",
            DeepSeekModel.EXPERT_THINK: "expert",
        }
        return mapping[self]

    @property
    def thinking_enabled(self) -> bool:
        return self in (DeepSeekModel.DEFAULT_THINK, DeepSeekModel.EXPERT_THINK,
                        DeepSeekModel.EXPERT, DeepSeekModel.EXPERT_SEARCH)

    @property
    def search_enabled(self) -> bool:
        return self in (DeepSeekModel.DEFAULT_SEARCH, DeepSeekModel.EXPERT_SEARCH)


@dataclass
class ProxyConfig:
    # === DeepSeek API 连接 ===
    api_base: str = "https://chat.deepseek.com/api/v0"
    wasm_url: str = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"

    # === HTTP 请求头 ===
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    )
    client_version: str = "1.8.0"
    client_platform: str = "web"
    app_version: str = "20241129.1"
    client_locale: str = "zh-CN"

    # === 认证 ===
    auth_mode: AuthMode = AuthMode.PASSWORD

    # --- AuthMode.PASSWORD 专用 ---
    account_email: str = ""
    account_mobile: str = ""
    account_area_code: str = ""
    account_password: str = ""

    # --- AuthMode.TOKEN 专用 ---
    user_token: str = ""

    # === 会话策略 ===
    session_mode: SessionMode = SessionMode.REUSE

    # === 服务器 ===
    server_host: str = "127.0.0.1"
    server_port: int = 5317
    api_tokens: list[str] = field(default_factory=list)  # 空 = 不鉴权

    # === 模型 ===
    default_model: DeepSeekModel = DeepSeekModel.DEFAULT
    model_types: list[str] = field(default_factory=lambda: ["default", "expert"])

    # === 超时 (秒) ===
    http_timeout: float = 120.0
    token_refresh_interval: int = 3000  # 秒 (50 分钟, 保守)

    @classmethod
    def from_dict(cls, d: dict) -> "ProxyConfig":
        """从字典构造, 覆盖默认值"""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in fields}
        # 处理枚举
        if "auth_mode" in kwargs and isinstance(kwargs["auth_mode"], str):
            kwargs["auth_mode"] = AuthMode(kwargs["auth_mode"])
        if "session_mode" in kwargs and isinstance(kwargs["session_mode"], str):
            kwargs["session_mode"] = SessionMode(kwargs["session_mode"])
        if "default_model" in kwargs and isinstance(kwargs["default_model"], str):
            kwargs["default_model"] = DeepSeekModel(kwargs["default_model"])
        return cls(**kwargs)


# === 默认全局配置 (硬编码, 开发阶段直接改这里) ===
CONFIG = ProxyConfig(
 # 认证: 改为 TOKEN 模式则填 user_token
 auth_mode=AuthMode.TOKEN,

 # 账号密码登录 (auth_mode=password 时使用)
 account_email="",
 account_mobile="",
 account_area_code="",
 account_password="",

 # Bearer Token (auth_mode=token 时使用)
 user_token="1TR1JHiK3s20p9PQslvGsXn9skWljioxUhKBqTP4FQhmG+cmRMYWiTmmo2sReb1N",

 # 此token已作废，已畏惧，不要再用我的token了
 # I'm scared. Don't use up my tokens! This token has been revoked.
    # 会话策略
    session_mode=SessionMode.REUSE,

    # 服务器
    server_host="127.0.0.1",
    server_port=5317,
    api_tokens=[],   # 空 = 无需 API key
)