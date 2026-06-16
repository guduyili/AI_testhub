from __future__ import annotations

"""真实 LLM / browser-use 运行所需的环境配置。

学习版默认不依赖环境变量；真实模式则通过 .env 或 shell 环境读取 OpenAI / browser
参数，避免把敏感配置写进源码。
"""

import os

from .config import AIModelConfig


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def load_real_model_config() -> AIModelConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for real LLM execution")

    return AIModelConfig(
        name="real-env",
        model_type=os.getenv("OPENAI_PROVIDER", "opencode"),
        model_name=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://opencode.ai/zen/go/v1"),
        is_active=True,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
    )