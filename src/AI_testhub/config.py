from __future__ import annotations

"""AI 模型配置相关的轻量实现。

原 TestHub 项目中，AI 模型配置通常来自 Django ORM 中的模型表，并按照
role='browser_use_text' 这样的角色筛选当前启用的模型。这个学习项目不依赖
数据库，所以用 dataclass + 内存仓库复刻同样的读取流程，便于理解主链路：

1. 读取当前可用的 AI 模型配置；
2. 将配置传给 ChatOpenAI / FakeChatOpenAI 初始化；
3. 后续 BrowserAgent 只依赖配置对象，不关心配置来自数据库还是内存。
"""

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class AIModelConfig:
    """浏览器自动化专用 AI 模型配置。

    字段含义对应真实项目里的 AIModelConfig：
    - name: 配置名称，方便在后台或日志中识别。
    - model_type: 模型供应商/类型，例如 openai、deepseek、fake。
    - model_name: 具体模型名，例如 gpt-4o-mini 或 fake-model。
    - api_key: 模型调用密钥。学习版允许 fake-key。
    - base_url: OpenAI-compatible API 地址；为空时使用 SDK 默认地址。
    - is_active: 是否启用；仓库只返回启用中的配置。
    - temperature: 采样温度；自动化任务通常使用 0，降低随机性。
    - role: 模型角色；这里固定复刻 browser_use_text 场景。
    """

    name: str
    model_type: str
    model_name: str
    api_key: str
    base_url: str = ""
    is_active: bool = True
    temperature: float = 0.0
    role: str = "browser_use_text"


class InMemoryConfigStore:
    """内存版配置仓库，用来替代真实项目中的数据库查询。

    设计成 repository/store 的形式，是为了让 BrowserAgent 的代码结构接近真实
    项目：Agent 只调用 get_active(role)，不直接关心配置保存在哪里。以后如果
    要换成 JSON 文件、SQLite 或 Django ORM，只需要替换这个仓库实现。
    """
    def __init__(self, configs: Optional[Iterable[AIModelConfig]] = None):
        # 将传入的 Iterable 固化成 list，避免生成器被消费后无法二次读取。
        # self.configs = list(configs or [])
        self.configs = list(configs or [])

    def get_active(self, role: str = "browser_use_text") -> Optional[AIModelConfig]:
        """按角色返回第一个启用中的模型配置。

        返回 None 表示没有配置；BrowserAgent 会在学习模式下回退到 fake 模型。
        真实生产系统通常不应静默回退，而应提示用户先配置模型。
        """
        for config in self.configs:
            if config.is_active and config.role == role:
                return config
        return None