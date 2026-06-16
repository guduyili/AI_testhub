from __future__ import annotations

"""真实 browser-use 执行器。

这个模块是离线学习版和真实浏览器执行之间的桥接层：
1. 使用 BrowserAgent 复用配置加载、任务拆解和执行 Prompt 构造逻辑；
2. 使用 build_chat_openai(..., use_real=True) 创建真实 LangChain ChatOpenAI；
3. 封装 browser_use.Agent，把真实 Agent.run() 的结果转换为当前项目统一的
   ExecutionHistory / AgentEvent 风格结果。

注意：单元测试通过 agent_class 注入 fake agent，不会真的打开浏览器。
真正运行 examples/real_browser_demo.py 时才会调用 browser-use。
"""

from dataclasses import dataclass, field
import asyncio
from typing import Any, Callable, Dict, List, Optional, Type

from .agent import BrowserAgent, build_chat_openai
from .config import AIModelConfig, InMemoryConfigStore
from .events import EventBus, EventType
from .history import ExecutionHistory


@dataclss
class RealBrowserRunResult:
    """真实 browser-use 运行结果。"""

    history: ExecutionHistory
    raw_result: Any = None
    events: List[Dict[str, Any]] = field(default_factory=list)

class RealBrowserRunner:
    """封装 browser_use.Agent 的真实执行入口。"""

    def __init__(
        self,
        agent_class: Optional[Type[Any]] = None,
        browser_profile_class: Optional[Type[Any]] = None,
    ):
        self.agent_class = agent_class
        self.browser_profile_class = browser_profile_class

    def _load_browser_use_classes(self) -> tuple[Type[Any], Optional[Type[Any]]]:
        """延迟导入 browser-use，避免离线测试强依赖真实浏览器包。"""
        if self.agent_class is not None:
            return self.agent_class, self.browser_profile_class

        try:
            from browser_use import Agent, BrowserProfile
        except ImportError as exc:
            raise RuntimeError("browser-use is not installed. Install with: uv sync --extra real --extra test") from exc
        return Agent, BrowserProfile

    def _build_browser_profile(self, headless: bool) -> Any:
        """创建 browser-use BrowserProfile；测试可通过 browser_profile_class=None 跳过。"""
        _, browser_profile_class = self._load_browser_use_classes()
        if browser_profile_class is None:
            return None
        return browser_profile_class(headless=headless)
    
    async def run_async(
            self,
            task_description: str,
            config: AIModelConfig,
            headless: bool = True,
            use_real_llm: bool = True,
            callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            max_steps: Optional[int] = None,
    ) -> RealBrowserRunResult:
        """异步运行 browser-use 任务。"""
        events: List[Dict[str, Any]] = []

        def collect(event: Dict[str, Any]) -> None:
            events.append(event)
            if callback:
                callback(event)

            bus = EventBus(collect)