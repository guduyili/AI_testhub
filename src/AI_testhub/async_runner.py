from __future__ import annotations

"""后台异步执行器。

真实平台通常不会在请求线程里同步跑完整浏览器任务，而是把任务放到后台，
再通过 SSE/WebSocket 持续推送事件。学习版使用 threading 模拟这个形态：
- AsyncTaskRunner.start 立即返回 handle；
- 后台线程调用 BrowserAgent.run_full_process；
- handle.events 收集结构化事件；
- handle.wait 等待最终 ExecutionHistory。
"""

from dataclasses import dataclass, field
import threading
from typing import Any, Dict, List, Optional

from .agent import BrowserAgent
from .history import ExecutionHistory


@dataclass
class AsyncRunHandle:
    """一次后台运行的句柄。"""

    thread: threading.Thread
    events: List[Dict[str, Any]] = field(default_factory=list)
    history: Optional[ExecutionHistory] = None
    error: Optional[BaseException] = None


    @property
    def done(self) -> bool:
        """后台线程是否已经结束。"""
        return not self.thread.is_alive()
    
    def wait(self, timeout: Optional[float] = None) -> ExecutionHistory:
        """等待后台任务结束并返回 ExecutionHistory。"""
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("async task did not finish before timeout")
        if self.error:
            raise self.error
        if self.history is None:
            raise RuntimeError("async task finished without history")
        return self.history
    
class AsyncTaskRunner:
    """用后台线程运行 BrowserAgent 的轻量 runner。"""

    def start(
        self,
        task_description: str,
        action_plan: Optional[List[List[Dict[str, Any]]]] = None,
        stop_before_start: bool = False,
        **agent_kwargs: Any,
    ) -> AsyncRunHandle:
        """启动后台执行并立即返回句柄。

        Args:
            task_description: 待执行的自然语言任务。
            action_plan: 执行计划。
            stop_before_start: 是否在开始前停止。
            **agent_kwargs: 传递给 BrowserAgent 的关键字参数。
        Returns:
            AsyncRunHandle: 后台运行的句柄。
        """

        stop_event = threading.Event()
        if stop_before_start:
            stop_event.set()

        handle = AsyncRunHandle(thread=threading.Thread(target=lambda: None))

        def collect_event(event: Dict[str, Any]) -> None:
            handle.events.append(event)

        def run() -> None:
            try:
                agent = BrowserAgent(action_plan=action_plan, **agent_kwargs)
                handle.history = agent.run_full_process(
                    task_description,
                    step_callback=collect_event,
                    should_stop=stop_event.is_set,
                )
            except BaseException as e:
                handle.error = e

        