from __future__ import annotations

"""执行历史记录结构。

真实项目里执行过程通常会保存到数据库，并通过 SSE/WebSocket 同步到前端。
学习版用一个 dataclass 承载最终结果，方便离线测试和打印演示。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List



@dataclass
class ExecutionHistory:
    """一次浏览器智能任务执行的结果快照。

    Attributes:
        status: 整体执行状态，例如 running、passed、failed、stopped。
        planned_tasks: 拆解后的子任务列表，每项含 id/description/status。
        logs: 人类可读日志，模拟真实项目中的执行日志。
        steps: 每一步实际执行/模拟执行的 action 列表，用于调试动作流。
    """


    status: str
    planned_tasks: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)