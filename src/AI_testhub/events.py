from __future__ import annotations

"""结构化执行事件协议。

真实项目通常通过 SSE/WebSocket 把浏览器自动化进度推送到前端。
学习版先把事件类型和事件载荷固定下来，使 callback 不再只是松散 dict，
后续可以在此基础上扩展为真正的 SSE 流。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class EventType(str, Enum):
    """浏览器自动化执行过程中的结构化事件类型。"""

    TASK_ANALYSIS = "task_analysis"
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    STEP_LOG = "step_log"
    AUTH_FAILURE = "auth_failure"
    INFRA_FAILURE = "infra_failure"
    PROCESS_COMPLETE = "process_complete"
    PROCESS_FAILED = "process_failed"
    PROCESS_STOPPED = "process_stopped"


@dataclass
class AgentEvent:
    """一次可推送给前端的结构化事件。

    Attributes:
        type: 事件类型，使用 EventType 枚举避免字符串拼写分散。
        task_id: 当前子任务 id；整体流程事件可为空。
        task_description: 当前子任务描述；整体流程事件可为空。
        message: 面向人类阅读的简短消息。
        data: 机器可读的扩展字段，例如 actions、step、status、error。
    """

    type: EventType
    task_id: Optional[int] = None
    task_description: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 SSE/JSON 友好的 dict。"""
        return {
            "type": self.type.value,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "message": self.message,
            "data": self.data,
        }


class EventBus:
    """把 AgentEvent 发送给外部 callback 的轻量事件总线。"""

    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.callback = callback

    def emit(
        self,
        type: EventType,
        task_id: Optional[int] = None,
        task_description: Optional[str] = None,
        message: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        """创建事件并发送给 callback。

        即使没有 callback，也返回 AgentEvent，方便测试和未来记录事件历史。
        """
        event = AgentEvent(
            type=type,
            task_id=task_id,
            task_description=task_description,
            message=message,
            data=data or {},
        )
        if self.callback:
            self.callback(event.to_dict())
        return event
