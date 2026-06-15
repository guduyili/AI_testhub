from __future__ import annotations

"""计划任务状态管理。

这个模块专门处理 planned_tasks 的状态变更和整体结果判定。拆出来的好处是：
BrowserAgent 专注执行流程，状态规则可以独立测试，也更容易和前端状态枚举对齐。
"""

from typing import Any, Dict, List, Optional

# 到达这些状态的子任务不会再继续执行。
TERMINAL_TASK_STATUSES = {"completed", "failed", "skipped"}

# 仍然可能被当前步骤处理或补标的状态。
ACTIVE_TASK_STATUSES = {"pending", "in_progress"}



def update_planned_task_status(planned_tasks: List[Dict[str, Any]], task_id: Any, task_status: str) -> bool:
    """按 task_id 更新子任务状态。

    Returns:
        bool: 找到并更新成功返回 True；参数无效或 id 不存在返回 False。
    """
    if not planned_tasks or task_id is None or not task_status:
        return False
    normalized = str(task_status).strip().lower()
    for task in planned_tasks:
        if str(task.get("id")) == str(task_id):
            task["status"] = normalized
            return True
        


def backfill_prior_pending_tasks(planned_tasks: List[Dict[str, Any]], current_task_id: Any) -> List[int]:
    """在安全依赖场景下自动回填上一个 pending 任务为 completed。

    背景：模型有时会直接标记“搜索订单”完成，但忘了标记前置的“访问首页”。如果两者
    是强依赖关系，并且当前任务已经完成，那么前置任务大概率也已经完成，可以安全补标。

    为避免误判，只允许少量强相关任务对：访问 -> 搜索/输入/点击、搜索 -> 点击结果、
    打开详情 -> 关闭/返回等。验证/断言类任务不会被自动补标。
    """
    if not planned_tasks or current_task_id is None:
        return []
    

    try:
        current_id = int(current_task_id)
    except (TypeError, ValueError):
        return []
    
    # 建立 int id 到任务的索引， 兼容字符串 id
    by_id: Dict[int, Dict[str, Any]] = {}
    for task in planned_tasks:
        try:
            by_id[int(task.get("id"))] = task
        except (TypeError, ValueError):
            continue

    current = by_id.get(current_id)
    previous = by_id.get(current_id - 1)
    if not current or not previous:
        return []
    
    if previous.get("status","pending") not in ACTIVE_TASK_STATUSES:
        return []
    
    previous_desc = str(previous.get("description",""))
    current_desc = str(current.get("description",""))

    # 校验/确认/检查类任务不能凭后续动作自动推断完成，否则会掩盖真实断言失败。
    if any(k in previous_desc for k in ["校验", "确认", "检查", "验证", "断言"]):
        return []
    
    pairs = [
        (["访问", "打开", "进入"], ["搜索", "输入", "点击", "查看"]),
        (["搜索"], ["点击第", "点击第2条", "点击第二条", "查看详情"]),
        (["点击第", "点击第2条", "点击第二条", "查看详情"], ["关闭", "关闭该标签页", "关闭标签页"]),
        (["打开详情", "查看详情"], ["关闭", "返回"]),
    ]

    def matches(desc: str, keywords: List[str]) -> bool:
        return any(k in desc for k in keywords)
    
    allowed = any(matches(previous_desc, left) and matches(current_desc, right) for left, right in pairs)

    if not allowed:
        return []
    
    previous["status"] = "completed"
    return [current_id - 1]


def mark_first_active_task(planned_tasks: List[Dict[str, Any]], task_status: str) -> Optional[Any]:
    """将第一个仍处于 pending/in_progress 的任务标记为指定状态。"""
    if not planned_tasks:
        return None
    
    normalized = str(task_status).strip().lower()
    for task in planned_tasks:
        if str(task.get("status", "pending")).lower() in ACTIVE_TASK_STATUSES:
            task["status"] = normalized
            return task.get("id")
    return None

def summarize_planned_tasks(planned_tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计不同状态的任务数量，返回状态分布摘要。"""
    summary = {"total": 0, "completed": 0, "failed": 0, "skipped": 0, "pending": 0, "in_progress": 0}
    if not planned_tasks:
        return summary
    summary["total"] = len(planned_tasks)
    for task in planned_tasks:
        status = task.get("status", "pending")
        if status in summary:
            summary[status] += 1
        else:
            # 未知状态按 pending 处理，避免误判整体通过。
            summary["pending"] += 1
    return summary


def resolve_execution_status(planned_tasks: List[Dict[str, Any]]) -> str:
    """根据所有子任务状态解析整体执行结果。"""
    summary = summarize_planned_tasks(planned_tasks)
    if summary["total"] == 0:
        return "passed"
    # 只要存在 failed/pending/in_progress，就不能算通过。
    if summary["failed"] > 0 or summary["pending"] > 0 or summary["in_progress"] > 0:
        return "failed"
    return "passed"


def is_infrastructure_failure(error_message: str) -> bool:
    """判断错误是否属于基础设施失败，而非业务步骤失败。

    基础设施失败包括 LLM 不可用、网络超时、API Key 错误、限流等。真实项目中这类错误
    通常不应直接归因到某个测试步骤，否则会误导用户以为业务页面有问题。
    """
    message = (error_message or "").lower()
    markers = [
        "execution llm unavailable", "connection error", "timed out", "timeout",
        "api key", "authentication", "unauthorized", "forbidden", "rate limit", "service unavailable",
    ]
    return any(marker in message for marker in markers)