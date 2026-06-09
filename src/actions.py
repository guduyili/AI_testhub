from __future__ import annotations

"""动作参数归一化与动作安全约束。

browser-use 的 Agent 每一步会输出一个或多个 action，例如 click、input_text、
mark_task_complete。真实 LLM 经常会生成“意思对但字段名不标准”的参数：

- element_id / node_id / id 表示点击元素下标；
- content / value 表示输入文本；
- tab / target 表示标签页 id；
- mark_task_complete 可能直接传整数而不是 {"task_id": 1}。

这个模块集中处理这些不稳定输出，并增加任务边界保护：一旦某个子任务被标记完成，
同一步里后续业务动作会被截断，避免模型“完成任务 1 后立刻开始任务 2”，导致前端
状态同步和真实浏览器状态错位。
"""

import json
import re
from typing import Any, Dict, List, Optional

# 显式任务终态动作。真实项目中这些动作通常注册为 browser-use Controller 的自定义动作。
TASK_STATUS_ACTIONS = {"mark_task_complete", "mark_task_failed", "mark_task_skipped"}

# 子任务的终态集合；update_task_status 只有落到这些状态时才算“完成当前任务边界”。
TERMINAL_STATUSES = {"completed", "failed", "skipped"}


def normalize_action_params(action_name: str, action_params: Any) -> Any:
    """将 LLM 生成的常见参数别名转换成 browser-use 风格参数。

    Args:
        action_name: 动作名称，例如 click、input_text、switch_tab。
        action_params: 原始动作参数，可能是 int、str、dict 或其他对象。
    Returns:
        归一化后的参数。无法识别时保持原值，避免误改模型输出。
    """
    # 许多模型会把 click: 3 写成“点击第 3 个元素”。对普通动作来说 int 应解释为 index；
    # 但对任务状态动作来说 int 应解释为 task_id。
    if isinstance(action_params, int):
        if action_name in TASK_STATUS_ACTIONS:
            return {"task_id": action_params}
        return {"index": action_params}

    # switch_tab 有时只给一个字符串 tab id，这里补成标准结构。
    if action_name == "switch_tab" and isinstance(action_params, str):
        return {"tab_id": action_params}

    # 非 dict 参数没有 key 可归一化，直接返回。
    if not isinstance(action_params, dict):
        return action_params
    
    normalized: Dict[str, Any] = {}
    for key, value in action_params.items():
        normalized_key = key

        # 元素定位字段统一成 index；但任务状态动作中的 id 不能误改成 index。
        if key in {"element_index", "element_id", "node_id", "id"} and action_name not in TASK_STATUS_ACTIONS:
            normalized_key = "index"    
        # 标签页 id 字段统一成 tab_id。
        elif key in {"tab", "target","target_id"} and action_name in {"switch_tab", "switch"}:
            normalized_key = "tab_id"
        # 输入文本字段统一成 text。
        elif key in {"content", "value"} and action_name in {"input_text", "type"}:
            normalized_key = "text"
    return normalized

def normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """归一化一个 action 字典中的所有动作参数。"""
    return {name: normalize_action_params(name, params) for name, params in action.items()}


def is_terminal_task_action(action_name: str, action_params: Any) -> bool:
    """判断一个动作是否是标记任务完成/失败/跳过的终态动作。"""
    if action_name in TASK_STATUS_ACTIONS:
        return True
    
    if action_name == "update_task_status" and isinstance(action_params, dict):
        # 任务状态动作参数可能是 {"task_id": 1} 或直接是整数 1。
        return str(action_params.get("status", "")).strip().lower() in TERMINAL_STATUSES
    
    # brower-use 的done动作也表示任务完成，但它没有明确的 status 字段，而是直接叫 "done"。
    if action_name == "done":
        return True
    

def enforce_single_task_step(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """保证单个 step 不跨越多个子任务边界。

    规则：一旦出现 mark_task_complete / mark_task_failed / mark_task_skipped / done 等
    终态动作，后面的业务动作全部丢弃。这样前端收到“任务 N 完成”后，不会同时看到
    模型已经偷偷开始任务 N+1。
    """
    if not isinstance(actions, list):
        return actions

    # 这个函数假设输入的 actions 已经是模型输出的原始动作列表，每个动作可能包含多个子动作。
    trimmed: List[Dict[str, Any]] = []
    terminal_seen =False

    for action in actions:
        if terminal_seen:
            # 已经看到终态动作，后续动作不再添加。
            continue
        
        trimmed.append(action)
        if isinstance(action, dict):
            for name, params in action.items():
                if isinstance(action, str):
                    


    
    

