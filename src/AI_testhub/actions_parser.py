from __future__ import annotations

"""LLM 动作输出解析与修复。

学习版不接真实 browser-use 时，仍要练习真实项目里最常见的痛点：
1. 模型把动作写成字符串函数调用；
2. action 被放在顶层字段而不是 action 数组里；
3. 某些业务动作参数是纯字符串，应该丢弃；
4. 状态动作的整数参数要被修复成 {task_id: N}。
"""

import ast
import re
from typing import Any, Dict, List, Optional

TASK_STATUS_ACTIONS = {"mark_task_complete", "mark_task_failed", "mark_task_skipped", "update_task_status"}
STRING_OK_ACTIONS = {"done", "switch_tab"}


def _parse_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def parse_string_action(text: str) -> List[Dict[str, Any]]:
    """解析字符串格式的动作声明。"""
    if not text:
        return []
    matched = re.fullmatch(r"\s*([a-zA-Z_][\w]*)\((.*)\)\s*", str(text), flags=re.S)
    if not matched:
        return []

    action_name = matched.group(1)
    args_text = matched.group(2).strip()
    if not args_text:
        return [{action_name: {}}]

    params: Dict[str, Any] = {}
    parts = [part.strip() for part in re.split(r",(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)", args_text) if part.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip()] = _parse_literal(value.strip())
    return [{action_name: params}]


def extract_external_status_actions(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把顶层状态动作补进 action 数组。"""
    if not isinstance(content, dict):
        return []

    actions: List[Dict[str, Any]] = []
    raw_actions = content.get("action", [])
    if isinstance(raw_actions, list):
        for action in raw_actions:
            if isinstance(action, dict):
                actions.append(action)
            elif isinstance(action, str):
                actions.extend(parse_string_action(action))

    for name, value in content.items():
        if name == "action":
            continue
        if name in TASK_STATUS_ACTIONS:
            if isinstance(value, int):
                value = {"task_id": value}
            elif not isinstance(value, dict):
                value = {"task_id": value}
            actions.append({name: value})
    return actions


def filter_invalid_string_params(action_name: str, action_params: Any) -> Any:
    """过滤不该作为纯字符串参数传入的动作。"""
    if action_name in STRING_OK_ACTIONS:
        return action_params
    if isinstance(action_params, str):
        return None
    return action_params


def _normalize_single_action(action: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(action, dict):
        return []

    repaired: List[Dict[str, Any]] = []
    for name, params in action.items():
        if isinstance(name, str) and name in TASK_STATUS_ACTIONS and isinstance(params, int):
            repaired.append({name: {"task_id": params}})
            continue
        filtered = filter_invalid_string_params(name, params)
        if filtered is None:
            continue
        repaired.append({name: filtered})
    return repaired


def repair_action_output(raw_actions: Any) -> List[Dict[str, Any]]:
    """把各种混合 LLM 输出统一修复成标准动作列表。"""
    if raw_actions is None:
        return []

    actions: List[Dict[str, Any]] = []
    sequence = raw_actions if isinstance(raw_actions, list) else [raw_actions]

    for item in sequence:
        if isinstance(item, str):
            actions.extend(parse_string_action(item))
            continue
        if not isinstance(item, dict):
            continue

        if "action" in item:
            actions.extend(extract_external_status_actions(item))
            continue

        actions.extend(_normalize_single_action(item))

    return actions
