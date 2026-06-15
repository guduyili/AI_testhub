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
    # breakpoint()
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
        normalized[normalized_key] = value
        # breakpoint()
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
                if is_terminal_task_action(name, params):
                    terminal_seen = True
                    break

    return trimmed


def get_task_status_action_task_id(action: Dict[str, Any]) -> Optional[int]: 
    """ 从任务状态动作种提取 task_id; 不是任务状态动作则返回 None"""
    if not isinstance(action, dict):
        return None
    for name, params in action.items():
        if name in TASK_STATUS_ACTIONS and isinstance(params, dict):
            return params.get("task_id")
        if name == "update_task_status" and isinstance(params, dict):
            if str(params.get("status", "")).strip().lower() in TERMINAL_STATUSES:
                return params.get("task_id")
    return None

def has_real_business_action(action: Dict[str, Any]) -> bool:
    """判断 action 是否是真正会操作浏览器/页面的业务动作。"""
    if not isinstance(action, dict):
        return False
    # 只要动作里有一个子动作不是任务状态相关的，就认为它包含真实业务动作。
    return any(name not in TASK_STATUS_ACTIONS | {"update_task_status", "done"} for name in action.keys())



def extract_task_literals(task_description: str) -> List[str]:
    """提取任务描述中的关键字面量，用于判断业务动作是否仍属于当前待结算任务。

    例如任务描述里有 URL、中文书名号、单双引号内容，这些通常会出现在真实 action
    参数里。若模型在同一步中先标记任务完成，又继续输入同一个 URL/关键词，则可认为
    该业务动作仍是当前任务的一部分，不必强行截断。
    """
    if not task_description:
        return []
    
    text = str(task_description)
    # 提取 URL
    literals : List[str] = []
    literals.extend(re.findall(r"「([^」]+)」", text))
    literals.extend(re.findall(r'"([^"\n]+)"', text))
    literals.extend(re.findall(r"'([^'\n]+)'", text))
    literals.extend(re.findall(r"https?://[^\s]+", text))

    # 去重并过滤空字符串，保持原始顺序，方便日志排查。
    deduped: List[str] = []
    for item in literals:
        cleaned = item.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def action_matches_pending_task(action: Dict[str, Any], pending_task_description: str) -> bool:
    """判断业务 action 是否明显还在处理上一个待结算任务。"""
    literals = extract_task_literals(pending_task_description)
    if not literals:
        return False
    payload = json.dumps(action, ensure_ascii=False)
    return any(literal in payload for literal in literals)




def enforce_pending_status_settlement(
    actions: List[Dict[str, Any]],
    pending_task_id: Any,
    pending_task_description: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """处理“上一步做了业务动作但没标状态”的补偿边界。

    场景：模型在 step 1 点击了按钮，但忘记 mark_task_complete。系统会记录
    pending_task_id。若 step 2 同时出现“补标 step 1 完成”和“开始 step 2 的业务动作”，
    这里会只保留补标动作，强制下一轮再开始 step 2。

    例外：如果业务动作仍包含当前 pending 任务的 URL/关键词，则认为它仍属于当前任务，
    不做截断。
    """
    if not pending_task_id or not isinstance(actions, list):
        return actions

    # 检查本轮动作里是否有标记 pending_task_id 完成的动作。
    marked_pending = any(str(get_task_status_action_task_id(a)) == str(pending_task_id) for a in actions)
    real_actions = [a for a in actions if has_real_business_action(a)]
    if not marked_pending or not real_actions:
        return actions
    # 如果有标记 pending_task_id 的动作，同时又有真实业务动作，则可能出现了“忘记上一步 mark_task_complete”的情况。
    # 这时如果业务动作里明显还在处理 pending_task_id 对应的任务（例如包含 pending_task_description 里的 URL/关键词），则认为它仍属于当前任务，不强行截断。
    if pending_task_description and any(action_matches_pending_task(a, pending_task_description) for a in real_actions):
        return actions
    
    # 否则强制只保留标记 pending_task_id 的动作，丢弃其他业务动作，等待下一轮模型输出真正的后续动作。
    return [a for a in actions if str(get_task_status_action_task_id(a)) == str(pending_task_id)]



def contains_auth_failure_signal(text: str) -> bool:
    """检测登录/认证失败信号。

    真实浏览器自动化中，登录失败可能出现在页面文本、模型观察摘要或 action 日志中。
    这里用中英文关键字模拟检测逻辑，连续失败后由 BrowserAgent 标记任务失败。
    """
    if not text:
        return False
    normalized = str(text).lower()
    keywords = [
        "登录失败", "login failed", "invalid credentials", "incorrect password",
        "用户名或密码", "账号或密码", "authentication failed", "auth failed",
        "bad credentials", "unauthorized", "401", "403",
    ]
    return any(keyword in normalized for keyword in keywords)


def clean_task_urls(text: str) -> str:
    """清理 URL 结尾中文标点导致的解析歧义。

    例如“访问 http://localhost:3000，然后登录”中，逗号紧贴 URL 时，部分解析器会把
    中文逗号也视为 URL 的一部分。这里在 URL 和中文标点之间插入空格。
    """
    return re.sub(r"(https?://[^\s\u4e00-\u9fa5]+?)(?=[，；。、！])", r"\1 ", text)
                    


    
    

