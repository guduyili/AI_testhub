from __future__ import annotations

"""BrowserAgent 主流程。

这是从 TestHub 的 ai_base / ai_agent 中抽取出来的学习版核心文件。它不直接依赖
Django、真实数据库、真实 browser-use 浏览器实例，因此可以在离线环境中运行测试。

本文件重点复刻 10 条主链路能力：

1. 加载 AI 模型配置：_load_model_config；
2. 初始化 LangChain ChatOpenAI：build_chat_openai；
3. 适配 browser-use：BrowserProfileConfig、动作注册、action 参数归一化；
4. 拆解自然语言任务：TaskAnalyzer；
5. 创建浏览器配置：create_browser_profile；
6. 注册自定义浏览器动作：register_actions；
7. 强化执行提示词：build_execution_prompt；
8. 控制任务状态同步：_handle_status_action / run_task；
9. 捕获认证失败、未标记任务、基础设施失败等异常场景；
10. 执行完整流程：run_full_process / run_full_process_sync。

真实项目中的 browser-use Agent 会实际控制浏览器。本学习版用 action_plan 模拟模型每
一步输出的动作，使参数归一化、任务边界、状态同步等逻辑可以被单元测试覆盖。
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional

from .actions_parser import repair_action_output
from .actions import (
    clean_task_urls,
    contains_auth_failure_signal,
    enforce_pending_status_settlement,
    enforce_single_task_step,
    normalize_action,
)
from .browser_profile import BrowserProfileConfig, create_browser_profile
from .config import AIModelConfig, InMemoryConfigStore
from .events import EventBus, EventType
from .history import ExecutionHistory
from .state import (
    backfill_prior_pending_tasks,
    is_infrastructure_failure,
    mark_first_active_task,
    resolve_execution_status,
    update_planned_task_status,
)
from .task_analysis import TaskAnalyzer



class FakeChatOpenAI:
    """
    FakeChatOpenAI 保留 model / api_key / base_url / temperature 等关键字段，
    并提供 invoke 方法，从接口上模拟 LangChain ChatOpenAI。
    """

    def __init__(self, model: str, api_key: str, base_url: str="", temperature: float = 0.0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature

        self.provider = "fake"

    def invoke(self, prompt: str) -> str:
        """模拟模型调用，返回固定响应。"""
        print(f"Invoking FakeChatOpenAI with prompt:\n{prompt}\n")

        return "OK"
    

def _parse_bool_env(name: str, default: bool = False) -> bool:
    """解析布尔环境变量，兼容常见真值写法。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _build_structured_output_flags() -> Dict[str, bool]:
    """构造 browser-use ChatOpenAI 的结构化输出兼容开关。

    真实 OpenAI 端点支持 `response_format={'type': 'json_schema'}`；但很多 OpenAI-compatible
    代理（例如 OpenCode Zen / DeepSeek 代理）只支持普通 completion，不支持该格式。
    默认开启兼容模式：把 JSON schema 写进 system prompt，模型返回纯文本后再解析校验。
    如果你用的是官方 OpenAI 且想使用原生结构化输出，可设置
    `BROWSER_USE_STRICT_STRUCTURED_OUTPUT=1`。
    """
    strict = _parse_bool_env("BROWSER_USE_STRICT_STRUCTURED_OUTPUT", default=False)
    if strict:
        return {
            "dont_force_structured_output": False,
            "add_schema_to_system_prompt": False,
            "remove_min_items_from_schema": False,
        }
    return {
        "dont_force_structured_output": _parse_bool_env(
            "BROWSER_USE_DONT_FORCE_STRUCTURED_OUTPUT", default=True
        ),
        "add_schema_to_system_prompt": _parse_bool_env(
            "BROWSER_USE_ADD_SCHEMA_TO_SYSTEM_PROMPT", default=True
        ),
        "remove_min_items_from_schema": _parse_bool_env(
            "BROWSER_USE_REMOVE_MIN_ITEMS_FROM_SCHEMA", default=True
        ),
    }


def build_chat_openai(config: AIModelConfig, use_real: bool = False) -> Any:
    """根据配置创建真实或离线的 ChatOpenAI-like 对象。

    Args:
        config: AIModelConfig，包含模型名、API Key、base_url、temperature 等。
        use_real: True 时尝试导入 browser-use 原生的 ChatOpenAI；False 时使用 FakeChatOpenAI。

    Returns:
        Any: 真实 ChatOpenAI 或 FakeChatOpenAI。

    Raises:
        RuntimeError: use_real=True 但未安装 browser-use 时抛出清晰错误。
    """
    if use_real:
        try:
            from browser_use.llm.openai.chat import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("browser-use is not installed. Install with: uv sync --extra real") from exc

        # browser-use 自带的 ChatOpenAI 封装了 OpenAI-compatible 调用，并且提供兼容开关：
        # - dont_force_structured_output=True: 不强制 response_format=json_schema，模型以纯文本返回；
        # - add_schema_to_system_prompt=True: 把 JSON schema 附加到 system prompt；
        # - remove_min_items_from_schema=True: 移除 schema 中的 minItems，避免部分模型报错。
        # 这些开关对 OpenCode Zen / DeepSeek 等代理至关重要，否则会得到
        # "This response_format type is unavailable now" 或 "items" 类解析失败。
        compat_flags = _build_structured_output_flags()
        llm = ChatOpenAI(
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url or None,
            temperature=config.temperature,
            **compat_flags,
        )

        # 给真实 ChatOpenAI 补充 provider/model 字段，方便日志和测试读取。
        # dataclass 可能禁止直接赋值，因此失败时忽略，不影响模型调用
        try:
            object.__setattr__(llm, "provider", config.model_type)
            object.__setattr__(llm, "model", config.model_name)
        except AttributeError:
            pass

        return llm

    # 默认路径：离线 fake LLM，保证项目开箱即可运行。
    llm = FakeChatOpenAI(config.model_name, config.api_key, config.base_url, config.temperature)
    llm.provider = config.model_type
    return llm
    

class SafeRecordSaver:
    """数据库保存重试逻辑的学习版替身。

    真实 TestHub 中，执行结果、日志、任务状态会写入数据库。MySQL 连接偶尔会出现
    “server has gone away”等瞬时错误，因此需要重试。学习版不引入数据库，只让调用方
    注入 save_func，从而可以在测试里模拟失败和重试。
    """

    def __init__(self, save_func: Callable[[Dict[str, Any]], None] = None):
        # 默认 save_func 什么都不做，便于离线运行。
        self.save_func = save_func or (lambda record: None)

    def save(self, record: Dict[str, Any], max_retries: int = 3) -> bool:
       """保存记录，遇到 MySQL 连接丢失类错误时进行有限重试。"""
       for attempt in range(max_retries):
           try:
               self.save_func(record)
               return True
           
           except Exception as e:
               message = str(e).lower()
               # MySQLdb / PyMySQL 常见断连信号：2006,MySQL server has gone away,0
               is_mysql_gone = "2006" in message or "MySQL server has gone away" in message or message == "0"
               if is_mysql_gone and attempt < max_retries -1:
                   # 学习版只 sleep 0.01 秒，避免测试变慢；真实项目可使用更长退避时间。
                    time.sleep(0.01)
                    continue
               raise
       return False
    
class BrowserAgent:
    """TestHub BaseBrowserAgent/BrowserAgent 的离线学习版。

    这个类把“AI 浏览器自动化”的关键环节串起来：模型配置、LLM 初始化、任务拆解、
    浏览器配置、自定义动作、提示词强化、执行状态同步和异常处理。

    Args:
        execution_mode: 执行模式。学习版固定使用 text，仅保留参数形态。
        enable_gif: 是否启用 GIF 录制。学习版不录制，只保留配置项。
        case_name: 用例名称，用于日志/展示。
        config_store: 模型配置仓库；不传则使用空仓库并回退到 fake 模型。
        action_plan: 模拟 browser-use/LLM 每一步输出的动作列表。形如：
            [[{"click": {"index": 1}}, {"mark_task_complete": {"task_id": 1}}], ...]
        use_real_llm: 是否真的创建 LangChain ChatOpenAI。
    """
    def __init__(
            self,
            execution_mode: str = "text",
            enable_gif: bool = False,
            case_name: str = "default_case",
            config_store: Optional[InMemoryConfigStore] = None,
            action_plan: Optional[List[List[Dict[str, Any]]]] = None,
            use_real_llm: bool = False,
    ):
        self.execution_mode = execution_mode
        self.enable_gif = enable_gif
        self.case_name = case_name or "Adboc Task"

        # 1. 加载AI模型配置；如果没有启用的配置，后续会回退到 fake 模型。
        self.config_store = config_store or InMemoryConfigStore()
        self.config = self._load_model_config()

        # 2.初始化 ChatOpenAI-like LLM
        self.llm = build_chat_openai(self.config, use_real=use_real_llm)

        # 3/4/5 初始化任务拆解器和浏览器配置
        self.task_analyzer = TaskAnalyzer()
        self.browser_profile = create_browser_profile()

        # action_plan 模拟 brower-use/LLM 每一步输出的动作列表，便于测试和演示；真实系统则由 LLM 实时生成。
        self.action_plan = [] or action_plan or []

        # pending_status_* 用来记录“上一步有业务动作但模型忘记标状态”的任务，下一步补偿。
        self.pending_status_task_id: Optional[int] = None
        self.pending_status_task_description: Optional[str] = None

        #  认证失败计数，同一任务连续失败达到阈值后，立即标记失败并停止
        self.auth_failure_task_id: Optional[int] = None
        self.auth_failure_count = 0

        # 6.注册自定义浏览器动作
        self.registered_actions = self.register_actions()



    def _load_model_config(self) -> Optional[AIModelConfig]:
        """读取浏览器自动化文本模式的模型配置。

        真实系统应从数据库里取 role='browser_use_text' 且启用的配置；学习版没有配置时
        回退 fake 模型，让示例和测试可以无密钥运行。
        """
        config = self.config_store.get_active(role="browser_use_text")
        if config:
            return config
        return AIModelConfig(name="offline-fake", model_type="fake", model_name="fake-model", api_key="fake")

    def verify_execution_result(self) -> None:
        """执行前检查 LLM 是否可用。

        这对应“模型连通性预检查”。如果模型不可用，应在真正启动浏览器流程前失败，
        避免用户等待很久才发现 API Key、网络或服务地址有问题。
        """
        try:
            result = self.llm.invoke("Reply with OK") if hasattr(self.llm, "invoke") else "OK"
            if result != "OK":
                raise RuntimeError("empty response")
        except Exception as e:
            raise RuntimeError(f"LLM connectivity check failed: {e}") from e

    def create_browser_profile() -> BrowserProfileConfig:
        """创建 browser-use 浏览器配置。

        学习版固定按 Linux/WSL 环境生成配置；真实项目可根据部署环境传入不同 system 和
        chrome_path。
        """
        return create_browser_profile(system="Linux", chrome_path=None)


    def register_actions(self) -> Dict[str, Callable[..., Dict[str, Any]]]:
        """注册自定义动作。

        真实 browser-use Controller 会把这些函数注册给 Agent，LLM 可以在动作列表中调用
        mark_task_complete / mark_task_failed 等动作。学习版只返回函数映射，便于理解每个
        自定义动作的输入输出格式。
        """

        def mark(task_id: int, status: str) -> Dict[str, Any]:
            """标记任务状态的通用函数，适用于 mark_task_complete / mark_task_failed / mark_task_skipped 等动作。"""
            return {"task_id": task_id, "status": status}


        return {
            # Done 表示整个浏览器任务结束。
            "Done": lambda success=True, text="": {"done": bool(success), "text": str(text)},
            # close_tab 用于复刻"新标签页打开后关闭/切换"的动作能力
            "close_tab": lambda: {"closed": True},
            # 显示任务状态动作，前端抓鬼太同步依赖这些动作
            "mark_task_complete": lambda task_id: mark(task_id, "completed"),
            "mark_task_failed": lambda task_id: mark(task_id, "failed"),
            "mark_task_skipped": lambda task_id: mark(task_id, "skipped"),
            "update_task_status": lambda task_id, status: mark(task_id, str(status).lower()),
        }

    def analyze_task(self, task_description:str) -> List[Dict[str, Any]]:
        """拆解自然语言任务，返回 planned_tasks。"""
        return self.task_analyzer.analyze_task(task_description)
    
    def build_execution_prompt(self, task_description:str, planned_tasks:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构造强化后的执行提示词。

        提示词的目的不是让模型“更会聊天”，而是明确自动化执行约束：
        - 必须按 planned_tasks 顺序执行；
        - 每个子任务完成/失败/跳过后必须立即调用状态动作；
        - 标记任务 N 后不能在同一步开始任务 N+1；
        - 不得编造登录凭据；
        - 新标签页打开后要切换到最新标签页；
        - 使用 browser-use 原生字段 index/text/tab_id。
        """
        # 先清理URL，避免中文标点粘在 URL 后导致浏览器访问错误地址
        prompt = clean_task_urls(task_description)
        if planned_tasks:
            prompt += "\n\n重要指令: \n"
            prompt += "你有一个子任务列表，必须严格按顺序执行.\n"
            prompt += (
                "每当你完成一个子任务，就必须立即调用对应的状态动作：\n"
                "- mark_task_complete(task_id)\n"
                "- mark_task_failed(task_id)\n"
                "- mark_task_skipped(task_id)\n"
                "如果你标记了某个子任务完成/失败/跳过，你不能在同一步开始下一个子任务了，必须等到下一步。\n"
            )
            prompt += "子任务: \n"
            prompt += "\n".join(f"{task['id']}, {task['description']}" for task in planned_tasks)
        
        prompt += "\n\n关键性能与同步规则: \n"
        prompt += "1. 仅在当前任务完成后标记该任务.\n"
        prompt += "2. 如果你标记了任务 N, 则停止当前步骤: 不要开始任务 N+1.\n"
        prompt += "3. 不要编造登录凭据，多次认证失败后停止"
        prompt += "4. 如果链接打开了新标签页, 切换到最新的标签页"
        prompt += "5. 使用原生动作参数： index/text/tab_id，不要使用 element_id/content/tab 等别名\n"
        return prompt
    
    def _emit(self, callback: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]) -> None:
        """向外部回调发送结构化事件。

        真实项目可能通过 SSE/WebSocket 推送；学习版用 EventBus 把内部 payload
        统一转换为 JSON 友好的事件 dict，再交给 callback。
        """
        if not callback:
            return

        bus = EventBus(callback)
        if payload.get("type") == "log":
            content = str(payload.get("content", ""))
            bus.emit(EventType.STEP_LOG, message=content, data={"content": content})
            return

        task_id = payload.get("task_id")
        status = str(payload.get("status", "")).lower()
        status_event_types = {
            "completed": EventType.TASK_COMPLETE,
            "failed": EventType.TASK_FAILED,
            "skipped": EventType.TASK_SKIPPED,
        }
        if task_id is not None and status in status_event_types:
            bus.emit(
                status_event_types[status],
                task_id=int(task_id),
                message=f"task {task_id} {status}",
                data={"status": status},
            )
            return

        callback(payload)

    def _default_action_plan(self, planned_tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """没有传入 action_plan 时的离线默认动作计划。

        它不会控制真实浏览器，只会逐个标记任务完成，确保示例可以跑通完整状态流。
        """
        return [[{"mark_task_complete": {"task_id": task["id"]}}] for task in planned_tasks]
        
        


    def _handle_status_action(self, action: Dict[str, Any], planned_tasks: List[Dict[str, Any]], callback) -> bool:
        """处理任务状态动作，并同步 planned_tasks 与外部回调。

        Returns:
            bool: True 表示 action 是状态动作；False 表示它是普通业务动作。
        """

        for name, params in action.items():
            if name == "mark_task_complete":
                task_id = params.get("task_id")
                # 当前任务完成时，尝试安全补标前一个强依赖 pending 任务
                backfill_prior_pending_tasks(planned_tasks, task_id)
                update_planned_task_status(planned_tasks, task_id, "completed")
                self._emit(callback, {"task_id": int(task_id), "status": "completed"})
                return True
            
            if name == "mark_task_failed":
                task_id = params.get("task_id")
                update_planned_task_status(planned_tasks, task_id, "failed")
                self._emit(callback, {"task_id": int(task_id), "status": "failed"})
                return True

            if name == "mark_task_skipped": 
                task_id = params.get("task_id")
                update_planned_task_status(planned_tasks, task_id, "skipped")
                self._emit(callback, {"task_id": int(task_id), "status": "skipped"})
                return True

            if name == "update_task_status":
                task_id = params.get("task_id")
                status = str(params.get("status", "")).lower()
                update_planned_task_status(planned_tasks, task_id, status)
                self._emit(callback, {"task_id": int(task_id), "status": status})
                return True
            return False

    def run_task(
            self,
            task_description: str,
            callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            planned_tasks: Optional[List[Dict[str, Any]]] = None,
            should_step: Optional[Callable[[], bool]] = None,
            ) -> ExecutionHistory:
        """执行已拆解的任务计划。

        Args:
            task_description: 原始自然语言任务，用于构造提示词和 URL 清理。
            callback: 每步日志/状态变化的回调。
            planned_tasks: 已拆解任务列表；不传则视为空计划。
            should_stop: 停止信号函数；返回 True 时中断执行。

        Returns:
            ExecutionHistory: 包含最终状态、任务状态、日志和步骤动作。
        """
        # breakpoint()
        # 先做一些执行前检查，确保模型可用，避免用户等待很久才发现问题。
        self.verify_execution_result()

        planned_tasks = planned_tasks or []

        # 构造执行提示词；真实系统会在每一步循环里根据最新状态动态构造，这里简化为一次性构造。
        self.build_execution_prompt(task_description, planned_tasks)

        # 如果外部没有提供模拟动作计划，就使用默认“逐个完成”的离线动作计划。
        plan = self.action_plan or self._default_action_plan(planned_tasks)
        history = ExecutionHistory(status="running", planned_tasks=planned_tasks)

        try:
            for step_index,raw_actions in enumerate(plan, start=1):
                # 支持用户 / 前端主动停止任务
                if should_step and  should_step():
                    history.status  = "stopped"
                    history.logs.append("[System] user requested stop")
                    EventBus(callback).emit(EventType.PROCESS_STOPPED, message="user requested stop", data={"status": "stopped"})
                    break

            # 1) 修复 LLM/browser-use 常见混合输出形态：字符串函数调用、顶层状态字段、非法字符串参数。
            repaired_actions = repair_action_output(raw_actions)

            # 2) 归一化参数：element_id -> index、content -> text、int -> task_id/index。
            actions = [normalize_action(action) for action in repaired_actions]

            # 3) 如果上一步有业务动作但没标状态，这一步只允许先结算 pending 任务。
            actions = enforce_pending_status_settlement(
                actions,
                self.pending_status_task_id,
                self.pending_status_task_description,
            )

            # 3) 单任务边界：标记终态后，截断后续业务动作。
            actions = enforce_single_task_step(actions)


            # 记录步骤日志，方便复盘模型到底输出了什么动作
            history.steps.append({"step": step_index, "actions": actions})
            log_line = f"[Step {step_index}] Actions: {actions}"
            history.logs.append(log_line)
            self._emit(callback, {"type": "log", "content": log_line})

            has_status = False
            has_business = False
            for action in actions:
                # 状态动作会更新 planned_tasks；非状态动作视为业务动作。
                if self._handle_status_action(action, planned_tasks, callback):
                   has_status = True
                else:
                    has_business = True


                # breakpoint()
                # 登录失败检测：如果动作/观察文本中连续出现认证失败信号，则停止当前任务。
                if contains_auth_failure_signal(str(action)):
                    active = next((t for t in planned_tasks if t.get("status") in {"pending","in_progress"}), None)
                    if active:
                        if self.auth_failure_task_id == active["id"]:
                            self.auth_failure_count += 1
                        else:
                            self.auth_failure_task_id = active["id"]
                            self.auth_failure_count = 1
                    if self.auth_failure_count >= 3:
                        update_planned_task_status(planned_tasks, active["id"], "failed")
                        history.logs.append(f"[System] repeated auth failure; task {active['id']} failed")
                        history.status = "failed"
                        EventBus(callback).emit(
                            EventType.PROCESS_FAILED,
                            message="repeated auth failure",
                            data={"status": "failed", "task_id": active["id"]},
                        )
                        return history
                    
            # 如果这一轮有真实业务动作但没有任何状态动作，说明模型忘记显式标记任务状态。
            # 记录 pending，下一轮优先要求它补标，避免任务状态长期停留 pending。
            if has_business and not has_status:
                active = next((t for t in planned_tasks if t.get("status") in {"pending", "in_progress"}), None)
                if active:
                    self.pending_status_task_id = active["id"]
                    self.pending_status_task_description = active.get("description")

            # 所有步骤执行完后，根据 planned_tasks 汇总整体结果。
            if history.status == "running":
                history.status = resolve_execution_status(planned_tasks)
            
            if history.status == "passed":
                EventBus(callback).emit(EventType.PROCESS_COMPLETE, message="process complete", data={"status": history.status})
            elif history.status == "failed":
                EventBus(callback).emit(EventType.PROCESS_FAILED, message="process failed", data={"status": history.status})
            elif history.status == "stopped":
                EventBus(callback).emit(EventType.PROCESS_STOPPED, message="user requested stop", data={"status": history.status})
            return history
        except Exception as exc:
            # 基础设施失败（LLM/API/网络）不强行归因到业务子任务；业务异常则标记当前任务失败。
            if not is_infrastructure_failure(str(exc)):
                mark_first_active_task(planned_tasks, "failed")
            history.status = "failed"
            history.logs.append(f"error: {exc}")
            EventBus(callback).emit(EventType.PROCESS_FAILED, message="process failed", data={"status": "failed", "error": str(exc)})
            return history
        
                
    def run_full_process(
        self,
        task_description: str,
        analysis_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> ExecutionHistory:
        """执行完整链路：任务拆解 -> 回调同步 -> 执行任务。"""
        planned_tasks = self.analyze_task(task_description)
        if analysis_callback:
            analysis_callback(planned_tasks)
        elif step_callback:
            EventBus(step_callback).emit(
                EventType.TASK_ANALYSIS,
                message="task analysis complete",
                data={"tasks": planned_tasks},
            )
        # breakpoint()               
        return self.run_task(task_description, step_callback,planned_tasks, should_stop)




def get_agent_class(execution_mode: str) -> Any:
    """根据执行模式返回 Agent 类。

    真实项目可能根据 text/vision 等模式选择不同 Agent；学习版统一返回 BrowserAgent。
    """
    return BrowserAgent

def run_ai_task_sync(
    task_description: str,
    planned_tasks: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    execution_mode: str = "text",
    **kwargs,
) -> ExecutionHistory:
    """同步执行入口：创建 Agent 并执行已拆解任务。"""
    agent = BrowserAgent(execution_mode=execution_mode, **kwargs)
    return agent.run_task(task_description, callback, planned_tasks, should_stop)

def analyze_task_sync(task_description: str, execution_mode: str = "text", **kwargs) -> List[Dict[str, Any]]:
    """同步任务拆解入口：创建 Agent 并返回 planned_tasks。"""
    agent = BrowserAgent(execution_mode="text", **kwargs)
    return agent.analyze_task(task_description)



def run_full_process_sync(
    task_description: str,
    analysis_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    execution_mode: str = "text",
    **kwargs,
) -> ExecutionHistory:
    agent = BrowserAgent(execution_mode="text", **kwargs)
    return agent.run_full_process(task_description, analysis_callback, step_callback, should_stop)