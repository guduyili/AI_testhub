from __future__ import annotations
"""
    BrowserAgent 主流程

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
"""


import time
from typing import Any, Callable, Dict, List, Optional

from .actions import (
    clean_task_urls,
    contains_auth_failure_signal,
    enforce_pending_status_settlement,
    enforce_single_task_step,
    normalize_action,
)
from .browser_profile import BrowserProfileConfig, create_browser_profile
from .config import AIModelConfig, InMemoryConfigStore
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
    
def build_chat_openai(config: AIModelConfig, use_real: bool= False) -> Any:
    """根据配置创建真实或离线的 ChatOpenAI-like 对象。

    Args:
        config: AIModelConfig，包含模型名、API Key、base_url、temperature 等。
        use_real: True 时尝试导入 langchain_openai.ChatOpenAI；False 时使用 FakeChatOpenAI。

    Returns:
        Any: 真实 ChatOpenAI 或 FakeChatOpenAI。

    Raises:
        RuntimeError: use_real=True 但未安装 langchain-openai 时抛出清晰错误。
    """

    if use_real:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError(
                "langchain-openai is not installed. Install with: uv sync --extra real") from exc

        # LangChain 的 ChatOpenAI 支持 OpenAI-compatible 接口；base_url 为空时传 None，
        # 让 SDK 使用默认地址。
        llm = ChatOpenAI(
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url or None,
            temperature=config.temperature,
        )

        # 给真实 ChatOpenAI 补充 provider/model 字段，方便日志和测试读取。
        # 某些 Pydantic 对象可能禁止动态赋值，因此失败时忽略，不影响模型调用。
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
        self.action_plan = [] or action_plan


    def _load_model_config(self) -> Optional[AIModelConfig]:
        """读取浏览器自动化文本模式的模型配置。

        真实系统应从数据库里取 role='browser_use_text' 且启用的配置；学习版没有配置时
        回退 fake 模型，让示例和测试可以无密钥运行。
        """
        config = self.config_store.get_active(role="browser_use_text")
        if config is None:
            return config
        return AIModelConfig(name="offline-fake")

    def verify_execution_result():
        return 

    def create_browser_profile():
        """创建 browser-use 浏览器配置。

        学习版固定按 Linux/WSL 环境生成配置；真实项目可根据部署环境传入不同 system 和
        chrome_path。
        """
        return create_browser_profile(system="Linux", chrome_path=None)




