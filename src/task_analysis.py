from __future__ import annotations

"""自然语言任务拆解模块。

真实 TestHub 的 BaseBrowserAgent 会让大模型把用户输入的自然语言任务拆成多个可执行
子任务，再把这些子任务同步到前端。这个学习版保留同样的核心思想，但默认不依赖网络：

1. 优先解析用户显式写出的编号步骤；
2. 如果没有编号步骤，则使用可注入的 llm_breakdown 或离线规则拆分；
3. 对拆分结果做二次归一化，去掉重复编号、空步骤和过碎步骤；
4. 输出统一的 planned_tasks: [{id, description, status}]。
"""


import re
from typing import Any, Callable, List, Optional


class  TaskAnalyzer:
    """自然语言任务拆分器。

    Args:
        llm_breakdown: 可选的任务拆分函数。真实接入 LLM 时，可以传入一个函数，
            接收 task_description 并返回字符串列表；不传时使用离线规则。
    """
    def __init__(self, llm_breakdown: Optional[Callable[[str], List[str]]] = None):
        self.llm_breakdown = llm_breakdown
    
    def extract_structured_steps(self, text:str) -> List[str]:
        """
        从文本中提取显示编号步骤
        支持格式示例：
        - 1. 打开首页
        - 2、输入账号
        - 1.1: 点击菜单
        - 单行内联：1. 打开首页 2. 输入账号
        """ 
        if not text:
            return []
        
        # 统一换行符，避免windows/Unix 换行差异影响解析
        normalized = str(text).replace("\r\n","\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        

        # 匹配以数字编号开头的行， 支持 1 / 1.1 / 1.2.3 等层级编号
        numbered_line_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)[\.\s、:：-]+(.*)$")

        # 用于保存成功识别出来的编号步骤
        extracted: List[str] = []

        # 没有编号的普通行先保留，作为没有结构化编号时的回退。
        plain_lines: List[str] = []

        for raw_line in normalized.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            
            match = numbered_line_pattern.match(line)
            if match:
                # group(1) 是编号，group(2) 是真实任务描述
                desc = match.group(2).strip()
                if desc:
                    extracted.append(desc)
            else:
                plain_lines.append(line)
        
        # 解析到编号步骤，优先返回
        if extracted:
            # 如果整段文本只有一个编号匹配，但原始文本没有换行，可能是“1. 打开首页 2. 输入账号 3. 点击登录”这种
            # 内联编号。此时先按“编号前空格”重新切行，再递归解析。
            # 只有一个元素且无换行
            if len(extracted) == 1 and "\n" not in text:
                split_inline = re.sub(r"\s+(?=\d+(?:\.\d+)*[\.\s、:：-]+)", "\n", normalized)
                inline_steps = self.extract_structured_steps(split_inline)
                if len(inline_steps) > 1:
                    return inline_steps
                
            return extracted
        
        # 没有按行解析到编号时，再尝试内联编号切分。
        split_inline = re.sub(r"\s+(?=\d+(?:\.\d+)*[\.\s、:：-]+)", "\n", normalized)
        if split_inline != normalized:
            inline_steps = self.extract_structured_steps(split_inline)
            if inline_steps:
                return inline_steps

        # 最后回退：返回非空普通行；如果连普通行也没有，返回完整文本。
        return plain_lines or [normalized]
    
    def normalize_steps(self, raw_steps: Any, fallback_text: str) -> List[str]:
        """把任意拆分结果整理成干净的字符串步骤列表。

        - 非 list 输入视为空；
        - None/空字符串会被过滤；
        - 如果某个步骤里仍包含编号子步骤，会继续拆开；
        - 每个步骤开头残留的“1.”、“2、”等编号会被去掉。
        """

        steps = raw_steps if isinstance(raw_steps, list) else []
        normalized: list[str] = []

        for step in steps:
            if step is None:
                continue
            desc = str(step).strip()
            if not desc:
                continue
            
            # 如果步骤里仍包含编号子步骤，继续拆开
            nested = self.extract_structured_steps(desc)
            # 如果拆开后不止一个步骤，且拆开后的步骤与原步骤不同，才使用拆开的结果；否则保留原步骤，避免过度拆分。
            if nested and not (len(nested) == 1 and nested[0] == desc):
                normalized.extend(nested)
            else:
                normalized.append(desc)

        if not normalized:
            normalized = self.extract_structured_steps(fallback_text)

        # 最后再去掉每个步骤开头残留的编号，例如“1. 打开首页” -> “打开首页”，支持多层编号。
        cleaned: List[str] = []
        for desc in normalized:
            current = str(desc).strip()
            # 循环去掉多层编号，例如“1. 2. 打开首页”
            # 1. 2. 打开首页 -> 2. 打开首页 -> 打开首页
            while True:
                match = re.match(r"^\s*\d+(?:\.\d+)*[\.\s、:：-]+(.*)", current, re.S)
                if not match:
                    break
                current = match.group(1).strip()
            if current:
                cleaned.append(current)
        return cleaned or [fallback_text.strip()]
    
    def compact_steps(self, steps: List[str]) -> List[str]:
        """合并过碎的步骤，减少浏览器 Agent 的执行抖动。

        LLM 或规则拆分有时会把“打开浏览器 / 输入 URL / 回车”拆成多个步骤。对
        browser-use 来说，这些可以合并成“访问 URL”。搜索场景也类似，可以合并成
        “搜索关键词”。
        """
        if not steps:
            return []
        compacted: List[str] = []
        i = 0

        while i < len(steps):
            current = str(steps[i]).strip()
            lower = current.lower()

            # 浏览器 + URL 相关连续步骤，合并为一个访问动作。
            if ("浏览器" in current or "browser" in lower or "地址栏" in current) and i + 1 < len(steps):
                window = " ".join(str(s).strip() for s in steps[i:i+3])
                

