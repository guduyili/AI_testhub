from __future__ import annotations

"""浏览器配置构造器。

真实 TestHub 会把这些配置传给 browser-use / Playwright，让大模型可以控制浏览器。
本学习项目不启动真实浏览器，只保留可序列化的 BrowserProfileConfig，重点学习：

- Linux/WSL/headless 环境为什么需要额外启动参数；
- 如何集中创建浏览器配置，而不是在执行流程中散落硬编码；
- 如何把真实 browser-use 的 BrowserProfile 概念简化成可测试的数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BrowserProfileConfig:
    """browser-use BrowserProfile 的轻量替身。

    这里保留最重要的参数：
    - headless: 是否无头运行；Linux/CI/WSL 环境通常为 True。
    - executable_path: Chrome/Chromium 可执行文件路径；None 表示使用默认发现逻辑。
    - args: 传给浏览器进程的启动参数。
    - wait_*: 页面加载和动作间隔等待时间，用于减少自动化抖动。
    """

    headless: bool
    executable_path: Optional[str] = None
    args: List[str] = field(default_factory=list)
    wait_for_network_idle_page_load_time: float = 0.2


def create_browser_profile(system: str = "Linux", chrome_path: Optional[str] = None) -> BrowserProfileConfig:
    """创建浏览器运行配置。

    Args:
        system: 操作系统名称。学习版只区分 Linux 和非 Linux。
        chrome_path: 可选的 Chrome/Chromium 路径。
    Returns:
        BrowserProfileConfig: 后续可传给真实 browser-use BrowserProfile 的配置草稿
    """
    extra_args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--disable-notifications",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "--disable-extensions",
        "--disable-web-security",
    ]

    if system == "Linux":
        extra_args.extend([
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--headless=new",
            "--disable-software-rasterizer",
            "--remote-debugging-port=9222",
            "--remote-debugging-address=0.0.0.0",
            "--no-zygote",
            "--single-process",
        ])
    else:
        # 非 Linux 环境的默认配置，通常适合有桌面的Windows/macOS。
        extra_args.extend(["--no-sandbox", "--disable-gpu", "--remote-debugging-port=9222"])

    return BrowserProfileConfig(
                                headless=(system == "Linux"),
                                executable_path=chrome_path,
                                args=extra_args)
