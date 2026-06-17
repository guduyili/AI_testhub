from __future__ import annotations

"""真实 browser-use 浏览器执行示例。

安全起见，默认只检查配置和打印将要执行的任务，不打开浏览器、不调用 LLM：

    PYTHONPATH=src uv run --extra real python examples/real_browser_demo.py

确认 .env 后，显式传 --run 才会真正调用 browser-use：

    PYTHONPATH=src uv run --extra real python examples/real_browser_demo.py --run \
      --task "打开 https://example.com 并总结页面标题"
"""

import argparse

from dotenv import load_dotenv
load_dotenv(override=True)

from AI_testhub.real_browser_runner import RealBrowserRunner
from AI_testhub.real_config import load_real_model_config, parse_bool_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real browser-use task with the learning project's contracts.")
    parser.add_argument("--run", action="store_true", help="Actually invoke browser-use. Without this flag this script is a dry run.")
    parser.add_argument("--task", default="打开 https://example.com 并总结页面标题", help="Natural-language browser task.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max_steps passed to browser-use Agent.")
    args = parser.parse_args()

    load_dotenv()
    config = load_real_model_config()
    headless = parse_bool_env("BROWSER_HEADLESS", default=True)

    print(f"model: {config.model_name}")
    print(f"base_url: {config.base_url or '<sdk default>'}")
    print(f"headless: {headless}")
    print(f"task: {args.task}")

    if not args.run:
        print("browser_run: skipped; pass --run to open browser and invoke the model")
        return

    runner = RealBrowserRunner()
    result = runner.run_sync(
        task_description=args.task,
        config=config,
        headless=headless,
        use_real_llm=True,
        max_steps=args.max_steps,
    )
    print(f"status: {result.history.status}")
    print("events:")
    for event in result.events:
        print(f"  - {event['type']}: {event.get('message', '')}")
    print("logs:")
    for line in result.history.logs:
        print(f"  {line}")


if __name__ == "__main__":
    main()
