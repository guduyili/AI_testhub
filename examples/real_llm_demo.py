from __future__ import annotations

"""真实 LLM 连通性检查示例。

默认只构造 ChatOpenAI，不发起网络请求：

    PYTHONPATH=src uv run --extra real python examples/real_llm_demo.py

确认环境变量后，如需真的调用模型：

    PYTHONPATH=src uv run --extra real python examples/real_llm_demo.py --run
"""

import argparse

from dotenv import load_dotenv
load_dotenv(override=True)

from AI_testhub.agent import build_chat_openai
from AI_testhub.real_config import load_real_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real LangChain ChatOpenAI initialization.")
    parser.add_argument("--run", action="store_true", help="Actually invoke the LLM. Without this flag only constructs the client.")
    args = parser.parse_args()

    load_dotenv()
    config = load_real_model_config()
    llm = build_chat_openai(config, use_real=True)
    print(f"constructed: {llm.__class__.__name__}")
    print(f"provider: {getattr(llm, 'provider', None)}")
    print(f"model: {getattr(llm, 'model', None)}")

    if not args.run:
        print("network_call: skipped; pass --run to invoke the model")
        return

    result = llm.ainvoke("Reply with OK only.")
    print(f"response: {getattr(result, 'content', result)}")


if __name__ == "__main__":
    main()
