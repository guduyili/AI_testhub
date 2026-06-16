from AI_testhub import AIModelConfig, BrowserAgent, InMemoryConfigStore

def main():
    """运行一个最小的端到端示例。"""
    store = InMemoryConfigStore([
        AIModelConfig(
            name="demo",
            model_type="fake",
            model_name="fake-model",
            api_key="fake-key",
            base_url="http://fake.local",
            is_active=True,
        )
    ])

    # 收集回调事件，模拟前端通过 SSE/WebSocket 接收任务拆解和执行日志。
    events = []

    # enable_gif=False 表示不录制 GIF；学习版本身也不会启动真实浏览器。
    agent = BrowserAgent(config_store=store, enable_gif=False)


    # run_full_process 会先 analyze_task，再 run_task。

    history = agent.run_full_process(
        "1. 访问 http://localhost:3000，打开登录页\n2. 输入用户名 admin\n3. 点击登录并确认进入首页",
        analysis_callback=lambda tasks: events.append({"type": "analysis", "tasks": tasks}),
        step_callback=lambda event: events.append({"type": "step", "event": event}),
    )


    # 打印最终状态，便于确认离线流程可运行。
    print("status:", history.status)
    print("planned_tasks:")
    for task in history.planned_tasks:
        print(f"  {task['id']}. [{task['status']}] {task['description']}")
    print("logs:")
    for log in history.logs:
        print(" ", log)

if __name__ == "__main__":
    main()

