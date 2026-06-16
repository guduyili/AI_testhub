import pytest

from AI_testhub.actions import (
    normalize_action_params,
    enforce_single_task_step,
    enforce_pending_status_settlement,
    contains_auth_failure_signal,
    clean_task_urls,
)
from AI_testhub.task_analysis import TaskAnalyzer
from AI_testhub.state import (
    update_planned_task_status,
    backfill_prior_pending_tasks,
    summarize_planned_tasks,
    is_infrastructure_failure,
)
from AI_testhub.agent import BrowserAgent, run_full_process_sync
from AI_testhub.config import AIModelConfig, InMemoryConfigStore


def test_normalize_action_params_converts_common_llm_aliases():
    assert normalize_action_params("click", {"element_id": 5}) == {"index": 5}

    assert normalize_action_params("input_text", {"content": "hello"}) == {"text": "hello"}
    assert normalize_action_params("switch_tab", "abc") == {"tab_id": "abc"}
    assert normalize_action_params("mark_task_complete", 2) == {"task_id": 2}


def test_enforce_single_task_step_drops_business_actions_after_terminal_status():
    actions = [
        {"click": {"index": 1}},
        {"mark_task_complete": {"task_id": 1}},
        {"input_text": {"index": 2, "text": "should drop"}},
    ]
    assert enforce_single_task_step(actions) == [
        {"click": {"index": 1}},
        {"mark_task_complete": {"task_id": 1}},
    ]


def test_enforce_pending_status_settlement_prevents_next_business_action_in_same_step():
    actions = [
        {"mark_task_complete": {"task_id": 1}},
        {"click": {"index": 9}},
    ]
    assert enforce_pending_status_settlement(actions, 1, "访问首页") == [
        {"mark_task_complete": {"task_id": 1}}
    ]


def test_auth_failure_detection_matches_chinese_and_english_signals():
    assert contains_auth_failure_signal("用户名或密码错误") is True
    assert contains_auth_failure_signal("invalid credentials") is True
    assert contains_auth_failure_signal("everything looks good") is False


def test_clean_task_urls_separates_chinese_punctuation_from_url():
    text = "访问 http://localhost:3000，然后登录"
    assert clean_task_urls(text) == "访问 http://localhost:3000 ，然后登录"


def test_task_analyzer_extracts_numbered_steps_without_llm():
    analyzer = TaskAnalyzer()
    tasks = analyzer.analyze_task("1. 访问首页\n2. 输入账号\n3. 点击登录")
    assert tasks == [
        {"id": 1, "description": "访问首页", "status": "pending"},
        {"id": 2, "description": "输入账号", "status": "pending"},
        {"id": 3, "description": "点击登录", "status": "pending"},
    ]


def test_task_status_update_and_summary():
    tasks = [
        {"id": 1, "description": "访问首页", "status": "pending"},
        {"id": 2, "description": "登录", "status": "pending"},
    ]
    assert update_planned_task_status(tasks, 1, "completed") is True
    assert summarize_planned_tasks(tasks) == {
        "total": 2,
        "completed": 1,
        "failed": 0,
        "skipped": 0,
        "pending": 1,
        "in_progress": 0,
    }


def test_backfill_prior_pending_tasks_only_for_safe_dependency_pairs():
    tasks = [
        {"id": 1, "description": "访问首页", "status": "pending"},
        {"id": 2, "description": "搜索订单", "status": "completed"},
    ]
    assert backfill_prior_pending_tasks(tasks, 2) == [1]
    assert tasks[0]["status"] == "completed"


def test_infrastructure_failure_detection_does_not_confuse_business_failure():
    assert is_infrastructure_failure("Execution LLM unavailable: timeout") is True
    assert is_infrastructure_failure("按钮不存在导致断言失败") is False


def test_browser_agent_full_process_uses_callbacks_and_scripted_actions():
    store = InMemoryConfigStore([
        AIModelConfig(name="demo", model_type="fake", model_name="fake-model", api_key="fake", base_url="http://fake", is_active=True)
    ])
    events = []
    agent = BrowserAgent(
        config_store=store,
        action_plan=[
            [{"go_to_url": {"url": "http://localhost:3000"}}, {"mark_task_complete": {"task_id": 1}}],
            [{"input_text": {"index": 1, "text": "admin"}}, {"mark_task_complete": {"task_id": 2}}],
        ],
    )
    history = agent.run_full_process(
        "1. 访问 http://localhost:3000\n2. 输入 admin",
        analysis_callback=lambda tasks: events.append(("analysis", tasks.copy())),
        step_callback=lambda event: events.append(("step", event)),
    )
    assert history.status == "passed"
    assert history.planned_tasks[0]["status"] == "completed"
    assert history.planned_tasks[1]["status"] == "completed"
    assert events[0][0] == "analysis"
    assert any(event[0] == "step" and event[1].get("type") == "log" for event in events)


def test_run_full_process_sync_factory_creates_agent_and_runs():
    history = run_full_process_sync(
        "1. 访问首页",
        config_store=InMemoryConfigStore([
            AIModelConfig(name="demo", model_type="fake", model_name="fake-model", api_key="fake", is_active=True)
        ]),
        action_plan=[[{"mark_task_complete": {"task_id": 1}}]],
    )
    assert history.status == "passed"
