from __future__ import annotations

from typing import Any

from agents.react_agent import ReActAgent
from models.config import Config


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(messages)
        if not self._responses:
            return ""
        return self._responses.pop(0)


class FakeToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_tools_description(self) -> str:
        return "- Search: search docs\n- Lookup: lookup facts"

    def execute_tool(self, name: str, input_text: str) -> str:
        self.calls.append((name, input_text))
        return f"ok:{name}:{input_text}"

    def register_tool(self, tool: Any) -> None:
        return None


def test_run_executes_multiple_actions_then_finish() -> None:
    llm = FakeLLM(
        [
            "Thought: 先查两次\nAction:\n- Search[python]\n- Lookup[pydantic]",
            "Thought: 信息足够\nAction: Finish[测试结论]",
        ]
    )
    registry = FakeToolRegistry()
    agent = ReActAgent(
        name="react",
        llm=llm,
        tool_registry=registry,
        config=Config(),
        max_steps=3,
    )

    result = agent.run("什么是默认工厂")

    assert result == "测试结论"
    assert registry.calls == [("Search", "python"), ("Lookup", "pydantic")]
    history = agent.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_run_sends_system_and_user_messages() -> None:
    llm = FakeLLM(["Thought: 完成\nAction: Finish[done]"])
    registry = FakeToolRegistry()
    agent = ReActAgent(
        name="react",
        llm=llm,
        tool_registry=registry,
        config=Config(),
        system_prompt="你是系统提示词",
        max_steps=1,
    )

    result = agent.run("你好")

    assert result == "done"
    assert llm.calls
    assert llm.calls[0][0]["role"] == "system"
    assert llm.calls[0][0]["content"] == "你是系统提示词"
    assert llm.calls[0][1]["role"] == "user"


def test_parse_output_supports_numbered_actions() -> None:
    llm = FakeLLM([])
    agent = ReActAgent(name="react", llm=llm, tool_registry=FakeToolRegistry(), config=Config())

    thought, actions = agent._parse_output(
        "Thought: 逐个执行\nAction:\n1) Search[a]\n2. Lookup[b]"
    )

    assert thought == "逐个执行"
    assert actions == ["Search[a]", "Lookup[b]"]


def test_run_executes_tool_then_finish_in_same_response() -> None:
    llm = FakeLLM(
        [
            "Thought: 先查再收敛\nAction:\n- Search[edge-case]\n- Finish[同轮结束]",
        ]
    )
    registry = FakeToolRegistry()
    agent = ReActAgent(
        name="react",
        llm=llm,
        tool_registry=registry,
        config=Config(),
        max_steps=2,
    )

    result = agent.run("测试同轮结束")

    assert result == "同轮结束"
    assert registry.calls == [("Search", "edge-case")]
    assert len(llm.calls) == 1
