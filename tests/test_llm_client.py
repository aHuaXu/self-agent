import os
from types import SimpleNamespace

import pytest

from llm.llm import LLMClient


def test_chat_calls_openai_compatible_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_ID", "gpt-4o-mini")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")

    captured: dict = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("llm.llm.OpenAI", FakeOpenAI)

    client = LLMClient()
    result = client.chat(
        messages=[{"role": "user", "content": "ping"}],
        temperature=0.3,
        max_tokens=20,
    )

    assert result == "ok"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["messages"] == [{"role": "user", "content": "ping"}]
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 20


@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION") != "1",
    reason="Set RUN_LLM_INTEGRATION=1 to run real LLM API integration test.",
)
def test_chat_integration_real_api() -> None:
    client = LLMClient()
    result = client.chat(
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0,
        max_tokens=16,
    )
    assert isinstance(result, str)
    assert result.strip() != ""
