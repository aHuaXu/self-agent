from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMClient:
    """Wraps an OpenAI-compatible chat completion client."""

    PROVIDER_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "moonshot": "https://api.moonshot.cn/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
    }

    def __init__(self) -> None:
        provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        api_key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL_ID")
        base_url = os.getenv("LLM_BASE_URL") or self.PROVIDER_BASE_URLS.get(provider)

        missing = [
            name
            for name, value in {
                "LLM_API_KEY": api_key,
                "LLM_MODEL_ID": model,
                "LLM_BASE_URL": base_url,
            }.items()
            if not value or str(value).upper().startswith("YOUR")
        ]
        if missing:
            raise ValueError(
                f"Missing/placeholder env vars: {', '.join(missing)}. "
                "Please set them in .env."
            )

        self.provider = provider
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = 0.7,
        max_tokens: int | None = 512,
        **kwargs: Any,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""


if __name__ == "__main__":
    try:
        client = LLMClient()
        result = client.chat(
            messages=[{"role": "user", "content": "你好，请回复：连接成功"}],
            temperature=0,
            max_tokens=64,
        )
        print(result)
    except Exception as exc:
        print(f"LLM call failed: {exc}")
