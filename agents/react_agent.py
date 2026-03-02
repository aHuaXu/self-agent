"""ReAct agent implementation."""

from __future__ import annotations

import re
from typing import Optional

from models.agent import Agent
from models.message import Message

try:
    from tools.tool_register import ToolRegistry, global_registry
except Exception:
    ToolRegistry = object  # type: ignore[assignment]
    global_registry = None


class ReactAgent(Agent):
    """A minimal ReAct agent with tool-use loop."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a ReAct assistant.\n"
        "When tools are needed, use this exact format:\n"
        "Thought: <reasoning>\n"
        "Action: <tool_name>\n"
        "Action Input: <input_string>\n"
        "Observation: <tool_result>\n"
        "... (repeat if needed)\n"
        "Final Answer: <answer_for_user>\n"
        "Rules:\n"
        "1. Use only tools from the available tools list.\n"
        "2. If no tool is needed, reply directly with Final Answer.\n"
        "3. Keep action input concise and executable.\n"
        "4. You may emit multiple Action/Action Input blocks in one turn if needed.\n"
    )

    _ACTION_BLOCK_RE = re.compile(
        r"Action\s*:\s*(?P<action>[^\n\r]+)\s*"
        r"(?:\nAction\s*Input\s*:\s*(?P<input>[\s\S]*?))?"
        r"(?=\nAction\s*:|\nObservation\s*:|\nFinal\s*Answer\s*:|\Z)",
        re.IGNORECASE,
    )
    _FINAL_ANSWER_RE = re.compile(
        r"Final\s*Answer\s*:\s*(?P<answer>[\s\S]+)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        name: str,
        llm,
        system_prompt: Optional[str] = None,
        config=None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 5,
    ) -> None:
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        self.tool_registry = tool_registry if tool_registry is not None else global_registry
        self.max_steps = max_steps

    def run(self, input_text: str, **kwargs) -> str:
        """Execute one ReAct run and return the final output text."""
        self.add_message(Message(role="user", content=input_text))

        max_steps = int(kwargs.pop("max_steps", self.max_steps))
        scratchpad = ""
        last_response = ""

        for _ in range(max_steps):
            messages = self._build_messages(input_text=input_text, scratchpad=scratchpad)
            response = self.llm.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            ).strip()
            last_response = response

            final_answer = self._extract_final_answer(response)
            if final_answer is not None:
                self.add_message(Message(role="assistant", content=final_answer))
                self._trim_history()
                return final_answer

            actions = self._extract_actions(response)
            if not actions:
                self.add_message(Message(role="assistant", content=response))
                self._trim_history()
                return response

            observation_lines: list[str] = []
            for action, action_input in actions:
                observation = self._execute_tool(action=action, action_input=action_input)
                observation_lines.append(f"Observation: {observation}")

            scratchpad += f"{response}\n" + "\n".join(observation_lines) + "\n"

        self.add_message(Message(role="assistant", content=last_response))
        self._trim_history()
        return last_response

    def _build_messages(self, input_text: str, scratchpad: str) -> list[dict[str, str]]:
        tools_desc = self._tools_description()
        system_content = f"{self.system_prompt or self.DEFAULT_SYSTEM_PROMPT}\nAvailable tools:\n{tools_desc}\n"

        user_content = f"Question: {input_text}\n"
        if scratchpad:
            user_content += f"{scratchpad}\nContinue based on the latest observation."

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _tools_description(self) -> str:
        registry = self.tool_registry
        if registry is None:
            return "No tools available"

        desc_getter = getattr(registry, "get_tools_description", None)
        if callable(desc_getter):
            try:
                return str(desc_getter())
            except Exception:
                return "No tools available"

        return "No tools available"

    def _execute_tool(self, action: str, action_input: str) -> str:
        registry = self.tool_registry
        if registry is None:
            return "Error: tool registry is not configured"

        executor = getattr(registry, "execute_tool", None)
        if not callable(executor):
            return "Error: tool registry does not implement execute_tool"

        try:
            return str(executor(action, action_input))
        except Exception as exc:
            return f"Error: tool execution failed: {exc}"

    @classmethod
    def _extract_final_answer(cls, text: str) -> Optional[str]:
        match = cls._FINAL_ANSWER_RE.search(text)
        if not match:
            return None
        return match.group("answer").strip()

    @classmethod
    def _extract_actions(cls, text: str) -> list[tuple[str, str]]:
        actions: list[tuple[str, str]] = []
        for match in cls._ACTION_BLOCK_RE.finditer(text):
            action = (match.group("action") or "").strip()
            action_input = (match.group("input") or "").strip()
            if action:
                actions.append((action, action_input))
        return actions

    def _trim_history(self) -> None:
        limit = self.config.max_history_length
        if limit > 0 and len(self._history) > limit:
            self._history = self._history[-limit:]
