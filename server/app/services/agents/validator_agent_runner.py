"""Utility helpers for orchestrating designer/programmer/validator workflow."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.agents.setup_agents import (
    designer_agent,
    programmer_agent,
    validator_agent,
)
from app.services.sandbox.agent_sandbox import Sandbox


def _extract_content(response: Any) -> str:
    """Best-effort extraction of text content from OpenAI responses."""
    if response is None:
        return ""

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message and hasattr(message, "content"):
            return message.content or ""
        if hasattr(first, "content"):
            return first.content or ""

    if hasattr(response, "content"):
        return response.content or ""

    return str(response)


class ValidatorAgentRunner:
    """Coordinates the sandbox-backed workflow between agents and exposes helpers."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    @classmethod
    def from_repo(
        cls,
        repo_url: str,
        branch_name: str = "agent-changes",
        image: str = "python:3.9",
    ) -> "ValidatorAgentRunner":
        sandbox = Sandbox(repo_url_str=repo_url, branch_name=branch_name, image=image)
        return cls(sandbox)

    def __enter__(self) -> "ValidatorAgentRunner":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup(push_to_remote=False)

    def cleanup(self, commit_message: str = "Agent automated changes", push_to_remote: bool = False) -> None:
        if self.sandbox:
            self.sandbox.cleanup(commit_message=commit_message, push_to_remote=push_to_remote)

    def validate(self, design: str, code: str, conversation_history: Optional[list] = None) -> str:
        return validator_agent(design, code, self.sandbox, conversation_history)

    def run_full_cycle(self, user_prompt: str) -> Dict[str, str]:
        """Runs designer -> programmer -> validator and returns their raw outputs."""
        design_response = designer_agent(user_prompt, sandbox=self.sandbox)
        design_text = _extract_content(design_response)

        code_text = programmer_agent(design_text, self.sandbox)
        validation_text = self.validate(design_text, code_text)

        return {
            "design": design_text,
            "code": code_text,
            "validation": validation_text,
        }
