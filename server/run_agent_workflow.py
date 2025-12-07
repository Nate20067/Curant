#!/usr/bin/env python3
"""
Utility script to run the multi-agent workflow without audio hardware.

Usage examples:
    python3 server/run_agent_workflow.py "Add a save button to the toolbar"
    python3 server/run_agent_workflow.py "Refactor the logging service" \
        --repo-url https://github.com/example/repo.git --branch agent-changes
"""

import argparse
import sys
from typing import Optional

from app.services.audio_system import audio_service


def _configure_sandbox(repo_url: Optional[str], branch: str, image: str):
    """Initializes sandbox when repo_url is provided."""
    if not repo_url:
        return None

    from app.services.sandbox.agent_sandbox import Sandbox

    sandbox = Sandbox(repo_url_str=repo_url, branch_name=branch, image=image)
    audio_service.configure_agent_sandbox(sandbox)
    return sandbox


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run designer/programmer/validator workflow.")
    parser.add_argument("prompt", help="Natural language request to send to the designer agent.")
    parser.add_argument("--repo-url", help="Git repository URL to clone inside the sandbox.")
    parser.add_argument("--branch", default="agent-changes", help="Branch name for sandbox changes.")
    parser.add_argument("--image", default="python:3.9", help="Docker image used for sandbox.")

    args = parser.parse_args(argv)

    sandbox = _configure_sandbox(args.repo_url, args.branch, args.image)

    try:
        workflow = audio_service._run_agent_workflow(args.prompt)  # pylint: disable=protected-access
    finally:
        if sandbox:
            sandbox.cleanup(push_to_remote=False)

    print("\n=== DESIGN TASK ===")
    print(workflow.get("design_task", "").strip() or "(empty)")
    print("\n=== CODE RESULT ===")
    print(workflow.get("code_result", "").strip() or "(empty)")
    print("\n=== VALIDATION REPORT ===")
    print(workflow.get("validation_report", "").strip() or "(empty)")
    print("\n=== SPOKEN RESPONSE ===")
    print(workflow.get("speech", "").strip() or "(empty)")


if __name__ == "__main__":
    main()
