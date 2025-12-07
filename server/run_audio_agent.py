#!/usr/bin/env python3
"""Launches the live audio conversation loop with optional sandbox support."""

import argparse

from app.services.audio_system import audio_service


def _configure_sandbox(repo_url: str | None, branch: str, image: str):
    if not repo_url:
        return None

    from app.services.sandbox.agent_sandbox import Sandbox

    sandbox = Sandbox(repo_url_str=repo_url, branch_name=branch, image=image)
    audio_service.configure_agent_sandbox(sandbox)
    return sandbox


def main():
    parser = argparse.ArgumentParser(description="Run audio-based agent conversation.")
    parser.add_argument("--repo-url", help="Git repository URL for sandboxed tool access")
    parser.add_argument("--branch", default="agent-changes", help="Branch name used inside the sandbox")
    parser.add_argument("--image", default="python:3.9", help="Docker image used for sandbox execution")
    args = parser.parse_args()

    sandbox = _configure_sandbox(args.repo_url, args.branch, args.image)

    try:
        audio_service.parallel_audio_stream()
    finally:
        if sandbox:
            sandbox.cleanup(push_to_remote=False)


if __name__ == "__main__":
    main()
