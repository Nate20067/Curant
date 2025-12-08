#!/usr/bin/env python3
"""Launches the live audio conversation loop with optional sandbox support."""

import argparse
import logging
import sys

from app.services.audio_system import audio_service


def _configure_sandbox(repo_url: str | None, branch: str, image: str, repo_branch: str | None):
    if not repo_url:
        logging.info("Running without sandbox (no repo URL provided)")
        return None

    try:
        from app.services.sandbox.agent_sandbox import Sandbox
        logging.info("Initializing sandbox from %s", repo_url)
        sandbox = Sandbox(
            repo_url_str=repo_url,
            branch_name=branch,
            image=image,
            repo_branch=repo_branch
        )
        audio_service.configure_agent_sandbox(sandbox)
        return sandbox
    except Exception as e:
        logging.error("Failed to create sandbox: %s", e)
        logging.error("Continuing without sandbox - agents will have limited capabilities")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run audio-based agent conversation.",
        epilog="Press Ctrl+C to stop the audio session."
    )
    parser.add_argument("--repo-url", help="Git repository URL for sandboxed tool access")
    parser.add_argument("--branch", default="agent-changes", help="Branch name used for sandbox commits")
    parser.add_argument(
        "--repo-branch",
        help="Existing repo branch/tag/commit to checkout before creating the sandbox branch"
    )
    parser.add_argument("--image", default="python:3.9", help="Docker image used for sandbox execution")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level)")
    args = parser.parse_args()

    # Configure logging based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logging.info("Starting audio agent runner")

    sandbox = _configure_sandbox(args.repo_url, args.branch, args.image, args.repo_branch)
    if sandbox:
        logging.info("Sandbox ready at %s on branch %s", sandbox.workdir, args.branch)
        if args.repo_branch:
            logging.info("Repository checked out to %s before sandbox branch creation", args.repo_branch)

    try:
        logging.info("Launching parallel audio stream...")
        logging.info("Speak into your microphone to interact with the agents")
        audio_service.parallel_audio_stream()
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logging.error("Audio stream encountered an error: %s", e)
        logging.exception("Full error details:")
        sys.exit(1)
    finally:
        if sandbox:
            logging.info("Cleaning up sandbox...")
            try:
                sandbox.cleanup(push_to_remote=False)
                logging.info("Sandbox cleaned up successfully")
            except Exception as e:
                logging.error("Error during sandbox cleanup: %s", e)

    logging.info("Audio agent runner finished cleanly")


if __name__ == "__main__":
    main()
