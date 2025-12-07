#!/usr/bin/env python3
"""
Utility script to run the multi-agent workflow without audio hardware.

This script orchestrates Designer → Programmer → Validator agents to fulfill
coding requests. It can optionally clone a Git repo into a sandboxed Docker
environment for real file operations.

Usage examples:
    # Run without sandbox (no file changes)
    python3 run_agent_workflow.py "Explain how to add logging"
    
    # Run with sandbox (creates files in repo)
    python3 run_agent_workflow.py \
        "Add a save button to the toolbar" \
        --repo-url https://github.com/myuser/myrepo.git \
        --branch feature/save-button
    
    # Use custom Docker image and save to file
    python3 run_agent_workflow.py \
        "Add Redis caching" \
        --repo-url git@github.com:myorg/api.git \
        --image python:3.11-slim \
        --output results.json \
        --format json \
        --verbose

Exit codes:
    0 = Success
    1 = Workflow error
    124 = Timeout
"""

import argparse
import json
import logging
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from app.services.audio_system import audio_service


#Context manager to timeout long running operations -> prevents infinite agent loops
@contextmanager
def timeout(seconds: int):
    """Context manager for timing out long operations"""
    #Handler function called when timeout alarm triggers
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds}s timeout")
    
    #If timeout is 0 or negative -> no timeout enforcement
    if seconds <= 0:
        yield
        return
    
    #Setting up signal handler for alarm -> saves old handler to restore later
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        #Canceling alarm and restoring previous handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


#Validates git repository URL format -> ensures URL is valid before cloning
def validate_repo_url(url: str) -> None:
    """Validate Git repository URL format"""
    #Checking for SSH format URLs -> git@github.com:user/repo.git
    if url.startswith(("git@", "ssh://")):
        #SSH URLs must contain @ and : characters
        if "@" not in url or ":" not in url:
            raise ValueError(f"Invalid SSH repo URL: {url}")
        return
    
    #Parsing HTTP/HTTPS URLs using urlparse
    parsed = urlparse(url)
    #Validating URL scheme is one of the allowed git protocols
    if parsed.scheme not in ('https', 'http', 'git'):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    #Ensuring URL has a domain component
    if not parsed.netloc:
        raise ValueError("URL missing domain")


#Function to initialize sandbox with docker and git -> returns None if no repo URL provided
def _configure_sandbox(repo_url: Optional[str], branch: str, image: str):
    """Initializes sandbox when repo_url is provided"""
    #If no repo URL provided -> running without sandbox environment
    if not repo_url:
        print("[INFO] Running without sandbox (no repo URL provided)")
        return None

    print(f"[INFO] Initializing sandbox from {repo_url}...")
    
    try:
        #Validating repo URL format before attempting to clone
        validate_repo_url(repo_url)
        
        #Importing Sandbox class -> lazy import to avoid docker dependency if not needed
        from app.services.sandbox.agent_sandbox import Sandbox
        
        #Creating sandbox instance -> clones repo and starts docker container
        sandbox = Sandbox(repo_url_str=repo_url, branch_name=branch, image=image)
        #Registering sandbox with audio service so agents can use file operations
        audio_service.configure_agent_sandbox(sandbox)
        
        #Printing sandbox configuration for user visibility
        print(f"[SUCCESS] Sandbox ready")
        print(f"   Branch: {branch}")
        print(f"   Image: {image}")
        print(f"   Workdir: {sandbox.workdir}")
        
        return sandbox
        
    except Exception as e:
        #Logging full error details and re-raising for caller to handle
        print(f"[ERROR] Failed to create sandbox: {e}", file=sys.stderr)
        logging.exception("Sandbox creation details:")
        raise

#Formats workflow results in text, json, or yaml format
def format_workflow_output(workflow: dict, format_type: str) -> str:
    """Format workflow results in requested format"""
    #Returning JSON formatted output with indentation
    if format_type == "json":
        return json.dumps(workflow, indent=2)
    #Returning YAML formatted output -> falls back to JSON if yaml not installed
    elif format_type == "yaml":
        try:
            import yaml
            return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
        except ImportError:
            print("[WARNING] PyYAML not installed, falling back to JSON", file=sys.stderr)
            return json.dumps(workflow, indent=2)
    else:  #Text format with simple section headers
        sections = []
        for section in ["design_task", "code_result", "validation_report", "speech"]:
            title = section.replace("_", " ").upper()
            content = workflow.get(section, "")
            sections.append(title)
            sections.append(content.strip() if content else "(empty)")
            sections.append("")  #Blank line between sections
        return '\n'.join(sections).strip()


#Main entry point for running agent workflow from command line
def main(argv=None):
    #Setting up argument parser with full help documentation
    parser = argparse.ArgumentParser(
        description="Run designer/programmer/validator workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    #Required positional argument for user prompt
    parser.add_argument(
        "prompt",
        help="Natural language request to send to the designer agent"
    )
    
    #Optional sandbox configuration arguments
    parser.add_argument(
        "--repo-url",
        help="Git repository URL to clone inside the sandbox"
    )
    parser.add_argument(
        "--branch",
        default="agent-changes",
        help="Branch name for sandbox changes (default: agent-changes)"
    )
    parser.add_argument(
        "--image",
        default="python:3.9",
        help="Docker image used for sandbox (default: python:3.9)"
    )
    
    #Output formatting options
    parser.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output",
        help="Write results to file instead of stdout"
    )
    
    #Execution control options
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Workflow timeout in seconds, 0 for no timeout (default: 300)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip sandbox cleanup for debugging"
    )
    
    #Logging configuration options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        help="Write logs to file"
    )
    
    #Parsing command line arguments
    args = parser.parse_args(argv)
    
    #Validating prompt is not empty or whitespace only
    if not args.prompt.strip():
        parser.error("Prompt cannot be empty")
    
    #Configuring logging level based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    #Setting up logging handlers for stderr and optional file output
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    
    #Applying logging configuration
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    #Initializing sandbox and workflow variables
    sandbox = None
    workflow = None
    
    try:
        #Creating sandbox if repo URL provided
        sandbox = _configure_sandbox(args.repo_url, args.branch, args.image)
        
        #Printing workflow start message with truncated prompt
        print(f"\n[START] Running workflow with prompt: {args.prompt[:60]}...")
        if args.timeout > 0:
            print(f"[INFO] Timeout set to: {args.timeout}s")
        
        #Running workflow with timeout protection -> prevents infinite agent loops
        with timeout(args.timeout):
            #Calling audio service to run full agent workflow
            workflow = audio_service._run_agent_workflow(args.prompt)
        
        #Checking if workflow returned valid results
        if not workflow:
            print("[WARNING] Workflow returned empty result", file=sys.stderr)
            return 1
        
        print("\n[SUCCESS] Workflow completed successfully")
        
        #Formatting output based on requested format
        output_text = format_workflow_output(workflow, args.format)
        
        #Writing output to file or printing to stdout
        if args.output:
            Path(args.output).write_text(output_text)
            print(f"[INFO] Results written to {args.output}")
        else:
            print(output_text)
        
        return 0
        
    except TimeoutError as e:
        #Handling workflow timeout -> returns standard timeout exit code
        print(f"\n[TIMEOUT] {e}", file=sys.stderr)
        return 124
        
    except KeyboardInterrupt:
        #Handling user interrupt -> returns standard interrupt exit code
        print("\n[INTERRUPTED] User cancelled operation", file=sys.stderr)
        return 130
        
    except Exception as e:
        #Handling all other errors with full logging
        print(f"\n[ERROR] Workflow failed: {e}", file=sys.stderr)
        logging.exception("Full error details:")
        return 1
        
    finally:
        #Cleaning up sandbox resources in finally block -> ensures cleanup even on errors
        if sandbox:
            if args.no_cleanup:
                #Preserving sandbox for debugging -> prints info for manual cleanup
                print(f"\n[DEBUG] Sandbox preserved for debugging:")
                print(f"   Workdir: {sandbox.workdir}")
                print(f"   Container: {sandbox.container.id if sandbox.container else 'N/A'}")
                if sandbox.container:
                    print(f"   Run 'docker stop {sandbox.container.id}' when done")
            else:
                #Running standard cleanup -> stops container and removes temp files
                print("\n[CLEANUP] Cleaning up sandbox...")
                try:
                    sandbox.cleanup(push_to_remote=False)
                    print("[SUCCESS] Cleanup complete")
                except Exception as e:
                    #Logging cleanup errors but not failing the script
                    print(f"[WARNING] Cleanup error: {e}", file=sys.stderr)


#Script entry point -> calls main function and exits with return code
if __name__ == "__main__":
    sys.exit(main())
