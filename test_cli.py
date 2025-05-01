#!/usr/bin/env python
"""
Test script for the agent-memory CLI.

This script tests the basic functionality of the CLI commands.
It doesn't actually run the servers or schedule tasks, but it
verifies that the commands are properly registered and can be
invoked without errors.
"""

import subprocess
import sys


def run_command(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"stderr: {e.stderr}")
        return None


def test_version():
    """Test the version command."""
    print("Testing 'agent-memory version'...")
    output = run_command([sys.executable, "-m", "agent_memory_server.cli", "version"])
    if output and "agent-memory-server version" in output:
        print("✅ Version command works")
    else:
        print("❌ Version command failed")


def test_api_help():
    """Test the api command help."""
    print("Testing 'agent-memory api --help'...")
    output = run_command(
        [sys.executable, "-m", "agent_memory_server.cli", "api", "--help"]
    )
    if output and "Run the REST API server" in output:
        print("✅ API command help works")
    else:
        print("❌ API command help failed")


def test_mcp_help():
    """Test the mcp command help."""
    print("Testing 'agent-memory mcp --help'...")
    output = run_command(
        [sys.executable, "-m", "agent_memory_server.cli", "mcp", "--help"]
    )
    if output and "Run the MCP server" in output:
        print("✅ MCP command help works")
    else:
        print("❌ MCP command help failed")


def test_schedule_task_help():
    """Test the schedule-task command help."""
    print("Testing 'agent-memory schedule-task --help'...")
    output = run_command(
        [sys.executable, "-m", "agent_memory_server.cli", "schedule-task", "--help"]
    )
    if output and "Schedule a background task by path" in output:
        print("✅ Schedule task command help works")
    else:
        print("❌ Schedule task command help failed")


def test_task_worker_help():
    """Test the task-worker command help."""
    print("Testing 'agent-memory task-worker --help'...")
    output = run_command(
        [sys.executable, "-m", "agent_memory_server.cli", "task-worker", "--help"]
    )
    if output and "Start a Docket worker using the Docket name from settings" in output:
        print("✅ Task worker command help works")
    else:
        print("❌ Task worker command help failed")


if __name__ == "__main__":
    print("Testing agent-memory CLI commands...")
    test_version()
    test_api_help()
    test_mcp_help()
    test_schedule_task_help()
    test_task_worker_help()
    print("All tests completed.")
