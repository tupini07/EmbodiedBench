#!/usr/bin/env python3
"""Run EmbodiedBench evaluations in parallel across multiple environments.

WARNING: Parallel execution may cause resource conflicts:
- AI2THOR (EB-ALFRED, EB-Navigation) may have port/display conflicts
- Habitat-sim (EB-Habitat) needs exclusive GPU/rendering access
- CoppeliaSim (EB-Manipulation) has its own server ports
- All share the same X display server

For production use, consider running evaluations sequentially (see bash script).
This script is kept as a reference for parallel execution patterns.
"""

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Lock
from urllib.error import URLError
from urllib.request import urlopen

# File lock for thread-safe writing
file_lock = Lock()

# Event for graceful shutdown
shutdown_event = Event()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nReceived interrupt signal. Shutting down gracefully...")
    print("Waiting for running evaluations to complete...")
    shutdown_event.set()


def check_remote_url(url):
    """Check if the remote URL is reachable."""
    try:
        with urlopen(f"{url}/models", timeout=5) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


def write_to_dones_file(dones_file, message):
    """Thread-safe writing to the dones file."""
    with file_lock:
        with open(dones_file, "a") as f:
            f.write(f"{message}\n")


def run_evaluation(eval_config, exp_name, n_shots, dones_file):
    """Run a single evaluation task."""
    name = eval_config["name"]
    conda_env = eval_config["conda_env"]
    env_vars = eval_config["env_vars"]
    command = eval_config["command"]

    # Check if shutdown was requested
    if shutdown_event.is_set():
        print(f"{name} - Skipping due to shutdown signal")
        return False

    print(f"Running {name} evaluation...")

    # Build conda activation
    conda_setup = (
        f". ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env}"
    )

    # Build environment variable exports
    env_exports = " && ".join(f"export {k}='{v}'" for k, v in env_vars.items())
    env_prefix = f"{env_exports} && " if env_exports else ""

    # Build and run full command
    full_command = f"{conda_setup} && {env_prefix}{command}"
    result = subprocess.run(
        full_command, shell=True, executable="/bin/bash", capture_output=False
    )

    success = result.returncode == 0
    if success:
        write_to_dones_file(dones_file, name)
        print(f"{name} completed successfully")
    else:
        print(f"{name} failed")

    return success
def main():
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get experiment name from command line
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <exp_name>")
        sys.exit(1)

    exp_name = sys.argv[1]

    remote_url = os.environ.get("REMOTE_URL", "http://localhost:43289/v1")
    os.environ["REMOTE_URL"] = remote_url
    os.environ["OPENAI_API_KEY"] = "empty"
    
    n_shots = os.environ.get("N_SHOTS", "3")
    os.environ["N_SHOTS"] = n_shots
    os.environ["DISPLAY"] = ":1"

    # Check if REMOTE_URL is reachable
    if not check_remote_url(remote_url):
        print(f"Error: REMOTE_URL {remote_url} is not reachable.")
        print("Maybe need to start the SSH tunnel with 'scripts/start_ssh_tunnel.sh'?")
        sys.exit(1)

    print(f"Running evaluation with exp_name: {exp_name}")

    # Set up display and start X server
    subprocess.Popen(
        "python -m embodiedbench.envs.eb_alfred.scripts.startx 1",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    # Create dones file directory
    dones_file = Path("running") / f"{exp_name}_dones.txt"
    dones_file.parent.mkdir(exist_ok=True)

    # Get current working directory for CoppeliaSim paths
    coppeliasim_root = Path.cwd() / "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
    assert (
        coppeliasim_root.exists()
    ), f"CoppeliaSim root not found at {coppeliasim_root}"

    # Define evaluation tasks
    evaluations = [
        # {
        #     "name": "EB-ALFRED",
        #     "conda_env": "embench",
        #     "env_vars": {},
        #     "command": f"python -m embodiedbench.main env=eb-alf model_name='vllm-model' exp_name='{exp_name}' n_shots={n_shots}",
        # },
        # {
        #     "name": "EB-Habitat",
        #     "conda_env": "embench",
        #     "env_vars": {},
        #     "command": f"python -m embodiedbench.main env=eb-hab model_name='vllm-model' exp_name='{exp_name}' n_shots={n_shots}",
        # },
        # {
        #     "name": "EB-Manipulation",
        #     "conda_env": "embench_man",
        #     "env_vars": {
        #         "COPPELIASIM_ROOT": str(coppeliasim_root),
        #         "LD_LIBRARY_PATH": f"${{LD_LIBRARY_PATH}}:{coppeliasim_root}",
        #         "QT_QPA_PLATFORM_PLUGIN_PATH": str(coppeliasim_root),
        #     },
        #     "command": f"python -m embodiedbench.main env=eb-man model_name='vllm-model' exp_name='{exp_name}' n_shots={n_shots}",
        # },
        {
            "name": "EB-Navigation",
            "conda_env": "embench_nav",
            "env_vars": {},
            "command": f"python -m embodiedbench.main env=eb-nav model_name='vllm-model' exp_name='{exp_name}' n_shots={n_shots}",
        },
    ]

    # Run evaluations in parallel using ThreadPoolExecutor
    # Note: This may cause resource conflicts (see docstring warning above)
    with ThreadPoolExecutor(max_workers=len(evaluations)) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                run_evaluation, eval_config, exp_name, n_shots, str(dones_file)
            )
            for eval_config in evaluations
        ]

        # Wait for all tasks to complete and collect results
        results = [
            future.result()
            if not (exc := future.exception())
            else (print(f"Evaluation generated an exception: {exc}") or False)
            for future in as_completed(futures)
        ]

    # Check if shutdown was requested
    if shutdown_event.is_set():
        print("\nEvaluations interrupted by user.")
        return 130  # Standard exit code for SIGINT

    # Check if all evaluations completed successfully
    if all(results):
        print("\nAll evaluations completed successfully.")
        write_to_dones_file(str(dones_file), "ALL DONE")
        return 0
    else:
        print("\nSome evaluations failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
