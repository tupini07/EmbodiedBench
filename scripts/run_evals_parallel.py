#!/usr/bin/env python3
"""
Parallel evaluation runner for EmbodiedBench.
Runs all environment evaluations in parallel processes.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import threading
from queue import Queue
import requests


class EvalRunner:
    """Manages parallel evaluation runs for different environments."""
    
    def __init__(self, exp_name: str, n_shots: int = 3, remote_url: str = "http://localhost:43289/v1"):
        self.exp_name = exp_name
        self.n_shots = n_shots
        self.remote_url = remote_url
        self.display = ":1"
        self.x_server_pid = None
        self.dones_file = Path("running") / f"{exp_name}_dones.txt"
        self.log_dir = Path("running") / "logs" / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure running directory exists
        Path("running").mkdir(exist_ok=True)
        
        # Environment configurations
        self.environments = [
            {
                "name": "EB-ALFRED",
                "conda_env": "embench",
                "env_code": "eb-alf",
                "extra_env_vars": {}
            },
            {
                "name": "EB-Habitat",
                "conda_env": "embench",
                "env_code": "eb-hab",
                "extra_env_vars": {}
            },
            {
                "name": "EB-Manipulation",
                "conda_env": "embench_man",
                "env_code": "eb-man",
                "extra_env_vars": {
                    "COPPELIASIM_ROOT": str(Path.cwd() / "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"),
                    "LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{Path.cwd() / 'CoppeliaSim_Pro_V4_1_0_Ubuntu20_04'}",
                    "QT_QPA_PLATFORM_PLUGIN_PATH": str(Path.cwd() / "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04")
                }
            },
            {
                "name": "EB-Navigation",
                "conda_env": "embench_nav",
                "env_code": "eb-nav",
                "extra_env_vars": {}
            }
        ]
    
    def check_remote_url(self) -> bool:
        """Check if the remote URL is reachable."""
        try:
            response = requests.get(f"{self.remote_url}/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ Remote URL {self.remote_url} is reachable")
                return True
            else:
                print(f"❌ Remote URL {self.remote_url} returned status code {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error connecting to {self.remote_url}: {e}")
            print("Maybe need to start the SSH tunnel with 'scripts/start_ssh_tunnel.sh'?")
            return False
    
    def start_x_server(self) -> bool:
        """Start X server on display :1."""
        print(f"Starting X server on display {self.display}...")
        
        try:
            # Start X server
            process = subprocess.Popen(
                ["python", "-m", "embodiedbench.envs.eb_alfred.scripts.startx", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.x_server_pid = process.pid
            
            # Wait for X server to start
            max_attempts = 10
            for attempt in range(1, max_attempts + 1):
                try:
                    result = subprocess.run(
                        ["xdpyinfo", "-display", self.display],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=2
                    )
                    if result.returncode == 0:
                        print(f"✅ X server started successfully on display {self.display}")
                        return True
                except Exception:
                    pass
                
                print(f"⏳ Waiting for X server... (attempt {attempt}/{max_attempts})")
                time.sleep(2)
            
            print(f"❌ Failed to start X server on display {self.display}")
            return False
            
        except Exception as e:
            print(f"❌ Error starting X server: {e}")
            return False
    
    def run_environment(self, env_config: Dict) -> Dict:
        """Run evaluation for a single environment."""
        env_name = env_config["name"]
        conda_env = env_config["conda_env"]
        env_code = env_config["env_code"]
        extra_env_vars = env_config["extra_env_vars"]
        
        log_file = self.log_dir / f"{env_name.replace('-', '_').lower()}.log"
        
        print(f"\n{'='*60}")
        print(f"Starting {env_name} evaluation...")
        print(f"Conda env: {conda_env}")
        print(f"Log file: {log_file}")
        print(f"{'='*60}\n")
        
        # Prepare environment variables
        env_vars = os.environ.copy()
        env_vars["DISPLAY"] = self.display
        env_vars["REMOTE_URL"] = self.remote_url
        env_vars["OPENAI_API_KEY"] = "empty"
        
        # Add extra environment variables
        for key, value in extra_env_vars.items():
            if "$" in value:
                # Expand environment variables in the value
                value = os.path.expandvars(value)
            env_vars[key] = value
        
        # Build the command
        conda_base = os.path.expanduser("~/miniconda3")
        cmd = f"""
        source {conda_base}/etc/profile.d/conda.sh && \
        conda activate {conda_env} && \
        python -m embodiedbench.main \
            env={env_code} \
            model_name='vllm-model' \
            exp_name="{self.exp_name}" \
            n_shots={self.n_shots}
        """
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    executable="/bin/bash",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env_vars
                )
                
                # Wait for process to complete
                returncode = process.wait()
                
                duration = time.time() - start_time
                
                if returncode == 0:
                    print(f"✅ {env_name} completed successfully in {duration:.1f}s")
                    status = "success"
                else:
                    print(f"❌ {env_name} failed with return code {returncode} after {duration:.1f}s")
                    status = "failed"
                
                return {
                    "name": env_name,
                    "status": status,
                    "returncode": returncode,
                    "duration": duration,
                    "log_file": str(log_file)
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {env_name} crashed with exception: {e}")
            return {
                "name": env_name,
                "status": "crashed",
                "returncode": -1,
                "duration": duration,
                "error": str(e),
                "log_file": str(log_file)
            }
    
    def run_parallel(self) -> List[Dict]:
        """Run all environments in parallel."""
        print(f"\n{'='*60}")
        print(f"Running parallel evaluation with exp_name: {self.exp_name}")
        print(f"N_SHOTS: {self.n_shots}")
        print(f"Remote URL: {self.remote_url}")
        print(f"{'='*60}\n")
        
        # Check prerequisites
        if not self.check_remote_url():
            return []
        
        if not self.start_x_server():
            return []
        
        # Run evaluations in parallel using threads
        results_queue = Queue()
        threads = []
        
        def worker(env_config):
            result = self.run_environment(env_config)
            results_queue.put(result)
        
        # Start all threads
        for env_config in self.environments:
            thread = threading.Thread(target=worker, args=(env_config,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Sort results by original order
        env_names = [env["name"] for env in self.environments]
        results.sort(key=lambda x: env_names.index(x["name"]))
        
        return results
    
    def write_completion_status(self, results: List[Dict]):
        """Write completion status to the dones file."""
        with open(self.dones_file, 'w') as f:
            for result in results:
                if result["status"] == "success":
                    f.write(f"{result['name']}\n")
            
            # Check if all succeeded
            all_success = all(r["status"] == "success" for r in results)
            if all_success:
                f.write("ALL DONE\n")
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of all evaluation results."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}\n")
        
        total_duration = sum(r["duration"] for r in results)
        
        for result in results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} {result['name']}: {result['status']} ({result['duration']:.1f}s)")
            print(f"   Log: {result['log_file']}")
            if result.get("error"):
                print(f"   Error: {result['error']}")
            print()
        
        print(f"Total time: {total_duration:.1f}s")
        print(f"Completion status written to: {self.dones_file}")
        
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\nResults: {success_count}/{len(results)} environments completed successfully")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run EmbodiedBench evaluations in parallel"
    )
    parser.add_argument(
        "exp_name",
        help="Experiment name"
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=int(os.environ.get("N_SHOTS", 3)),
        help="Number of shots (default: 3 or $N_SHOTS)"
    )
    parser.add_argument(
        "--remote-url",
        default=os.environ.get("REMOTE_URL", "http://localhost:43289/v1"),
        help="Remote API URL (default: http://localhost:43289/v1 or $REMOTE_URL)"
    )
    
    args = parser.parse_args()
    
    runner = EvalRunner(
        exp_name=args.exp_name,
        n_shots=args.n_shots,
        remote_url=args.remote_url
    )
    
    try:
        results = runner.run_parallel()
        
        if results:
            runner.write_completion_status(results)
            runner.print_summary(results)
            
            # Exit with error if any evaluation failed
            if any(r["status"] != "success" for r in results):
                sys.exit(1)
        else:
            print("❌ No evaluations were run")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
