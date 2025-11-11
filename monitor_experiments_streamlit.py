#!/usr/bin/env python3
"""
Streamlit app for monitoring EmbodiedBench experiment status.

Run with: streamlit run experiment_monitor.py
"""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# Configuration
RUNNING_DIR = Path("running")
LOGS_DIR = Path("logs")
LOCKS_DIR = RUNNING_DIR / "locks"

# Task definitions
TASKS = {
    "eb_alfred": ["base", "common_sense", "complex_instruction", "spatial", "visual_appearance", "long_horizon"],
    "eb_habitat": ["base", "common_sense", "complex_instruction", "spatial_relationship", "visual_appearance", "long_horizon"],
    "eb_manipulation": ["base", "common_sense", "complex", "spatial", "visual"],
    "eb_nav": ["base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"],
}


def extract_experiment_prefix(full_name: str) -> str:
    """Extract the base experiment name without temp/maxTokens/rep suffix."""
    # Remove the _temp0.X_maxTokensXXXX_repX suffix
    pattern = r"_temp[\d.]+_maxTokens\d+_rep\d+$"
    return re.sub(pattern, "", full_name)


def get_running_processes() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all running experiment processes by checking for __run_single_experiment.nu.
    
    This is more reliable than checking for embodiedbench.main since that may not be
    running continuously (e.g., waiting for remote servers, retrying habitat, etc.).
    
    Returns:
        Dict mapping prefix -> list of process info dicts
    """
    import subprocess
    
    try:
        # Run ps to find all __run_single_experiment.nu processes
        result = subprocess.run(
            ["ps", "a"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return {}
        
        running_processes = defaultdict(list)
        
        for line in result.stdout.split('\n'):
            if '__run_single_experiment.nu' not in line:
                continue
            
            # Additional paranoid check: verify the process is associated with a pts (pseudo-terminal)
            # This helps catch zombie processes that aren't actually in a gnome-terminal
            if 'pts/' not in line:
                continue
            
            # Parse the command line to extract the experiment prefix
            # Example: nu /path/to/__run_single_experiment.nu 20251108-combined-5-total-gated-rm2-step20--actor--huggingface /path/to/signal.file
            # The prefix is the first argument after __run_single_experiment.nu
            match = re.search(r'__run_single_experiment\.nu\s+([^\s]+)', line)
            
            # Extract PID (first column)
            pid_match = re.match(r'\s*(\d+)', line)
            
            if match:
                prefix = match.group(1)
                
                process_info = {
                    "pid": int(pid_match.group(1)) if pid_match else None,
                    "env": "unknown",  # Not available in this wrapper process
                    "eval_sets": "unknown",  # Not available in this wrapper process
                    "command": line.strip(),
                }
                running_processes[prefix].append(process_info)
        
        return dict(running_processes)
        
    except subprocess.TimeoutExpired:
        st.sidebar.warning("Timeout while checking running processes")
        return {}
    except Exception as e:
        st.sidebar.warning(f"Error checking running processes: {e}")
        return {}


def get_lock_info() -> Dict[str, Dict[str, Any]]:
    """Parse lock files to get lock file information."""
    lock_info = {}
    
    if not LOCKS_DIR.exists():
        return lock_info
    
    for lock_file in LOCKS_DIR.glob("*.lock"):
        try:
            content = lock_file.read_text().strip()
            # Parse format: timestamp=...; pid=12345
            match = re.search(r'pid=(\d+)', content)
            timestamp_match = re.search(r'timestamp=([^;]+)', content)
            
            if match:
                pid = int(match.group(1))
                timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
                
                prefix = lock_file.stem
                lock_info[prefix] = {
                    "pid": pid,
                    "timestamp": timestamp,
                    "lock_file": lock_file,
                }
        except Exception as e:
            st.sidebar.warning(f"Error reading lock file {lock_file.name}: {e}")
    
    return lock_info


def scan_experiments() -> Dict[str, Dict]:
    """
    Scan the running directory to find all experiments and their status.
    
    Returns:
        Dict with structure: {exp_name: {task: {subtask: status}}}
    """
    experiments = defaultdict(lambda: defaultdict(dict))
    
    for task_name, subtasks in TASKS.items():
        task_dir = RUNNING_DIR / task_name
        if not task_dir.exists():
            continue
        
        # Each directory under task_dir is an experiment run
        for exp_dir in task_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            
            # Extract experiment name (remove model prefix if present)
            exp_name = exp_dir.name
            if exp_name.startswith("Qwen2.5-VL-7B-Instruct_"):
                exp_name = exp_name[len("Qwen2.5-VL-7B-Instruct_"):]
            
            # Check each subtask
            for subtask in subtasks:
                subtask_dir = exp_dir / subtask
                summary_file = subtask_dir / "results" / "summary.json"
                summary_all_file = subtask_dir / "results" / "summary_all.json"
                
                # Check for either summary.json or summary_all.json
                summary_to_use = None
                if summary_file.exists():
                    summary_to_use = summary_file
                elif summary_all_file.exists():
                    summary_to_use = summary_all_file
                
                if summary_to_use:
                    try:
                        with open(summary_to_use) as f:
                            data = json.load(f)
                        
                        # Normalize key names (same as print_results_table.py)
                        if 'success_rate' in data and 'task_success' not in data:
                            data['task_success'] = data['success_rate']
                        
                        status = {
                            "status": "completed",
                            "task_success": data.get("task_success", None),
                            "summary_data": data,
                            "summary_path": summary_to_use,
                        }
                    except Exception as e:
                        status = {
                            "status": "error",
                            "error": str(e),
                        }
                elif subtask_dir.exists():
                    status = {
                        "status": "in_progress",
                    }
                else:
                    status = {
                        "status": "not_started",
                    }
                
                experiments[exp_name][task_name][subtask] = status
    
    return dict(experiments)


def get_log_path(exp_name: str, task: str, subtask: str) -> Optional[Path]:
    """Get the log file path for a specific experiment/task/subtask."""
    # Log structure: logs/{exp_name}/{TASK}/{subtask}.log
    
    # Convert task names
    task_map = {
        "eb_alfred": "EB-ALFRED",
        "eb_habitat": "EB-Habitat",
        "eb_manipulation": "EB-Manipulation",
        "eb_nav": "EB-Navigation",
    }
    
    task_dir_name = task_map.get(task, task)
    log_path = LOGS_DIR / exp_name / task_dir_name / f"{subtask}.log"
    
    if log_path.exists():
        return log_path
    return None


def format_status_badge(status: str) -> str:
    """Return colored badge HTML for status."""
    colors = {
        "completed": "🟢",
        "in_progress": "🟡",
        "not_started": "⚪",
        "error": "🔴",
    }
    return colors.get(status, "❓")


def main():
    st.set_page_config(
        page_title="EmbodiedBench Experiment Monitor",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🤖 EmbodiedBench Experiment Monitor")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
    
    if st.sidebar.button("🔄 Refresh Now") or auto_refresh:
        st.rerun()
    
    # Add timestamp
    st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Filter options
    st.sidebar.header("Filters")
    show_completed = st.sidebar.checkbox("Show completed", value=True)
    show_in_progress = st.sidebar.checkbox("Show in progress", value=True)
    show_not_started = st.sidebar.checkbox("Show not started", value=True)
    
    selected_tasks = st.sidebar.multiselect(
        "Select tasks",
        list(TASKS.keys()),
        default=list(TASKS.keys())
    )
    
    # Get data
    with st.spinner("Scanning experiments..."):
        experiments = scan_experiments()
        running_processes = get_running_processes()
        lock_info = get_lock_info()
    
    # Show running processes info
    st.header("🏃 Running Processes")
    
    if running_processes:
        # Create a flattened view of all running processes
        process_data = []
        for exp_name, processes in sorted(running_processes.items()):
            for proc in processes:
                process_data.append({
                    "Experiment": exp_name,
                    "PID": proc["pid"],
                    "Environment": proc["env"],
                    "Eval Set": proc["eval_sets"],
                })
        st.dataframe(pd.DataFrame(process_data), use_container_width=True)
        
        st.info(f"Found {len(process_data)} running process(es) across {len(running_processes)} experiment(s)")
    else:
        st.info("No experiments currently running")
    
    # Show lock file info
    if lock_info:
        with st.expander(f"🔒 Lock Files ({len(lock_info)})"):
            # Check which locks correspond to running processes
            lock_data = []
            for prefix, info in sorted(lock_info.items()):
                # Check if any running process matches this prefix
                has_running = any(
                    prefix in exp_name or exp_name.startswith(prefix)
                    for exp_name in running_processes.keys()
                )
                lock_data.append({
                    "Prefix": prefix,
                    "PID": info["pid"],
                    "Timestamp": info["timestamp"],
                    "Has Running Process": "✓" if has_running else "✗",
                })
            st.dataframe(pd.DataFrame(lock_data), use_container_width=True)
            
            # Count stale locks
            stale_count = sum(1 for item in lock_data if item["Has Running Process"] == "✗")
            if stale_count > 0:
                st.warning(f"⚠️ {stale_count} stale lock(s) detected (no corresponding running process)")
    
    # Experiment overview
    st.header("📊 Experiment Overview")
    
    if not experiments:
        st.warning("No experiments found in the running directory")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Summary Table", "🔍 Detailed View", "📈 Statistics"])
    
    with tab1:
        # Summary table showing all experiments and their overall status
        st.subheader("Experiment Status Summary")
        
        summary_data = []
        for exp_name, tasks in sorted(experiments.items()):
            prefix = extract_experiment_prefix(exp_name)
            is_running = exp_name in running_processes
            
            row = {
                "Experiment": exp_name,
                "Running": "🏃" if is_running else "",
            }
            
            # Count status for each task
            for task in selected_tasks:
                if task in tasks:
                    subtasks = tasks[task]
                    completed = sum(1 for s in subtasks.values() if s["status"] == "completed")
                    total = len(subtasks)
                    row[task] = f"{completed}/{total}"
                else:
                    row[task] = "N/A"
            
            # Overall completion
            all_subtasks = [s for t in tasks.values() for s in t.values()]
            completed_all = sum(1 for s in all_subtasks if s["status"] == "completed")
            total_all = len(all_subtasks)
            row["Overall"] = f"{completed_all}/{total_all}"
            
            summary_data.append(row)
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
    
    with tab2:
        # Detailed view with expandable sections per experiment
        st.subheader("Detailed Experiment Status")
        
        # Search/filter
        search = st.text_input("🔍 Search experiments", "")
        
        for exp_name, tasks in sorted(experiments.items()):
            if search and search.lower() not in exp_name.lower():
                continue
            
            prefix = extract_experiment_prefix(exp_name)
            is_running = exp_name in running_processes
            
            # Calculate overall progress
            all_subtasks = [s for t in tasks.values() for s in t.values()]
            completed = sum(1 for s in all_subtasks if s["status"] == "completed")
            in_progress = sum(1 for s in all_subtasks if s["status"] == "in_progress")
            total = len(all_subtasks)
            
            # Filter based on status
            if not show_completed and completed == total:
                continue
            if not show_in_progress and in_progress > 0 and completed < total:
                continue
            if not show_not_started and completed == 0 and in_progress == 0:
                continue
            
            # Create expander for each experiment
            progress_pct = (completed / total * 100) if total > 0 else 0
            status_icon = "🏃" if is_running else "⏸️"
            
            with st.expander(f"{status_icon} **{exp_name}** - {completed}/{total} completed ({progress_pct:.0f}%)"):
                # Create columns for each task
                task_cols = st.columns(len(selected_tasks))
                
                for idx, task in enumerate(selected_tasks):
                    with task_cols[idx]:
                        st.markdown(f"**{task.upper()}**")
                        
                        if task not in tasks:
                            st.info("No data")
                            continue
                        
                        subtasks = tasks[task]
                        
                        for subtask, info in sorted(subtasks.items()):
                            status_badge = format_status_badge(info["status"])
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"{status_badge} {subtask}")
                            
                            with col2:
                                if info["status"] == "completed":
                                    success = info.get("task_success", 0)
                                    if success is not None:
                                        st.metric("Success", f"{success:.2%}")
                            
                            # Show log path and button
                            log_path = get_log_path(exp_name, task, subtask)
                            if log_path and log_path.exists():
                                st.caption(f"📄 `{log_path}`")
                                
                                # View log button - opens in dialog
                                if st.button(f"👁️ View Log", key=f"log_{exp_name}_{task}_{subtask}"):
                                    try:
                                        log_content = log_path.read_text()
                                        lines = log_content.split('\n')
                                        total_lines = len(lines)
                                        
                                        # Use dialog for better viewing
                                        @st.dialog(f"Log: {subtask} ({total_lines} lines)", width="large")
                                        def show_log():
                                            st.code(log_content, language='text', line_numbers=False)
                                        
                                        show_log()
                                    except Exception as e:
                                        st.error(f"Error reading log: {e}")
                            
                            # Show summary details for completed tasks
                            if info["status"] == "completed" and "summary_data" in info:
                                with st.expander("📊 Details"):
                                    summary_data = info["summary_data"]
                                    for key, value in summary_data.items():
                                        if isinstance(value, float):
                                            st.write(f"**{key}:** {value:.4f}")
                                        else:
                                            st.write(f"**{key}:** {value}")
                        
                        st.markdown("---")
    
    with tab3:
        # Statistics view
        st.subheader("Experiment Statistics")
        
        # Overall statistics
        total_experiments = len(experiments)
        total_subtasks = sum(len([s for t in tasks.values() for s in t.values()]) for tasks in experiments.values())
        completed_subtasks = sum(len([s for t in tasks.values() for s in t.values() if s["status"] == "completed"]) for tasks in experiments.values())
        in_progress_subtasks = sum(len([s for t in tasks.values() for s in t.values() if s["status"] == "in_progress"]) for tasks in experiments.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Experiments", total_experiments)
        with col2:
            st.metric("Total Subtasks", total_subtasks)
        with col3:
            st.metric("Completed", completed_subtasks)
        with col4:
            st.metric("In Progress", in_progress_subtasks)
        
        # Progress by task
        st.markdown("### Progress by Task")
        task_stats = []
        for task in TASKS.keys():
            task_total = 0
            task_completed = 0
            for exp_tasks in experiments.values():
                if task in exp_tasks:
                    subtasks = exp_tasks[task]
                    task_total += len(subtasks)
                    task_completed += sum(1 for s in subtasks.values() if s["status"] == "completed")
            
            task_stats.append({
                "Task": task,
                "Total": task_total,
                "Completed": task_completed,
                "Completion %": (task_completed / task_total * 100) if task_total > 0 else 0
            })
        
        df_task_stats = pd.DataFrame(task_stats)
        st.dataframe(df_task_stats, use_container_width=True)
        
        # Task success rates (for completed tasks)
        st.markdown("### Average Task Success Rates")
        success_stats = []
        for task in TASKS.keys():
            successes = []
            for exp_tasks in experiments.values():
                if task in exp_tasks:
                    for subtask_info in exp_tasks[task].values():
                        if subtask_info["status"] == "completed" and "task_success" in subtask_info:
                            ts = subtask_info["task_success"]
                            if ts is not None:
                                successes.append(ts)
            
            if successes:
                avg_success = sum(successes) / len(successes)
                success_stats.append({
                    "Task": task,
                    "Avg Success": f"{avg_success:.2%}",
                    "Count": len(successes)
                })
        
        if success_stats:
            df_success = pd.DataFrame(success_stats)
            st.dataframe(df_success, use_container_width=True)
        else:
            st.info("No completed tasks with success metrics yet")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
