"""
Streamlit app to display ETAs for running experiments across all reps and subtasks.
Usage: streamlit run print_job_etas.py
"""

import os
import re
import streamlit as st
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_eta_from_line(line):
    """
    Extract ETA from a line like:
    Episodes:  76%|███████▌  | 38/50 [4:01:57<1:24:16, 421.39s/it][2025-11-20 19:31:55,634][INFO] - Evaluating episode 38 ...
    Returns the ETA string (e.g., "1:24:16") or None if not found.
    Note: The format is [elapsed<remaining, speed] where remaining is the ETA
    """
    # Pattern to match the ETA after the < symbol
    # The time format can be H:MM:SS, HH:MM:SS, MM:SS, or even HH:MM:SS:MS
    match = re.search(r'<([\d:]+),', line)
    if match:
        return match.group(1)
    return None


def get_latest_eta_for_log(log_file_path):
    """
    Read a log file from the bottom and return the latest ETA found.
    Returns None if no ETA is found or if file doesn't exist.
    Efficiently reads large files by searching backwards from the end.
    
    Note: Progress bars use \\r (carriage return) to overwrite the same line,
    so we need to search through all \\r-separated segments, not just \\n lines.
    """
    if not os.path.exists(log_file_path):
        return None
    
    try:
        # Read file from the end in chunks to find the latest ETA
        # This is much more efficient for large files (200MB+)
        chunk_size = 16384  # Read 16KB at a time (increased for better \r handling)
        
        with open(log_file_path, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            
            if file_size == 0:
                return None
            
            # Read backwards in chunks
            buffer = b''
            position = file_size
            
            while position > 0:
                # Calculate how much to read
                read_size = min(chunk_size, position)
                position -= read_size
                
                # Seek to position and read
                f.seek(position)
                chunk = f.read(read_size)
                
                # Prepend to buffer (since we're reading backwards)
                buffer = chunk + buffer
                
                # Try to decode and search for ETA
                try:
                    text = buffer.decode('utf-8', errors='ignore')
                    
                    # Progress bars use \r to overwrite, so we need to check all segments
                    # Split on \r first to get all overwrites, then check each segment
                    segments = text.split('\r')
                    
                    # Search backwards through segments (most recent overwrites are last)
                    for segment in reversed(segments):
                        # Clean segment and check for Episodes progress with ETA
                        if 'Episodes:' in segment and '<' in segment and '|' in segment:
                            eta = parse_eta_from_line(segment)
                            if eta:
                                return eta
                        
                except UnicodeDecodeError:
                    # Continue reading more data if decode fails
                    continue
                    
    except Exception as e:
        st.warning(f"Error reading {log_file_path}: {e}")
        return None
    
    return None


def collect_etas(experiment_base_name, logs_dir="logs"):
    """
    Collect ETAs for all reps and subtasks of an experiment.
    Returns a nested dict: {rep: {benchmark/subtask: eta}}
    """
    etas = defaultdict(dict)
    
    # Find all rep directories
    for rep in [1, 2, 3]:
        rep_dir = os.path.join(logs_dir, f"{experiment_base_name}_rep{rep}")
        
        if not os.path.exists(rep_dir):
            continue
        
        # Look for benchmark directories (EB-ALFRED, EB-Habitat, etc.)
        for benchmark_dir in Path(rep_dir).iterdir():
            if not benchmark_dir.is_dir():
                continue
            
            benchmark_name = benchmark_dir.name
            
            # Look for .log files in the benchmark directory
            for log_file in benchmark_dir.glob("*.log"):
                subtask_name = log_file.stem  # filename without .log extension
                full_task_name = f"{benchmark_name}/{subtask_name}"
                
                eta = get_latest_eta_for_log(str(log_file))
                if eta:
                    etas[rep][full_task_name] = eta
                else:
                    # Check if file is empty or has no progress yet
                    if log_file.stat().st_size == 0:
                        etas[rep][full_task_name] = "N/A (empty)"
                    else:
                        # File has content but no ETA found - might be done or not started
                        etas[rep][full_task_name] = "N/A"
    
    return etas


def parse_time_to_seconds(time_str):
    """
    Convert time string (HH:MM:SS or MM:SS or H:MM:SS) to seconds.
    Returns None if the time string is invalid or N/A.
    """
    if not time_str or time_str == "N/A" or "N/A" in time_str:
        return None
    
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        else:
            return None
    except (ValueError, AttributeError):
        return None


def format_seconds_to_time(seconds):
    """
    Convert seconds to HH:MM:SS or MM:SS format.
    """
    if seconds is None:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_eta_table(etas):
    """
    Format the ETAs as a markdown table.
    Rows are reps, columns are subtasks.
    """
    if not etas:
        return "No data found for this experiment."
    
    # Collect all unique subtasks across all reps
    all_subtasks = set()
    for rep_data in etas.values():
        all_subtasks.update(rep_data.keys())
    
    # Sort subtasks for consistent ordering
    sorted_subtasks = sorted(all_subtasks)
    
    if not sorted_subtasks:
        return "No subtasks found for this experiment."
    
    # Build the table header
    table_lines = []
    header = "| Rep | " + " | ".join(sorted_subtasks) + " |"
    separator = "|-----|" + "|".join(["-------" for _ in sorted_subtasks]) + "|"
    
    table_lines.append(header)
    table_lines.append(separator)
    
    # Build rows for each rep
    for rep in sorted(etas.keys()):
        row_data = [f"Rep {rep}"]
        for subtask in sorted_subtasks:
            eta = etas[rep].get(subtask, "N/A")
            row_data.append(eta)
        
        row = "| " + " | ".join(row_data) + " |"
        table_lines.append(row)
    
    # Calculate statistics
    all_times = []
    for rep_data in etas.values():
        for eta_str in rep_data.values():
            seconds = parse_time_to_seconds(eta_str)
            if seconds is not None:
                all_times.append(seconds)
    
    if all_times:
        max_time = max(all_times)
        avg_time = sum(all_times) / len(all_times)
        
        table_lines.append("")
        table_lines.append(f"**Statistics:**")
        table_lines.append(f"- Maximum ETA: {format_seconds_to_time(max_time)}")
        table_lines.append(f"- Average ETA: {format_seconds_to_time(avg_time)}")
        table_lines.append(f"- Total tasks with ETA: {len(all_times)}")
    
    return "\n".join(table_lines)


def main():
    st.set_page_config(
        page_title="Experiment ETA Monitor",
        page_icon="⏱️",
        layout="wide"
    )
    
    st.title("⏱️ Experiment ETA Monitor")
    st.markdown("Monitor the estimated time to completion for running experiments")
    
    # Get the script directory to find the logs folder
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "logs"
    
    # Input for experiment name
    col1, col2 = st.columns([3, 1])
    
    with col1:
        experiment_name = st.text_input(
            "Experiment Name",
            value="20251120-final-ablation-5temp0.6_maxTokens8192",
            help="Enter the experiment name without the _rep suffix"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        refresh_button = st.button("🔄 Refresh", use_container_width=True)
    
    # Auto-refresh toggle with warning
    st.markdown("---")
    auto_refresh = st.checkbox(
        "⚠️ Auto-refresh every 60 seconds (use sparingly - may still be slow for large logs)", 
        value=False
    )
    
    if auto_refresh:
        import time
        time.sleep(60)
        st.rerun()
    
    # Display last updated time
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if experiment_name:
        with st.spinner("Collecting ETAs..."):
            etas = collect_etas(experiment_name, str(logs_dir))
        
        if not etas:
            st.error(f"❌ No data found for experiment: {experiment_name}")
            st.info(f"Make sure the experiment directories exist in {logs_dir}")
        else:
            # Display the table
            table_md = format_eta_table(etas)
            st.markdown(table_md)
            
            # Show experiment info
            with st.expander("📊 Experiment Details"):
                total_reps = len(etas)
                all_subtasks = set()
                for rep_data in etas.values():
                    all_subtasks.update(rep_data.keys())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reps", total_reps)
                with col2:
                    st.metric("Total Subtasks", len(all_subtasks))
                with col3:
                    # Count running tasks
                    running_tasks = sum(
                        1 for rep_data in etas.values() 
                        for eta in rep_data.values() 
                        if eta != "N/A" and "N/A" not in eta and eta != "00:00"
                    )
                    st.metric("Running Tasks", running_tasks)


if __name__ == "__main__":
    main()
