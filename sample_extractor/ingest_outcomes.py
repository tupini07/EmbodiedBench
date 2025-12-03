"""
Ingest episode outcomes from EmbodiedBench result JSON files.

Scans running/{task_dir}/{model_run}/base/results/episode_*_final_res.json files
and populates the episode_outcomes table.

Usage:
    python -m sample_extractor.ingest_outcomes --base-dir running --tasks eb_alfred eb_habitat
    python -m sample_extractor.ingest_outcomes --base-dir running --tasks eb_alfred --run-pattern "*Qwen*"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from .db import get_conn, create_schema


def parse_run_id_from_path(run_path: str) -> tuple[str, str]:
    """
    Extract model_name and run_id from path like:
    'running/eb_alfred/Qwen2.5-VL-7B-Instruct_20251106-..._rep1'
    
    Returns: (model_name, full_run_id)
    """
    basename = os.path.basename(run_path)
    # Split on first underscore to separate model from run
    parts = basename.split("_", 1)
    if len(parts) == 2:
        model_name = parts[0]
        run_id = basename
    else:
        model_name = basename
        run_id = basename
    return model_name, run_id


def ingest_task_outcomes(
    conn,
    base_dir: str,
    task_dir: str,
    run_pattern: Optional[str] = None,
) -> int:
    """
    Scan {base_dir}/{task_dir}/*/base/results/episode_*_final_res.json
    and upsert to episode_outcomes table.
    
    Returns count of episodes ingested.
    """
    task_path = Path(base_dir) / task_dir
    if not task_path.exists():
        print(f"Task directory not found: {task_path}")
        return 0
    
    # Map task_dir back to task name (e.g., eb_alfred -> EB-ALFRED)
    task_name_map = {
        "eb_alfred": "EB-ALFRED",
        "eb_habitat": "EB-Habitat",
        "eb_navigation": "EB-Navigation",
    }
    task_name = task_name_map.get(task_dir.lower(), task_dir.upper())
    
    count = 0
    cur = conn.cursor()
    
    # Find all run directories
    for run_dir in task_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Apply run pattern filter if specified
        if run_pattern and not Path(run_dir.name).match(run_pattern):
            continue
        
        model_name, run_id = parse_run_id_from_path(str(run_dir))
        results_dir = run_dir / "base" / "results"
        
        if not results_dir.exists():
            continue
        
        # Find all episode_*_final_res.json files
        for result_file in results_dir.glob("episode_*_final_res.json"):
            # Extract episode number from filename
            match = re.search(r"episode_(\d+)_final_res\.json", result_file.name)
            if not match:
                continue
            
            episode = int(match.group(1))
            
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                
                # Extract fields
                task_success = float(data.get("task_success", 0.0))
                task_progress = float(data.get("task_progress", 0.0))
                num_invalid_actions = int(data.get("num_invalid_actions", 0))
                num_steps = int(data.get("num_steps", 0))
                instruction = data.get("instruction", "")
                
                # Upsert to DB
                cur.execute(
                    """
                    INSERT INTO episode_outcomes(
                        run_id, task, episode, task_success, task_progress,
                        num_invalid_actions, num_steps, instruction
                    )
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id, task, episode) DO UPDATE SET
                        task_success=excluded.task_success,
                        task_progress=excluded.task_progress,
                        num_invalid_actions=excluded.num_invalid_actions,
                        num_steps=excluded.num_steps,
                        instruction=excluded.instruction,
                        ingested_at=CURRENT_TIMESTAMP
                    """,
                    (run_id, task_name, episode, task_success, task_progress,
                     num_invalid_actions, num_steps, instruction),
                )
                count += 1
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error processing {result_file}: {e}")
                continue
    
    conn.commit()
    return count


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest episode outcomes from EmbodiedBench result JSON files"
    )
    parser.add_argument(
        "--base-dir",
        default="running",
        help="Base directory containing task subdirectories (default: running)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["eb_alfred", "eb_habitat"],
        help="Task directories to scan (default: eb_alfred eb_habitat)",
    )
    parser.add_argument(
        "--run-pattern",
        help="Glob pattern to filter run directories (e.g., '*Qwen*')",
    )
    parser.add_argument(
        "--db",
        default="outputs/steps_cache.sqlite",
        help="Database path (default: outputs/steps_cache.sqlite)",
    )
    
    args = parser.parse_args(argv)
    
    conn = get_conn(args.db)
    create_schema(conn)
    
    total_count = 0
    for task_dir in args.tasks:
        print(f"Scanning {task_dir}...")
        count = ingest_task_outcomes(conn, args.base_dir, task_dir, args.run_pattern)
        print(f"  Ingested {count} episodes from {task_dir}")
        total_count += count
    
    print(f"\nTotal episodes ingested: {total_count}")
    
    # Print summary stats
    cur = conn.cursor()
    cur.execute(
        """
        SELECT task, COUNT(DISTINCT run_id) as num_runs, 
               COUNT(*) as num_episodes,
               AVG(task_success) as avg_success
        FROM episode_outcomes
        GROUP BY task
        """
    )
    print("\nSummary:")
    for row in cur.fetchall():
        task, num_runs, num_episodes, avg_success = row
        print(f"  {task}: {num_runs} runs, {num_episodes} episodes, "
              f"{avg_success*100:.1f}% avg success rate")
    
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
