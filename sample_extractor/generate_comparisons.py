"""
Generate comparison candidates between target and baseline models.

Finds episodes where target model succeeds and baseline fails, then identifies
the best reasoning step from target and worst/failure step from baseline.

Usage:
    python -m sample_extractor.generate_comparisons \
        --target-runs "20251109-total-gated*" \
        --baseline-runs "Qwen*" \
        --pattern success_vs_failure
    
    python -m sample_extractor.generate_comparisons \
        --target-runs "20251109-total-gated*" \
        --baseline-runs "Qwen*" \
        --min-contrast 5.0 \
        --tasks EB-ALFRED
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional
from fnmatch import fnmatch

from .db import get_conn, create_schema

def generate_comparisons(
    conn,
    target_pattern: str,
    baseline_pattern: str,
    comparison_pattern: str = "success_vs_failure",
    min_contrast: float = 0.0,
    tasks: Optional[List[str]] = None,
) -> int:
    """
    Generate comparison candidates by finding episodes where target succeeds
    and baseline fails.
    
    Args:
        conn: Database connection
        target_pattern: Glob pattern for target run_ids (e.g., "*magma*")
        baseline_pattern: Glob pattern for baseline run_ids (e.g., "*Qwen*")
        comparison_pattern: Type of comparison ("success_vs_failure", "all")
        min_contrast: Minimum contrast score to include
        tasks: Optional list of tasks to filter (e.g., ["EB-ALFRED"])
    
    Returns:
        Number of comparison candidates generated
    """
    cur = conn.cursor()
    
    # Get all run_ids matching patterns
    cur.execute("SELECT DISTINCT run_id FROM episode_outcomes")
    all_runs = [row[0] for row in cur.fetchall()]
    
    target_runs = [r for r in all_runs if fnmatch(r, target_pattern)]
    baseline_runs = [r for r in all_runs if fnmatch(r, baseline_pattern)]
    
    if not target_runs:
        print(f"No target runs found matching pattern: {target_pattern}")
        return 0
    if not baseline_runs:
        print(f"No baseline runs found matching pattern: {baseline_pattern}")
        return 0
    
    print(f"Found {len(target_runs)} target runs, {len(baseline_runs)} baseline runs")
    print(f"Target runs: {target_runs[:3]}{'...' if len(target_runs) > 3 else ''}")
    print(f"Baseline runs: {baseline_runs[:3]}{'...' if len(baseline_runs) > 3 else ''}")
    
    count = 0
    
    # Build task filter
    task_filter = ""
    task_params = []
    if tasks:
        task_filter = "AND eo_target.task IN ({})".format(",".join("?" for _ in tasks))
        task_params = tasks
    
    # For each target run
    for target_run in target_runs:
        # For each baseline run
        for baseline_run in baseline_runs:
            # Find episodes where both have outcomes
            query = f"""
                SELECT 
                    eo_target.task,
                    eo_target.episode,
                    eo_target.task_success as target_success,
                    eo_baseline.task_success as baseline_success,
                    eo_target.num_invalid_actions as target_invalid,
                    eo_baseline.num_invalid_actions as baseline_invalid
                FROM episode_outcomes eo_target
                JOIN episode_outcomes eo_baseline
                    ON eo_target.task = eo_baseline.task
                    AND eo_target.episode = eo_baseline.episode
                WHERE eo_target.run_id = ?
                    AND eo_baseline.run_id = ?
                    {task_filter}
            """
            
            cur.execute(query, (target_run, baseline_run, *task_params))
            episode_pairs = cur.fetchall()
            
            for row in episode_pairs:
                task, episode, target_success, baseline_success, target_invalid, baseline_invalid = row
                
                # Apply comparison pattern filter
                if comparison_pattern == "success_vs_failure":
                    # Target succeeds (task_success == 1.0), baseline fails (< 1.0)
                    if not (target_success >= 1.0 and baseline_success < 1.0):
                        continue
                elif comparison_pattern == "all":
                    pass  # Include all pairs
                
                # Get best step from target
                cur.execute(
                    """
                    SELECT sc.step, sc.final_quality, sc.hallucination
                    FROM scores sc
                    WHERE sc.run_id = ? AND sc.episode = ?
                    ORDER BY sc.final_quality DESC
                    LIMIT 1
                    """,
                    (target_run, episode),
                )
                target_step_row = cur.fetchone()
                
                # Get worst step from baseline (prioritize hallucinations)
                cur.execute(
                    """
                    SELECT sc.step, sc.final_quality, sc.hallucination
                    FROM scores sc
                    WHERE sc.run_id = ? AND sc.episode = ?
                    ORDER BY sc.hallucination DESC, sc.final_quality ASC
                    LIMIT 1
                    """,
                    (baseline_run, episode),
                )
                baseline_step_row = cur.fetchone()
                
                # Skip if we don't have scores for both
                if not target_step_row or not baseline_step_row:
                    continue
                
                target_step, target_quality, _ = target_step_row
                baseline_step, baseline_quality, baseline_halluc = baseline_step_row
                
                # Compute contrast score
                quality_gap = (target_quality or 0) - (baseline_quality or 0)
                halluc_penalty = 10 if baseline_halluc else 0
                invalid_penalty = (baseline_invalid or 0) * 2
                success_bonus = 5 if (target_success >= 1.0 and baseline_success < 1.0) else 0
                
                contrast_score = quality_gap + halluc_penalty + invalid_penalty + success_bonus
                
                # Apply minimum contrast filter
                if contrast_score < min_contrast:
                    continue
                
                # Insert comparison candidate
                cur.execute(
                    """
                    INSERT INTO comparison_candidates(
                        task, episode, target_run_id, baseline_run_id,
                        target_best_step, target_best_quality,
                        baseline_worst_step, baseline_worst_quality,
                        baseline_has_hallucination, contrast_score
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(task, episode, target_run_id, baseline_run_id) DO UPDATE SET
                        target_best_step=excluded.target_best_step,
                        target_best_quality=excluded.target_best_quality,
                        baseline_worst_step=excluded.baseline_worst_step,
                        baseline_worst_quality=excluded.baseline_worst_quality,
                        baseline_has_hallucination=excluded.baseline_has_hallucination,
                        contrast_score=excluded.contrast_score,
                        created_at=CURRENT_TIMESTAMP
                    """,
                    (
                        task, episode, target_run, baseline_run,
                        target_step, target_quality,
                        baseline_step, baseline_quality,
                        int(baseline_halluc or 0), contrast_score,
                    ),
                )
                count += 1
    
    conn.commit()
    return count

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate comparison candidates between target and baseline models"
    )
    parser.add_argument(
        "--target-runs",
        required=True,
        help="Glob pattern for target run_ids (e.g., '*magma*')",
    )
    parser.add_argument(
        "--baseline-runs",
        required=True,
        help="Glob pattern for baseline run_ids (e.g., '*Qwen*')",
    )
    parser.add_argument(
        "--pattern",
        default="success_vs_failure",
        choices=["success_vs_failure", "all"],
        help="Comparison pattern (default: success_vs_failure)",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=0.0,
        help="Minimum contrast score to include (default: 0.0)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task filter (e.g., EB-ALFRED EB-Habitat)",
    )
    parser.add_argument(
        "--db",
        default="outputs/steps_cache.sqlite",
        help="Database path (default: outputs/steps_cache.sqlite)",
    )
    
    args = parser.parse_args(argv)
    
    conn = get_conn(args.db)
    create_schema(conn)
    
    print(f"Generating comparisons with pattern: {args.pattern}")
    if args.min_contrast > 0:
        print(f"Minimum contrast score: {args.min_contrast}")
    
    count = generate_comparisons(
        conn,
        args.target_runs,
        args.baseline_runs,
        args.pattern,
        args.min_contrast,
        args.tasks,
    )
    
    print(f"\nGenerated {count} comparison candidates")
    
    # Print summary stats
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            task,
            COUNT(*) as num_comparisons,
            AVG(contrast_score) as avg_contrast,
            MAX(contrast_score) as max_contrast,
            SUM(baseline_has_hallucination) as num_with_halluc
        FROM comparison_candidates
        GROUP BY task
        ORDER BY task
        """
    )
    
    print("\nSummary by task:")
    for row in cur.fetchall():
        task, num_comp, avg_contrast, max_contrast, num_halluc = row
        print(f"  {task}: {num_comp} comparisons, "
              f"avg contrast={avg_contrast:.1f}, max={max_contrast:.1f}, "
              f"{num_halluc} with hallucinations")
    
    # Show top contrasts
    cur.execute(
        """
        SELECT task, episode, contrast_score, 
               target_best_quality, baseline_worst_quality,
               baseline_has_hallucination
        FROM comparison_candidates
        ORDER BY contrast_score DESC
        LIMIT 10
        """
    )
    
    print("\nTop 10 contrasts:")
    for row in cur.fetchall():
        task, ep, contrast, tgt_q, base_q, halluc = row
        halluc_str = " 🚨HALLUC" if halluc else ""
        print(f"  {task} ep{ep}: contrast={contrast:.1f} "
              f"(target={tgt_q} vs baseline={base_q}){halluc_str}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
