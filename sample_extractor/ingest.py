"""
Ingest EmbodiedBench log steps into SQLite persistent cache.

Usage examples:
  python -m sample_extractor.ingest --runs 20251109-total-gated-combined-5-rl-step-55temp0.6_maxTokens8192_rep1,Qwen2.5-7b_Rerun_NoReasoning_temp0.0_maxTokens2048_rep2
  python -m sample_extractor.ingest --runs 20251109-total-gated-combined-5-rl-step-55temp0.6_maxTokens8192_rep1 --tasks EB-ALFRED
  python -m sample_extractor.ingest --db outputs/steps_cache.sqlite --limit-per-log 200

This phase only ingests step records; ranking upsert is a separate phase (to be added).

Schema created automatically if missing (see db.py).

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

from .parser import parse_log, StepRecord
from .db import get_conn, create_schema, upsert_step, update_run_stats, _hash_raw

DEFAULT_DB = "outputs/steps_cache.sqlite"
DEFAULT_TASKS = ["EB-ALFRED", "EB-Habitat"]

def _iter_log_files(run_id: str, tasks: List[str]) -> List[Tuple[str, str, str]]:
    """
    Return list of (task, subtask, path_to_log).
    subtask derived from filename (strip .log).
    """
    results: List[Tuple[str, str, str]] = []
    for task in tasks:
        task_dir = os.path.join("logs", run_id, task)
        if not os.path.isdir(task_dir):
            continue
        for name in os.listdir(task_dir):
            if not name.endswith(".log"):
                continue
            subtask = name[:-4]
            path = os.path.join(task_dir, name)
            results.append((task, subtask, path))
    return results

def ingest_run(conn, run_id: str, tasks: List[str], limit_per_log: int | None = None) -> Tuple[int, int, int]:
    """
    Ingest a single run_id over selected tasks.

    Returns (n_logs_processed, n_new_steps_inserted, n_skipped_duplicates)
    Global dedup: skips inserting if raw_output hash already exists anywhere in steps.
    """
    log_defs = _iter_log_files(run_id, tasks)
    processed_logs = 0
    new_steps = 0
    skipped_steps = 0
    # Preload existing content hashes for global dedup
    existing_hashes = {row[0] for row in conn.execute("SELECT content_hash FROM steps")}
    for task, subtask, path in log_defs:
        try:
            meta, records = parse_log(path)
        except Exception as e:
            print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
            continue
        if limit_per_log is not None:
            records = records[:limit_per_log]
        model_name = meta.model_name
        for r in records:
            h = _hash_raw(r.raw_output)
            if h in existing_hashes:
                skipped_steps += 1
                continue
            upsert_step(
                conn,
                run_id=run_id,
                task=task,
                subtask=subtask,
                model_name=model_name,
                episode=r.episode,
                step=r.step,
                prompt=r.prompt,
                raw_output=r.raw_output,
                parseable=r.parseable,
                retries_before_success=r.retries_before_success,
                log_path=path,
            )
            existing_hashes.add(h)
            new_steps += 1
        processed_logs += 1
    if processed_logs:
        update_run_stats(conn, run_id)
    return processed_logs, new_steps, skipped_steps

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Ingest EmbodiedBench logs into SQLite cache.")
    ap.add_argument("--runs", type=str, required=True,
                    help="Comma-separated run_ids (folder names under logs/)")
    ap.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                    help=f"Comma-separated task names (default: {','.join(DEFAULT_TASKS)})")
    ap.add_argument("--db", type=str, default=DEFAULT_DB, help="SQLite database path.")
    ap.add_argument("--limit-per-log", type=int, default=None,
                    help="If set, limit number of steps ingested per log file.")
    ap.add_argument("--dry", action="store_true",
                    help="Parse & report counts without writing to DB.")
    args = ap.parse_args(argv)

    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if not run_ids:
        print("No runs provided.", file=sys.stderr)
        return 1
    if not tasks:
        print("No tasks provided.", file=sys.stderr)
        return 1

    if args.dry:
        print("Dry run (no DB writes).")
        for run_id in run_ids:
            defs = _iter_log_files(run_id, tasks)
            step_total = 0
            for task, subtask, path in defs:
                try:
                    _, records = parse_log(path)
                    if args.limit_per_log is not None:
                        records = records[: args.limit_per_log]
                    step_total += len(records)
                except Exception as e:
                    print(f"[WARN] parse failed {path}: {e}", file=sys.stderr)
            print(f"Run {run_id}: logs={len(defs)} steps={step_total}")
        return 0

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    conn = get_conn(args.db)
    create_schema(conn)

    grand_logs = 0
    grand_new_steps = 0
    grand_skipped = 0
    for run_id in run_ids:
        logs_processed, new_steps, skipped_steps = ingest_run(
            conn, run_id, tasks, limit_per_log=args.limit_per_log
        )
        conn.commit()
        print(f"[INGEST] run={run_id} logs={logs_processed} new_steps={new_steps} skipped={skipped_steps}")
        grand_logs += logs_processed
        grand_new_steps += new_steps
        grand_skipped += skipped_steps

    print(f"Total logs processed: {grand_logs}  New steps: {grand_new_steps}  Skipped duplicates: {grand_skipped}")
    conn.commit()
    conn.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
