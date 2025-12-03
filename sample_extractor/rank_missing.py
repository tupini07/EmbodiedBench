"""
Score (rank) all missing step records in the SQLite cache and upsert results.

Usage:
  python -m sample_extractor.rank_missing --runs 20251109-total-gated-combined-5-rl-step-55temp0.6_maxTokens8192_rep1
  python -m sample_extractor.rank_missing --runs 20251109-total-gated-combined-5-rl-step-55temp0.6_maxTokens8192_rep1,Qwen2.5-7b_Rerun_NoReasoning_temp0.0_maxTokens2048_rep2 --limit 300
  python -m sample_extractor.rank_missing --trapi-instance gcr/internal --trapi-api-version 2024-12-01-preview --trapi-deployment gpt-5_2025-08-07 --trapi

Notes:
- Requires prior ingestion (see ingest.py) so steps table is populated.
- Determines missing scores by LEFT JOIN where scores.key IS NULL (key = first 16 hex chars of content hash).
- Uses existing ranking.evaluate_records helper (Azure rotation, direct TRAPI, or TRAPI Azure identity).
- Writes scores via db.upsert_score().
- Supports --dry to preview number of steps without scoring.

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .db import get_conn, fetch_missing_scores, upsert_score
from .ranking import evaluate_records, step_hash
from .parser import StepRecord  # reuse dataclass for compatibility with build_evaluation_requests

def _rows_to_step_records(rows) -> List[StepRecord]:
    """
    Convert DB fetch_missing_scores rows to StepRecord list.
    Row layout:
      run_id, model_name, task, subtask,
      episode, step, prompt, raw_output,
      parseable, retries_before_success
    We only need fields required for scorer.
    """
    records: List[StepRecord] = []
    for run_id, model_name, task, subtask, episode, step, prompt, raw_output, parseable, retries in rows:
        # attempt_index and log_path not known here; set sentinel values
        rec = StepRecord(
            episode=episode,
            step=step,
            prompt=prompt,
            raw_output=raw_output,
            parseable=bool(parseable),
            retries_before_success=retries,
            attempt_index=-1,
            log_path=f"{run_id}/{task}/{subtask}",
        )
        records.append(rec)
    return records

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Rank missing step records and upsert scores.")
    ap.add_argument("--db", type=str, default="outputs/steps_cache.sqlite", help="SQLite database path.")
    ap.add_argument("--runs", type=str, default="", help="Comma-separated run_ids to restrict (optional).")
    ap.add_argument("--subtask", type=str, default=None, help="Restrict to a single subtask name (e.g. base).")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of missing steps processed after filtering.")
    ap.add_argument("--batch-size", type=int, default=10, help="Number of records per scoring batch.")
    ap.add_argument("--dry", action="store_true", help="Do not call model; just report counts.")
    ap.add_argument("--fallback-azure", action="store_true", help="Retry failed (empty) TRAPI scores with Azure rotation.")
    # Scoring mode switches (mirror rank.py)
    ap.add_argument("--trapi", action="store_true", help="Use TRAPI scorer (direct or identity).")
    ap.add_argument("--trapi-model", type=str, default="gpt-5", help="TRAPI model name for direct OpenAI path.")
    ap.add_argument("--trapi-api-key", type=str, default=None, help="Explicit API key for direct TRAPI scorer.")
    ap.add_argument("--trapi-instance", type=str, default=None, help="TRAPI instance/apiPath for Azure identity mode.")
    ap.add_argument("--trapi-api-version", type=str, default=None, help="TRAPI API version for Azure identity mode.")
    ap.add_argument("--trapi-deployment", type=str, default=None, help="TRAPI deployment name for Azure identity mode.")
    args = ap.parse_args(argv)

    run_filter = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else None

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    conn = get_conn(args.db)

    rows = list(fetch_missing_scores(conn, run_filter=run_filter))
    if args.subtask:
        rows = [r for r in rows if r[3] == args.subtask]  # subtask field position
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry:
        print(f"Missing steps to score: {len(rows)} (dry run)")
        return 0

    if not rows:
        print("No missing scores.")
        return 0

    # Build StepRecord list
    step_records = _rows_to_step_records(rows)

    # Perform scoring in batches, upserting after each batch
    import asyncio
    total = len(step_records)
    total_scored = 0
    
    for start in range(0, total, args.batch_size):
        batch_records = step_records[start : start + args.batch_size]
        batch_rows = rows[start : start + args.batch_size]
        
        try:
            batch_results = asyncio.run(
                evaluate_records(
                    batch_records,
                    use_trapi=args.trapi,
                    trapi_model=args.trapi_model,
                    trapi_instance=args.trapi_instance,
                    trapi_api_version=args.trapi_api_version,
                    trapi_deployment=args.trapi_deployment,
                )
            )
        except Exception as e:
            print(f"[ERROR] scoring batch start={start} failed: {e}", file=sys.stderr)
            return 2

        # Fallback re-score for empty results
        if args.fallback_azure:
            failed = [
                r for r in batch_results
                if r.final_quality == 0
                and (not r.explanation)
                and all(v == 0 for v in r.intermediate_scores.values())
            ]
            if failed:
                from sample_extractor.ranking.azure_gpt import AzureGPTScorer, build_evaluation_requests
                fallback_scorer = AzureGPTScorer(concurrency=min(len(failed), 5))
                reqs = build_evaluation_requests([
                    StepRecord(
                        episode=fr.episode,
                        step=fr.step,
                        prompt=next(rec.prompt for rec in batch_records if rec.episode == fr.episode and rec.step == fr.step),
                        raw_output=next(rec.raw_output for rec in batch_records if rec.episode == fr.episode and rec.step == fr.step),
                        parseable=True,
                        retries_before_success=0,
                        attempt_index=-1,
                        log_path="fallback",
                    ) for fr in failed
                ])
                try:
                    fb_results = asyncio.run(fallback_scorer.score_many(reqs))
                    # Replace failed results with fallback results (match by key)
                    fb_map = {r.key: r for r in fb_results}
                    new_batch = []
                    for r in batch_results:
                        new_batch.append(fb_map.get(r.key, r))
                    batch_results = new_batch
                except Exception as e:
                    print(f"[WARN] fallback Azure scoring failed: {e}", file=sys.stderr)

        # Upsert results immediately after scoring each batch
        for row, res in zip(batch_rows, batch_results):
            run_id, model_name, task, subtask, episode, step, prompt, raw_output, parseable, retries = row
            upsert_score(
                conn,
                key=res.key,  # Already truncated 16 hex chars
                run_id=run_id,
                model_name=model_name,
                episode=episode,
                step=step,
                final_quality=res.final_quality,
                intermediate_scores=res.intermediate_scores,
                explanation=res.explanation,
                flags=res.flags,
                raw_output_len=res.raw_output_len,
            )
        
        # Commit after each batch to persist progress
        conn.commit()
        total_scored += len(batch_results)
        print(f"[BATCH] scored & committed {total_scored}/{total} steps")

    print(f"Complete: scored & upserted {total_scored} steps.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
