"""
CLI tool to rank reasoning quality for a single EmbodiedBench log.

Usage examples:
    python -m sample_extractor.rank logs/.../EB-ALFRED/base.log --limit 100
    python -m sample_extractor.rank logs/.../EB-ALFRED/base.log --json --limit 50
    python -m sample_extractor.rank logs/.../EB-ALFRED/base.log --dry --limit 20  # no network, placeholder scores

Options:
--limit N              Limit number of step records evaluated (after parsing order)
--episodes 1,2,5       Restrict to listed episodes
--concurrency N        Override concurrency (default env or 30)
--json                 Emit JSON list
--dry                  Skip network calls (produce zeroed scores; still caching format)
--cache PATH           Override cache path
--trapi                Use TRAPI scorer (direct OpenAI or Azure identity endpoint)
--trapi-model NAME     Model name for TRAPI scorer (default gpt-5)
--trapi-api-key KEY    Explicit API key override (else OPENAI_API_KEY env is used)
--trapi-instance NAME  TRAPI instance/apiPath (e.g. gcr/internal) for Azure identity mode
--trapi-api-version V  TRAPI API version (e.g. 2024-12-01-preview)
--trapi-deployment D   TRAPI deployment name (e.g. gpt-5_2025-08-07)
"""

from __future__ import annotations

import argparse
import json
import os
import asyncio
from typing import List, Set

from .parser import parse_log, StepRecord
from .ranking import (
    AzureGPTScorer,
    build_evaluation_requests,
    evaluate_records,
    EvaluationResult,
)

def _filter(records: List[StepRecord], episodes: Set[int]) -> List[StepRecord]:
    if not episodes:
        return records
    return [r for r in records if r.episode in episodes]

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Rank reasoning quality for EmbodiedBench log steps.")
    parser.add_argument("logpath")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--episodes", type=str, default="")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("RANK_CONCURRENCY", "30")))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry", action="store_true", help="Do not call Azure; return empty scores.")
    parser.add_argument("--cache", type=str, default=os.getenv("RANK_CACHE_PATH", "outputs/rank_cache.jsonl"))
    parser.add_argument("--trapi", action="store_true", help="Use direct OpenAI TRAPI scorer (bypass Azure rotation).")
    parser.add_argument("--trapi-model", type=str, default="gpt-5", help="TRAPI model name (default gpt-5).")
    parser.add_argument("--trapi-api-key", type=str, default=None, help="Explicit API key for TRAPI scorer (fallback OPENAI_API_KEY env).")
    parser.add_argument("--trapi-instance", type=str, default=None, help="TRAPI instance / api path (e.g. gcr/internal).")
    parser.add_argument("--trapi-api-version", type=str, default=None, help="TRAPI API version (e.g. 2024-12-01-preview).")
    parser.add_argument("--trapi-deployment", type=str, default=None, help="TRAPI deployment name (e.g. gpt-5_2025-08-07).")
    args = parser.parse_args(argv)

    episodes_set: Set[int] = set()
    if args.episodes:
        for part in args.episodes.split(","):
            part = part.strip()
            if part:
                try:
                    episodes_set.add(int(part))
                except ValueError:
                    pass

    meta, records = parse_log(args.logpath)
    records = _filter(records, episodes_set)
    subset = records[: args.limit]

    if args.dry:
        # Produce empty EvaluationResult objects without network
        scorer = AzureGPTScorer(concurrency=args.concurrency, cache_path=args.cache)
        reqs = build_evaluation_requests(subset)
        results: List[EvaluationResult] = []
        for req in reqs:
            results.append(
                EvaluationResult(
                    key=req.key,
                    episode=req.episode,
                    step=req.step,
                    final_quality=0,
                    intermediate_scores={d: 0 for d in scorer.cache.get(req.key).intermediate_scores.keys()} if scorer.cache.get(req.key) else {d: 0 for d in []},
                    explanation="(dry run)",
                    flags={"possible_hallucination": False, "overly_verbose": False},
                    raw_output_len=len(req.raw_output),
                    cached=False,
                )
            )
    else:
        # Use unified evaluate_records helper (auto-select scorer).
        trapi_api_key = args.trapi_api_key or os.getenv("OPENAI_API_KEY")
        results = asyncio.run(
            evaluate_records(
                subset,
                use_trapi=args.trapi,
                trapi_model=args.trapi_model,
                trapi_api_key=trapi_api_key,
                trapi_instance=args.trapi_instance,
                trapi_api_version=args.trapi_api_version,
                trapi_deployment=args.trapi_deployment,
            )
        )

    if args.json:
        out = [
            {
                "key": r.key,
                "episode": r.episode,
                "step": r.step,
                "final_quality": r.final_quality,
                "intermediate_scores": r.intermediate_scores,
                "explanation": r.explanation,
                "flags": r.flags,
            }
            for r in results
        ]
        print(json.dumps({
            "model": meta.model_name,
            "n_records": len(subset),
            "n_results": len(results),
            "results": out,
        }, indent=2))
    else:
        print(f"Model: {meta.model_name} Evaluated: {len(results)} (dry={args.dry})")
        for r in results:
            avg = (sum(r.intermediate_scores.values()) / len(r.intermediate_scores)) if r.intermediate_scores else 0.0
            print(f"[E{r.episode} S{r.step}] final={r.final_quality} avg={avg:.2f} key={r.key} expl={r.explanation[:50]}")

    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
