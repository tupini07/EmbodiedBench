"""
Entrypoint for comparing two EmbodiedBench log files.

Usage:
    python -m sample_extractor.compare /path/to/logA.log /path/to/logB.log --limit 20 --json

Behavior:
- Parses both logs (in memory).
- Indexes successful planner steps by (episode, step).
- Finds intersection of keys present in both logs.
- Emits paired comparison records:
    {
      "episode": int,
      "step": int,
      "a": {
          "raw_output": str,
          "hash": str,
          "parseable": bool,
          "retries_before_success": int
      },
      "b": { ... same ... }
    }
- Optionally prints as JSON lines (--json) else human-readable summary.

Future extensions:
- Inject ranking scores via RankingModel (scalar or pairwise).
- Filter by episode range or step range.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

from .parser import parse_log, StepRecord
from .ranking import step_hash  # for stable key hashing


def _index(records: List[StepRecord]) -> Dict[Tuple[int, int], StepRecord]:
    """
    Index records by (episode, step). If duplicates occur, keep first occurrence.
    """
    idx: Dict[Tuple[int, int], StepRecord] = {}
    for r in records:
        key = (r.episode, r.step)
        if key not in idx:
            idx[key] = r
    return idx


def build_pairs(path_a: str, path_b: str) -> Tuple[List[Dict], int, int]:
    meta_a, recs_a = parse_log(path_a)
    meta_b, recs_b = parse_log(path_b)

    idx_a = _index(recs_a)
    idx_b = _index(recs_b)

    keys_intersection = sorted(set(idx_a.keys()) & set(idx_b.keys()), key=lambda k: (k[0], k[1]))

    pairs: List[Dict] = []
    for (ep, st) in keys_intersection:
        ra = idx_a[(ep, st)]
        rb = idx_b[(ep, st)]
        pairs.append({
            "episode": ep,
            "step": st,
            "a": {
                "raw_output": ra.raw_output,
                "hash": step_hash(ra),
                "parseable": ra.parseable,
                "retries_before_success": ra.retries_before_success,
            },
            "b": {
                "raw_output": rb.raw_output,
                "hash": step_hash(rb),
                "parseable": rb.parseable,
                "retries_before_success": rb.retries_before_success,
            },
            "meta": {
                "model_a": meta_a.model_name,
                "model_b": meta_b.model_name,
            }
        })

    return pairs, len(recs_a), len(recs_b)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare two EmbodiedBench log files.")
    parser.add_argument("log_a", help="Path to first .log file")
    parser.add_argument("log_b", help="Path to second .log file")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of pair records printed")
    parser.add_argument("--json", action="store_true", help="Emit JSON (list or JSON lines if --lines)")
    parser.add_argument("--lines", action="store_true", help="Emit JSON lines instead of single list when --json is set")
    args = parser.parse_args(argv)

    pairs, count_a, count_b = build_pairs(args.log_a, args.log_b)

    if args.json:
        if args.lines:
            for p in pairs[: args.limit]:
                print(json.dumps(p))
        else:
            print(json.dumps({
                "n_pairs": len(pairs),
                "parsed_steps_log_a": count_a,
                "parsed_steps_log_b": count_b,
                "pairs": pairs[: args.limit],
            }, indent=2))
    else:
        print(f"Parsed steps: log_a={count_a} log_b={count_b}")
        print(f"Intersected step keys: {len(pairs)}")
        for p in pairs[: args.limit]:
            print(f"[Episode {p['episode']} Step {p['step']}] "
                  f"A(len={len(p['a']['raw_output'])} hash={p['a']['hash']}) | "
                  f"B(len={len(p['b']['raw_output'])} hash={p['b']['hash']})")

        if len(pairs) > args.limit:
            print(f"... ({len(pairs) - args.limit} more pairs)")

        print("Use --json for structured output.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
