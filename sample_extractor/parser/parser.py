"""
EmbodiedBench log parser.

Parses model run logs to extract successful planner step records.

Log anatomy (observed):
- Header/meta block until a separator line of only dashes.
- Repeated planning attempts:
  Input to model:
    (prompt block)
  Raw output from model:
    (model output block)
  Post-parse candidate (is_json_parseable=...):
  Either:
    Replanning due to invalid or empty plan. Remaining retries: N   (failure)

We ONLY record successful attempts (presence of Planner Output Action line).
A parseable=True candidate can still be rejected if followed by a Replanning line
instead of a Planner Output Action line.

Data schema:
  RunMeta
  StepRecord

Public API:
  parse_log(path) -> (RunMeta, List[StepRecord])
  iter_step_records(path) -> Iterator[StepRecord]

CLI:
  python -m sample_extractor.parser.parser <logpath> [--json]
Outputs either repr() or JSON (one JSON object per line).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union


# -----------------------
# Data classes
# -----------------------

ActionType = Union[int, List[int]]


@dataclass
class StepRecord:
    episode: int
    step: int
    prompt: Optional[str]
    raw_output: str
    parseable: bool
    retries_before_success: int
    attempt_index: int
    log_path: str


@dataclass
class RunMeta:
    model_name: Optional[str]
    eval_sets: List[str]
    exp_name: Optional[str]
    raw_header: List[str]


# -----------------------
# Regex patterns
# -----------------------

RE_SEPARATOR = re.compile(r"^-{5,}\s*$")
RE_INPUT = re.compile(r"^Input to model:\s*$")
RE_RAW = re.compile(r"^Raw output from model:\s*$")
RE_POST_PARSE = re.compile(r"^Post-parse candidate \(is_json_parseable=(True|False)\):\s*$")
RE_REPLAN = re.compile(r"^Replanning due to invalid or empty plan\. Remaining retries: \d+\s*$")
RE_PLANNER = re.compile(r"^\[Episode:(\d+);Step:(\d+)\] Planner Output Action: (.+)$")


# -----------------------
# Parsing helpers
# -----------------------

def _extract_meta(header_lines: List[str]) -> RunMeta:
    # Attempt to locate the first line that looks like a Python dict literal
    model_name = None
    eval_sets: List[str] = []
    exp_name = None
    for line in header_lines:
        line_stripped = line.strip()
        if line_stripped.startswith("{") and line_stripped.endswith("}"):
            try:
                meta_dict = ast.literal_eval(line_stripped)
                model_name = meta_dict.get("model_name")
                eval_sets = list(meta_dict.get("eval_sets", []))
                exp_name = meta_dict.get("exp_name")
                break
            except Exception:
                continue
    return RunMeta(
        model_name=model_name,
        eval_sets=eval_sets,
        exp_name=exp_name,
        raw_header=header_lines,
    )


# -----------------------
# Core parsing generator
# -----------------------

def iter_step_records(path: str) -> Iterator[StepRecord]:
    """
    Stream over successful planner step records.
    """
    attempt_index = 0
    retries_since_last_success = 0

    in_header = True
    header_lines: List[str] = []

    in_prompt = False
    prompt_lines: List[str] = []

    in_raw_output = False
    raw_lines: List[str] = []
    current_parseable: Optional[bool] = None

    pending_attempt_has_candidate = False  # we saw a Post-parse line
    post_parse_happened = False

    # We create run meta once we've finished header; meta needed only for overall return,
    # but streaming records do not require meta fields (except log_path saved per record).
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Header accumulation
            if in_header:
                if RE_SEPARATOR.match(line):
                    in_header = False
                else:
                    header_lines.append(line.rstrip("\n"))
                    continue

            # Detect phases
            if RE_INPUT.match(line):
                # Reset attempt buffers for a new attempt
                in_prompt = True
                in_raw_output = False
                prompt_lines = []
                raw_lines = []
                current_parseable = None
                pending_attempt_has_candidate = False
                post_parse_happened = False
                continue

            if in_prompt and RE_RAW.match(line):
                in_prompt = False
                in_raw_output = True
                raw_lines = []
                continue

            m_post = RE_POST_PARSE.match(line)
            if in_raw_output and m_post:
                in_raw_output = False
                current_parseable = m_post.group(1) == "True"
                pending_attempt_has_candidate = True
                post_parse_happened = True
                continue

            # Failure line
            if pending_attempt_has_candidate and RE_REPLAN.match(line):
                # Attempt discarded
                attempt_index += 1
                retries_since_last_success += 1
                # Reset state for next attempt
                pending_attempt_has_candidate = False
                continue

            # Success line (planner output)
            m_plan = RE_PLANNER.match(line)
            if pending_attempt_has_candidate and m_plan:
                episode = int(m_plan.group(1))
                step = int(m_plan.group(2))

                record = StepRecord(
                    episode=episode,
                    step=step,
                    prompt="\n".join(prompt_lines).split("---------------------------------------------------")[0].strip() if prompt_lines else None,
                    raw_output="\n".join(raw_lines).split("---------------------------------------------------")[0].strip(),
                    parseable=bool(current_parseable),
                    retries_before_success=retries_since_last_success,
                    attempt_index=attempt_index,
                    log_path=path,
                )
                yield record

                attempt_index += 1
                retries_since_last_success = 0
                pending_attempt_has_candidate = False
                continue

            # Accumulate prompt lines
            if in_prompt:
                prompt_lines.append(line.rstrip("\n"))
                continue

            # Accumulate raw output lines
            if in_raw_output:
                raw_lines.append(line.rstrip("\n"))
                continue

    # End of file: nothing special to flush because incomplete attempts are ignored.


def parse_log(path: str) -> tuple[RunMeta, List[StepRecord]]:
    """
    Convenience function: fully parse and return meta + all step records.
    """
    header_lines: List[str] = []
    records: List[StepRecord] = []

    # First pass to collect header + records using generator (to avoid duplicate header parse,
    # we replicate a small portion of logic here).
    attempt_index = 0
    retries_since_last_success = 0
    in_header = True

    in_prompt = False
    prompt_lines: List[str] = []
    in_raw_output = False
    raw_lines: List[str] = []
    current_parseable: Optional[bool] = None
    pending_attempt_has_candidate = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if in_header:
                if RE_SEPARATOR.match(line):
                    in_header = False
                else:
                    header_lines.append(line.rstrip("\n"))
                    continue

            if RE_INPUT.match(line):
                in_prompt = True
                in_raw_output = False
                prompt_lines = []
                raw_lines = []
                current_parseable = None
                pending_attempt_has_candidate = False
                continue

            if in_prompt and RE_RAW.match(line):
                in_prompt = False
                in_raw_output = True
                raw_lines = []
                continue

            m_post = RE_POST_PARSE.match(line)
            if in_raw_output and m_post:
                in_raw_output = False
                current_parseable = m_post.group(1) == "True"
                pending_attempt_has_candidate = True
                continue

            if pending_attempt_has_candidate and RE_REPLAN.match(line):
                attempt_index += 1
                retries_since_last_success += 1
                pending_attempt_has_candidate = False
                continue

            m_plan = RE_PLANNER.match(line)
            if pending_attempt_has_candidate and m_plan:
                episode = int(m_plan.group(1))
                step = int(m_plan.group(2))
                record = StepRecord(
                    episode=episode,
                    step=step,
                    prompt="\n".join(prompt_lines).split("---------------------------------------------------")[0].strip() if prompt_lines else None,
                    raw_output="\n".join(raw_lines).split("---------------------------------------------------")[0].strip(),
                    parseable=bool(current_parseable),
                    retries_before_success=retries_since_last_success,
                    attempt_index=attempt_index,
                    log_path=path,
                )
                records.append(record)
                attempt_index += 1
                retries_since_last_success = 0
                pending_attempt_has_candidate = False
                continue

            if in_prompt:
                prompt_lines.append(line.rstrip("\n"))
                continue
            if in_raw_output:
                raw_lines.append(line.rstrip("\n"))
                continue

    meta = _extract_meta(header_lines)
    return meta, records


# -----------------------
# CLI Support
# -----------------------

def _cli(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse EmbodiedBench log.")
    parser.add_argument("logpath", help="Path to .log file")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines of StepRecord")
    parser.add_argument("--meta", action="store_true", help="Print run meta as JSON first")
    args = parser.parse_args(argv)

    meta, records = parse_log(args.logpath)

    if args.meta:
        print(json.dumps({
            "model_name": meta.model_name,
            "eval_sets": meta.eval_sets,
            "exp_name": meta.exp_name,
            "header_lines": meta.raw_header,
        }))

    if args.json:
        for r in records:
            print(json.dumps({
                "episode": r.episode,
                "step": r.step,
                "prompt": r.prompt,
                "raw_output": r.raw_output,
                "parseable": r.parseable,
                "retries_before_success": r.retries_before_success,
                "attempt_index": r.attempt_index,
                "log_path": r.log_path,
            }))
            break
    else:
        print(f"Parsed {len(records)} successful step records")
        for r in records[:10]:  # show sample
            print(r)
        if len(records) > 10:
            print(f"... ({len(records) - 10} more)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli(sys.argv[1:]))
