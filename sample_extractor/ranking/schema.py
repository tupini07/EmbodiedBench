"""
Ranking schema & cache utilities for EmbodiedBench reasoning evaluation.

Evaluation dimensions (0-10 ints):
- tightness
- task_relevance
- logical_progression
- fidelity
- efficiency
- insight
- structure
- robustness

Final quality: 1-10 int (holistic, not necessarily average).

JSON result shape produced by model:
{
  "intermediate_scores": {
      "tightness": int,
      "task_relevance": int,
      "logical_progression": int,
      "fidelity": int,
      "efficiency": int,
      "insight": int,
      "structure": int,
      "robustness": int
  },
  "final_quality": int,
  "explanation": "short string",
  "flags": {
      "possible_hallucination": bool,
      "overly_verbose": bool
  }
}

This module provides:
- EvaluationRequest: input assembled per StepRecord
- EvaluationResult: parsed model output + metadata
- RankCache: JSONL append-only cache keyed by hash(raw_output)

Cache file format (one JSON per line):
{
  "key": "...",
  "episode": int,
  "step": int,
  "raw_output_hash": "...",
  "result": {...model json...},
  "timestamp": 1731330000.123
}

"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Iterable, List


@dataclass
class EvaluationRequest:
    key: str
    episode: int
    step: int
    instruction: Optional[str]
    prompt: Optional[str]
    raw_output: str
    previous_raw_output: Optional[str]
    parseable: bool
    retries_before_success: int


@dataclass
class EvaluationResult:
    key: str
    episode: int
    step: int
    final_quality: int
    intermediate_scores: Dict[str, int]
    explanation: str
    flags: Dict[str, bool]
    raw_output_len: int
    cached: bool = False

    @property
    def average_intermediate(self) -> float:
        if not self.intermediate_scores:
            return 0.0
        return sum(self.intermediate_scores.values()) / len(self.intermediate_scores)


# ----------------------
# Cache handling
# ----------------------

@dataclass
class RankCache:
    path: str
    _data: Dict[str, EvaluationResult] = field(default_factory=dict)
    _dirty: bool = False

    @classmethod
    def load(cls, path: str) -> "RankCache":
        cache = cls(path=path)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        result_obj = obj.get("result", {})
                        cache._data[obj["key"]] = EvaluationResult(
                            key=obj["key"],
                            episode=obj.get("episode", -1),
                            step=obj.get("step", -1),
                            final_quality=result_obj.get("final_quality", 0),
                            intermediate_scores=result_obj.get("intermediate_scores", {}),
                            explanation=result_obj.get("explanation", ""),
                            flags=result_obj.get("flags", {}),
                            raw_output_len=result_obj.get("raw_output_len", 0),
                            cached=True,
                        )
                    except Exception:
                        # Skip malformed line
                        continue
        return cache

    def get(self, key: str) -> Optional[EvaluationResult]:
        return self._data.get(key)

    def add(self, result: EvaluationResult) -> None:
        # If already cached with final_quality we do not overwrite unless new is better defined
        existing = self._data.get(result.key)
        if existing and existing.final_quality and result.final_quality == 0:
            return
        self._data[result.key] = result
        self._dirty = True
        self._append_line(result)

    def _append_line(self, result: EvaluationResult) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {
            "key": result.key,
            "episode": result.episode,
            "step": result.step,
            "raw_output_hash": result.key,
            "timestamp": time.time(),
            "result": {
                "final_quality": result.final_quality,
                "intermediate_scores": result.intermediate_scores,
                "explanation": result.explanation,
                "flags": result.flags,
                "raw_output_len": result.raw_output_len,
            },
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def all(self) -> Iterable[EvaluationResult]:
        return self._data.values()


# ----------------------
# Utility
# ----------------------

DIMENSIONS = [
    "tightness",
    "task_relevance",
    "logical_progression",
    "fidelity",
    "efficiency",
    "insight",
    "structure",
    "robustness",
]


def empty_result(key: str, episode: int, step: int, raw_len: int) -> EvaluationResult:
    return EvaluationResult(
        key=key,
        episode=episode,
        step=step,
        final_quality=0,
        intermediate_scores={d: 0 for d in DIMENSIONS},
        explanation="",
        flags={"possible_hallucination": False, "overly_verbose": False},
        raw_output_len=raw_len,
        cached=False,
    )


def parse_model_json(
    key: str,
    episode: int,
    step: int,
    raw_output_len: int,
    data: Dict[str, Any],
) -> EvaluationResult:
    inter = data.get("intermediate_scores") or {}
    final_quality = int(data.get("final_quality", 0))
    explanation = str(data.get("explanation", ""))[:400]
    flags = data.get("flags") or {}
    # Ensure dimension coverage
    for dim in DIMENSIONS:
        if dim not in inter:
            inter[dim] = 0
        else:
            inter[dim] = int(inter[dim])
    return EvaluationResult(
        key=key,
        episode=episode,
        step=step,
        final_quality=final_quality,
        intermediate_scores=inter,
        explanation=explanation,
        flags={
            "possible_hallucination": bool(flags.get("possible_hallucination", False)),
            "overly_verbose": bool(flags.get("overly_verbose", False)),
        },
        raw_output_len=raw_output_len,
        cached=False,
    )
