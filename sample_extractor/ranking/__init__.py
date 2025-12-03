"""
Ranking module package initializer.

Exports:
- EvaluationResult
- AzureGPTScorer
- TrapiGPTScorer
- step_hash
- build_evaluation_requests
- evaluate_records (auto-selects Azure vs TRAPI)
"""

from .schema import EvaluationResult  # noqa: F401
from .azure_gpt import AzureGPTScorer, TrapiGPTScorer, step_hash, build_evaluation_requests, evaluate_records  # noqa: F401
