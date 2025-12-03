"""
Ranking model abstraction + Azure GPT reasoning quality evaluator for EmbodiedBench.

Implements:
- step_hash(raw_output) for stable key
- AzureGPTScorer: async scalar reasoning evaluator producing structured scores
- build_evaluation_request: create EvaluationRequest objects with previous context
- evaluate_many: high-level helper
- Prompt template enforcing strict JSON output with required dimensions (see schema.py)

Hardcoded Azure resource list (round-robin):
[
    "search-learn-2",
    "west-us-3-aoai",
    "sweden-aoai-3428",
    "japan-aoai-423985",
    "west-us-3-aoai-9058433",
    "japan-aoai-effai-543987",
    "sweden-aoai-effai-23457954",
]

Environment variables (optional overrides):
- AZURE_OPENAI_RESOURCE_NAMES (comma separated)
- AZURE_OPENAI_API_VERSION (default 2025-04-01-preview)
- DEFAULT_IDENTITY_CLIENT_ID (for managed identity)
- RANK_CACHE_PATH (default outputs/rank_cache.jsonl)
- RANK_MAX_OUTPUT_TOKENS (default 512)
- RANK_CONCURRENCY (default 30)

Dependencies: azure-identity, openai>=1.0.0 (AzureOpenAI client).

If credentials/resources missing, scorer returns empty EvaluationResult without network call.

Retry strategy: up to 3 attempts with exponential backoff (1s, 2s, 4s) on recoverable errors.
JSON repair: second prompt attempt if initial output invalid.

NOTE: Instruction text per episode not currently stored in StepRecord. For now instruction passed as None.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from azure.identity import AzureCliCredential, DefaultAzureCredential, ChainedTokenCredential
from openai import AzureOpenAI, OpenAI

from sample_extractor.parser import StepRecord
from .schema import (
    EvaluationRequest,
    EvaluationResult,
    RankCache,
    parse_model_json,
    empty_result,
    DIMENSIONS,
)

# -------------------------
# Config / constants
# -------------------------

_HARDCODED_RESOURCES = [
    "search-learn-2",
    "west-us-3-aoai",
    "sweden-aoai-3428",
    "japan-aoai-423985",
    "west-us-3-aoai-9058433",
    "japan-aoai-effai-543987",
    "sweden-aoai-effai-23457954",
]

DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_CACHE_PATH = "outputs/rank_cache.jsonl"
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("RANK_MAX_OUTPUT_TOKENS", "512"))
DEFAULT_CONCURRENCY = int(os.getenv("RANK_CONCURRENCY", "30"))

SYSTEM_PROMPT = """You are an expert evaluator of embodied AI reasoning quality.
Return STRICT JSON ONLY. No prose outside JSON.
Scoring rules (0-10 integers for intermediate dimensions, 1-10 final_quality):

Definitions:
- tightness: minimal irrelevant or repetitive content.
- task_relevance: reasoning aligns with the instruction and immediate goal.
- logical_progression: clear, coherent steps advancing toward goal.
- fidelity: accurate references to environment/state; avoid hallucinations.
- efficiency: concise plan avoids redundancy.
- insight: presence of non-trivial inference or clever simplification.
- structure: clear separation of reasoning vs plan, formatting coherence.
- robustness: acknowledges prior failures / retries or potential errors sensibly.

Flags:
- possible_hallucination: True if content references entities obviously absent or contradicts given context.
- overly_verbose: True if reasoning contains large filler sections without advancing logic.

Final_quality: holistic expert judgment; not forced to be average. Justify divergence.

Output schema (single line JSON):
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
  "explanation": "short justification (<60 words)",
  "flags": {
    "possible_hallucination": bool,
    "overly_verbose": bool
  }
}
If any dimension not assessable, set 0. Explanation MUST be <= 60 words.
"""

USER_TEMPLATE = """EVALUATION_INPUT:

```
%(raw_output)s
```

The raw_output above is the complete response from an embodied AI model for this step.
Evaluate the quality of the reasoning, plan formulation, and overall output.

Return ONLY the specified JSON object—no additional text.
"""


# -------------------------
# Hash helper
# -------------------------

import hashlib


def step_hash(record: StepRecord) -> str:
    """
    Stable short hash of raw_output only (model generation text).
    """
    h = hashlib.sha256()
    h.update(record.raw_output.encode("utf-8"))
    return h.hexdigest()[:16]


# -------------------------
# Request assembly
# -------------------------

def build_evaluation_requests(records: Sequence[StepRecord]) -> List[EvaluationRequest]:
    # Build previous step mapping by (episode, step)
    by_episode: Dict[int, List[StepRecord]] = {}
    for r in records:
        by_episode.setdefault(r.episode, []).append(r)
    # Sort within each
    for ep in by_episode:
        by_episode[ep].sort(key=lambda x: x.step)

    prev_map: Dict[str, str] = {}
    for ep, lst in by_episode.items():
        for i, r in enumerate(lst):
            if i > 0:
                prev_map[f"{ep}:{r.step}"] = lst[i - 1].raw_output

    reqs: List[EvaluationRequest] = []
    for r in records:
        key = step_hash(r)
        prev = prev_map.get(f"{r.episode}:{r.step}")
        reqs.append(
            EvaluationRequest(
                key=key,
                episode=r.episode,
                step=r.step,
                instruction=None,  # Not captured yet
                prompt=r.prompt,
                raw_output=r.raw_output,
                previous_raw_output=prev,
                parseable=r.parseable,
                retries_before_success=r.retries_before_success,
            )
        )
    return reqs


# -------------------------
# Azure credential helpers
# -------------------------

def _azure_api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)


def _build_chained_credential():
    return ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(
            exclude_cli_credential=True,
            exclude_environment_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_interactive_browser_credential=True,
            exclude_visual_studio_code_credentials=True,
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        ),
    )


def _get_scope() -> str:
    return "https://cognitiveservices.azure.com/.default"


def _default_azure_resource_list() -> List[str]:
    return list(_HARDCODED_RESOURCES)


def _strip_split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _azure_endpoint(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/"


# -------------------------
# Scorer
# -------------------------

@dataclass
class AzureResourceState:
    resource: str
    cooldown_until: float = 0.0  # epoch timestamp while in cooldown


class AzureGPTScorer:
    """
    Asynchronous evaluator producing EvaluationResult objects.

    Uses round-robin resource rotation; skips resources in cooldown
    after repeated failures (HTTP 429/5xx).
    """

    def __init__(
        self,
        deployment_name: Optional[str] = None,
        cache_path: str = DEFAULT_CACHE_PATH,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        concurrency: int = DEFAULT_CONCURRENCY,
    ):
        # Deployment names (Azure side) – user stated single name 'gp4o'
        env_multi = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAMES")
        env_single = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        base = deployment_name or (env_multi or env_single or "gpt-4o")
        if env_multi:
            self._deployment_names = [x.strip() for x in env_multi.split(",") if x.strip()]
        else:
            self._deployment_names = [base]
        self._dep_rr_index = 0

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(concurrency)

        resources_env = os.getenv("AZURE_OPENAI_RESOURCE_NAMES")
        resources = (
            _strip_split_csv(resources_env) if resources_env else _default_azure_resource_list()
        )
        if not resources:
            logging.warning("No Azure resources configured; scorer will return empty results.")
        self._resource_states: List[AzureResourceState] = [AzureResourceState(r) for r in resources]

        self._api_version = _azure_api_version()
        self._credential = _build_chained_credential()
        scope = _get_scope()
        from azure.identity import get_bearer_token_provider

        self._token_provider = get_bearer_token_provider(self._credential, scope)
        self._client_cache: Dict[str, AzureOpenAI] = {}
        self._rr_index = 0
        self._lock = asyncio.Lock()

        self.cache = RankCache.load(cache_path)

    def _choose_deployment(self) -> str:
        name = self._deployment_names[self._dep_rr_index % len(self._deployment_names)]
        self._dep_rr_index += 1
        return name

    def _choose_active_resource(self) -> Optional[str]:
        now = time.time()
        # Iterate resources starting from rr index
        for _ in range(len(self._resource_states)):
            idx = self._rr_index % len(self._resource_states)
            self._rr_index += 1
            state = self._resource_states[idx]
            if now >= state.cooldown_until:
                return state.resource
        return None

    def _client_for(self, resource: str) -> AzureOpenAI:
        client = self._client_cache.get(resource)
        if client is None:
            client = AzureOpenAI(
                api_version=self._api_version,
                azure_endpoint=_azure_endpoint(resource),
                azure_ad_token_provider=self._token_provider,
            )
            self._client_cache[resource] = client
        return client

    async def score_request(self, req: EvaluationRequest) -> EvaluationResult:
        # Cache check
        cached = self.cache.get(req.key)
        if cached:
            return cached

        if not self._resource_states:
            return empty_result(req.key, req.episode, req.step, len(req.raw_output))

        user_payload = USER_TEMPLATE % {
            "raw_output": req.raw_output,
        }

        attempt = 0
        last_error = None
        while attempt < 3:
            resource = self._choose_active_resource()
            if resource is None:
                # All in cooldown
                await asyncio.sleep(2.0)
                attempt += 1
                continue
            client = self._client_for(resource)
            try:
                deployment = self._choose_deployment()
                completion = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=deployment,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_payload},
                        ],
                        response_format={"type": "json_object"},
                    ),
                )
                text_raw = completion.choices[0].message.content or ""
                text = text_raw.strip()
                try:
                    data = json.loads(text)
                except Exception:
                    # Single repair attempt
                    repair_prompt = f"Output invalid JSON. Provide ONLY valid JSON per schema. Original text:\n{text}"
                    deployment = self._choose_deployment()
                    completion2 = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.chat.completions.create(
                            model=deployment,
                            temperature=0,
                            max_tokens=self.max_output_tokens,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_payload},
                                {"role": "user", "content": repair_prompt},
                            ],
                            response_format={"type": "json_object"},
                        ),
                    )
                    text_raw = completion2.choices[0].message.content or ""
                    text = text_raw.strip()
                    data = json.loads(text)
                result = parse_model_json(
                    key=req.key,
                    episode=req.episode,
                    step=req.step,
                    raw_output_len=len(req.raw_output),
                    data=data,
                )
                self.cache.add(result)
                return result
            except Exception as e:
                last_error = e
                # Mark resource cooldown if 429/5xx pattern in str(e)
                err_text = str(e)
                if any(code in err_text for code in ["404", "429", "500", "502", "503", "504"]):
                    # cooldown 30s
                    for state in self._resource_states:
                        if state.resource == resource:
                            state.cooldown_until = time.time() + 30
                            break
                backoff = (2 ** attempt)
                await asyncio.sleep(backoff)
                attempt += 1

        logging.warning(f"Failed to score key={req.key}: {last_error}")
        return empty_result(req.key, req.episode, req.step, len(req.raw_output))

    async def score_many(self, requests: Sequence[EvaluationRequest]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        async def _run(req: EvaluationRequest):
            async with self.semaphore:
                return await self.score_request(req)

        tasks = [asyncio.create_task(_run(r)) for r in requests]
        for t in tasks:
            results.append(await t)
        return results


# -------------------------
# TRAPI / GPT-5 scorer (non-Azure direct OpenAI client)
# -------------------------

class TrapiGPTScorer:
    """
    Direct OpenAI (TRAPI) scorer for models like gpt-5 when Azure deployments are not used.
    Uses the same prompt and parsing logic; skips resource rotation.
    Environment:
      OPENAI_API_KEY must be set OR pass api_key explicitly.
      RANK_MAX_OUTPUT_TOKENS / RANK_CONCURRENCY respected.
    """

    def __init__(
        self,
        model_name: str = "gpt-5",
        cache_path: str = DEFAULT_CACHE_PATH,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        concurrency: int = DEFAULT_CONCURRENCY,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(concurrency)
        # Allow explicit api_key override to avoid env requirement
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.cache = RankCache.load(cache_path)

    async def score_request(self, req: EvaluationRequest) -> EvaluationResult:
        cached = self.cache.get(req.key)
        if cached:
            return cached

        user_payload = USER_TEMPLATE % {
            "raw_output": req.raw_output,
        }

        try:
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_output_tokens,
                        messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format={"type": "json_object"},
                ),
            )
            text_raw = completion.choices[0].message.content or ""
            text = text_raw.strip()
            data = json.loads(text)
            result = parse_model_json(
                key=req.key,
                episode=req.episode,
                step=req.step,
                raw_output_len=len(req.raw_output),
                data=data,
            )
            self.cache.add(result)
            return result
        except Exception as e:
            logging.warning(f"TRAPI scoring failed key={req.key}: {e}")
            return empty_result(req.key, req.episode, req.step, len(req.raw_output))

    async def score_many(self, requests: Sequence[EvaluationRequest]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        async def _run(req: EvaluationRequest):
            async with self.semaphore:
                return await self.score_request(req)

        tasks = [asyncio.create_task(_run(r)) for r in requests]
        for t in tasks:
            results.append(await t)
        return results

# -------------------------
# TRAPI (Azure identity) scorer for dedicated TRAPI endpoint
# -------------------------

class TrapiAzureIdentityScorer:
    """
    Azure identity backed scorer for TRAPI endpoint (no standard Azure OpenAI deployment rotation).
    Uses Azure CLI / Managed Identity credentials to obtain token for scope api://trapi/.default.

    Parameters:
      instance: e.g. 'gcr/internal'
      deployment: e.g. 'gpt-5_2025-08-07'
      api_version: e.g. '2024-12-01-preview'
    """
    def __init__(
        self,
        instance: str,
        deployment: str,
        api_version: str,
        cache_path: str = DEFAULT_CACHE_PATH,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        concurrency: int = DEFAULT_CONCURRENCY,
    ):
        from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider
        self.instance = instance.strip("/")
        self.deployment = deployment
        self.api_version = api_version
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(concurrency)
        cred = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        scope = "api://trapi/.default"
        token_provider = get_bearer_token_provider(cred, scope)
        endpoint = f"https://trapi.research.microsoft.com/{self.instance}"
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )
        self.cache = RankCache.load(cache_path)

    async def score_request(self, req: EvaluationRequest) -> EvaluationResult:
        cached = self.cache.get(req.key)
        if cached:
            return cached

        user_payload = USER_TEMPLATE % {
            "raw_output": req.raw_output,
        }
        try:
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                        model=self.deployment,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_output_tokens,
                        messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format={"type": "json_object"},
                ),
            )
            text_raw = completion.choices[0].message.content or ""
            data = json.loads(text_raw.strip())
            result = parse_model_json(
                key=req.key,
                episode=req.episode,
                step=req.step,
                raw_output_len=len(req.raw_output),
                data=data,
            )
            self.cache.add(result)
            return result
        except Exception as e:
            logging.warning(f"TRAPI Azure scorer failed key={req.key}: {e}")
            return empty_result(req.key, req.episode, req.step, len(req.raw_output))

    async def score_many(self, requests: Sequence[EvaluationRequest]) -> List[EvaluationResult]:
        async def _run(r: EvaluationRequest):
            async with self.semaphore:
                return await self.score_request(r)
        return [await t for t in [asyncio.create_task(_run(r)) for r in requests]]

# -------------------------
# High-level helper
# -------------------------

async def evaluate_records(
    records: Sequence[StepRecord],
    use_trapi: bool = False,
    trapi_model: str = "gpt-5",
    trapi_instance: Optional[str] = None,
    trapi_api_version: Optional[str] = None,
    trapi_deployment: Optional[str] = None,
    trapi_api_key: Optional[str] = None,
) -> List[EvaluationResult]:
    """
    Evaluate a sequence of StepRecord objects.

    Auto-selects TRAPI scorer when:
      - use_trapi=True
      - OR env RANK_FORCE_TRAPI=1
      - OR trapi_model startswith 'gpt-5'

    Otherwise uses AzureGPTScorer (Azure rotation).
    """
    force_trapi_env = os.getenv("RANK_FORCE_TRAPI", "0") == "1"
    if trapi_api_key is None:
        trapi_api_key = os.getenv("OPENAI_API_KEY")
    if trapi_instance and trapi_api_version and trapi_deployment:
        scorer = TrapiAzureIdentityScorer(
            instance=trapi_instance,
            deployment=trapi_deployment,
            api_version=trapi_api_version,
        )
    elif use_trapi or force_trapi_env or trapi_model.startswith("gpt-5"):
        scorer = TrapiGPTScorer(model_name=trapi_model, api_key=trapi_api_key)
    else:
        scorer = AzureGPTScorer()
    reqs = build_evaluation_requests(records)
    return await scorer.score_many(reqs)


# -------------------------
# Simple CLI (optional import in rank.py)
# -------------------------

def _cli(argv: List[str]) -> int:
    import argparse
    from sample_extractor.parser import parse_log

    parser = argparse.ArgumentParser(description="Evaluate reasoning quality for a log file.")
    parser.add_argument("logpath")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    meta, records = parse_log(args.logpath)
    subset = records[: args.limit]

    scorer = AzureGPTScorer(concurrency=args.concurrency)
    reqs = build_evaluation_requests(subset)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(scorer.score_many(reqs))

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
        print(json.dumps(out, indent=2))
    else:
        print(f"Model: {meta.model_name} Steps evaluated: {len(results)}")
        for r in results:
            print(
                f"[E{r.episode} S{r.step}] final={r.final_quality} "
                f"avg={r.average_intermediate:.2f} tight={r.intermediate_scores['tightness']} "
                f"insight={r.intermediate_scores['insight']} flags={r.flags} key={r.key}"
            )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))
