import base64
import json
import logging
import os
import random
import sys
import threading
from typing import Union

import anthropic
import google.generativeai as genai
import lmdeploy
import requests
import typing_extensions as typing
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from openai import AzureOpenAI, OpenAI

from embodiedbench.utils.duration_logger import DurationLogger
import time


# ---------------------------
# Azure helper functions (used for GPT model rotation)
# ---------------------------
def _default_azure_resource_list():
    return [
        "search-learn-2",
        "west-us-3-aoai",
        "sweden-aoai-3428",
        "japan-aoai-423985",
        "west-us-3-aoai-9058433",
        "japan-aoai-effai-543987",
        "sweden-aoai-effai-23457954",
    ]


def _strip_split_csv(val: str):
    return [p.strip() for p in val.split(",") if p.strip()]


def _azure_api_version():
    return os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")


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


def _azure_endpoint(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/"


def _get_scope():
    return "https://cognitiveservices.azure.com/.default"


from embodiedbench.planner.plan_validator import maybe_repair
from embodiedbench.planner.planner_config.generation_guide import (
    llm_generation_guide,
    vlm_generation_guide,
)
from embodiedbench.planner.planner_config.generation_guide_manip import (
    llm_generation_guide_manip,
    vlm_generation_guide_manip,
)
from embodiedbench.planner.planner_utils import (
    extract_box_json,
)  # legacy first-box extractor (kept for BC)
from embodiedbench.planner.planner_utils import (
    extract_first_json_object,
)  # fallback when no boxing present
from embodiedbench.planner.planner_utils import (
    extract_last_box_json,
)  # new robust final-box extractor
from embodiedbench.planner.planner_utils import (
    sanitize_box_payload,
)  # new sanitation helper
from embodiedbench.planner.planner_utils import (
    ActionPlan,
    ActionPlan_1,
    ActionPlan_1_manip,
    ActionPlan_lang,
    ActionPlan_lang_manip,
    ActionPlan_manip,
    convert_format_2claude,
    convert_format_2gemini,
    fix_json,
)

"""Remote model abstraction.

Reasoning Mode Scope:
---------------------
The environment flag EMB_REASONING_MODE=1 and optional REMOTE_MODEL_STOP_SEQS
are intentionally applied ONLY to Qwen 7B / 2.5-VL-7B style calls (handled in
`_call_qwen7b`). Other provider paths (GPT/Azure, Claude, Gemini, Llama, InternVL,
Fireworks) retain their original JSON schema response formatting without any
reasoning-mode alteration to keep upstream compatibility.

Minimal-change note: This file preserves original structure; only lightweight
comments and boxed-JSON post-processing for Qwen reasoning mode were added.
"""

temperature = float(os.environ.get("REMOTE_MODEL_TEMPERATURE", 0.0))
max_completion_tokens = int(os.environ.get("REMOTE_MODEL_MAX_TOKENS", 2048))
remote_url = os.environ.get("remote_url", "").split(",")
if not isinstance(remote_url, list):
    remote_url = [remote_url]

reasoning_mode = os.getenv("EMB_REASONING_MODE", "0") == "1"
_stop_env = os.getenv("REMOTE_MODEL_STOP_SEQS", "</answer>")
stop_seqs = (
    [s.strip() for s in _stop_env.split(",") if s.strip()] if reasoning_mode else None
)

vllm_frequency_penalty = os.getenv("VLLM_FREQUENCY_PENALTY", None)
vllm_top_p = os.getenv("VLLM_TOP_P", None)


print(f"REMOTE_MODEL_TEMPERATURE: {temperature}")
print(f"REMOTE_MODEL_MAX_TOKENS: {max_completion_tokens}")
print(f"REMOTE_MODEL_URL: {remote_url}")
print(f"EMB_REASONING_MODE: {reasoning_mode}")
print(f"REMOTE_MODEL_STOP_SEQS: {stop_seqs}")
print(f"VLLM_FREQUENCY_PENALTY: {vllm_frequency_penalty}")
print(f"VLLM_TOP_P: {vllm_top_p}")

if len(remote_url) > 0:
    print(f"Using remote model URL(s): {remote_url}")
    all_known_models = []
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for url in remote_url:
        success = False
        for attempt in range(max_retries):
            try:
                # check /models for every url, and print it out so it stays in the logs
                model_data = requests.get(f"{url}/models", timeout=10).json()
                print(f"Model data from {url}:\n{model_data}")
                all_known_models.append(model_data["data"][0]["root"])
                success = True
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error querying model name from {url} after {max_retries} attempts: {e}")
        
        if not success:
            print(f"Failed to connect to {url} after all retry attempts")
    
    assert len(all_known_models) == len(remote_url), f"All remote_url endpoints must point to a valid model, but some failed."

    known_model_names = set(all_known_models)

    if len(known_model_names) == 0:
        raise RuntimeError(
            f"Failed to connect to any remote_url endpoints. "
            f"Please ensure the model servers are running and accessible at: {remote_url}"
        )

    print(f"Detected model name from remote_url:\n\t{list(known_model_names)[0]}")

    expected_model_path = os.environ.get("EXPECTED_MODEL_PATH", "")
    if expected_model_path:
        print(
            f"EXPECTED_MODEL_PATH is set to: {expected_model_path}, verifying against detected model name..."
        )
        current_model_path = list(known_model_names)[0]

        assert (
            current_model_path.rstrip("/") == expected_model_path.rstrip("/")
        ), f"Model path mismatch! Expected to find '{expected_model_path}' in '{current_model_path}'"

        print(f"Model path verified successfully.")
    

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type="remote",
        language_only=False,
        tp=1,
        task_type=None,  # used to distinguish between manipulation and other environments
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type

        self.model: OpenAI | list[OpenAI] | lmdeploy.Pipeline = None  # type: ignore
        self._model_rotation_index = 0

        if isinstance(remote_url, list):
            # spread requests for different processes to avoid swamping the same URL all the time.
            self._model_rotation_index = random.randint(0, len(remote_url) - 1)

        if self.model_type == "local":
            backend_config = PytorchEngineConfig(
                session_len=12000, dtype="float16", tp=tp
            )
            self.model = pipeline(self.model_name, backend_config=backend_config)
        else:
            if "claude" in self.model_name:
                self.model = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            elif "gemini" in self.model_name:
                self.model = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
            elif "gpt" in self.model_name:
                # Always attempt Azure rotation for GPT models.
                # Env Vars:
                #   AZURE_OPENAI_RESOURCE_NAMES: comma separated resource names (e.g. "res1,res2,res3")
                #   AZURE_OPENAI_API_VERSION: optional override (default 2025-04-01-preview)
                #   DEFAULT_IDENTITY_CLIENT_ID: optional managed identity client id
                if self._init_azure_rotation():
                    logging.info(
                        "Azure OpenAI rotation enabled for GPT model: resources=%s",
                        self._azure_resources,
                    )
                else:
                    # Fallback: use standard public OpenAI client if rotation config invalid.
                    self.model = OpenAI()
            elif "qwen" in self.model_name:
                self.model = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif "Qwen2-VL" in self.model_name:
                print(f"Initializing Qwen2-VL model with remote URL: {remote_url}")
                if isinstance(remote_url, list):
                    self.model = [OpenAI(base_url=url) for url in remote_url]
                else:
                    self.model = OpenAI(base_url=remote_url)
            elif "Qwen2.5-VL" in self.model_name:
                print(f"Initializing Qwen2.5-VL model with remote URL: {remote_url}")
                if isinstance(remote_url, list):
                    self.model = [OpenAI(base_url=url) for url in remote_url]
                else:
                    self.model = OpenAI(base_url=remote_url)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                if isinstance(remote_url, list):
                    self.model = [OpenAI(base_url=url) for url in remote_url]
                else:
                    self.model = OpenAI(base_url=remote_url)
            elif "OpenGVLab/InternVL" in self.model_name:
                if isinstance(remote_url, list):
                    self.model = [OpenAI(base_url=url) for url in remote_url]
                else:
                    self.model = OpenAI(base_url=remote_url)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                if isinstance(remote_url, list):
                    self.model = [OpenAI(base_url=url) for url in remote_url]
                else:
                    self.model = OpenAI(base_url=remote_url)
            elif (
                "90b-vision-instruct" in self.model_name
            ):  # you can use fireworks to inference
                self.model = OpenAI(
                    base_url="https://api.fireworks.ai/inference/v1",
                    api_key=os.environ.get("firework_API_KEY"),
                )
            else:
                try:
                    if isinstance(remote_url, list):
                        self.model = [OpenAI(base_url=url) for url in remote_url]
                    else:
                        self.model = OpenAI(base_url=remote_url)

                except:
                    raise ValueError(f"Unsupported model name: {model_name}")

    # ---- Azure rotation setup (for GPT models) ----
    def _init_azure_rotation(self) -> bool:
        """Initialize Azure credential + resource rotation.

        Returns True if successful, False if fallback to standard OpenAI should occur.
        """
        try:
            resources_env = os.getenv("AZURE_OPENAI_RESOURCE_NAMES")
            resources = (
                _strip_split_csv(resources_env)
                if resources_env
                else _default_azure_resource_list()
            )
            if not resources:
                logging.warning(
                    "No Azure OpenAI resources configured; fallback to OpenAI()."
                )
                return False
            self._azure_resources = resources
            self._azure_api_version = _azure_api_version()
            self._azure_credential = _build_chained_credential()
            scope = _get_scope()
            self._azure_token_provider = get_bearer_token_provider(
                self._azure_credential, scope
            )
            self._azure_client_cache = {}
            self._azure_lock = threading.Lock()
            self._azure_rr_counter = 0
            # we purposefully do NOT set self.model here; _call_gpt will use rotated clients
            self.model = None  # type: ignore
            return True
        except Exception as e:
            logging.warning(f"Azure rotation init failed: {e}; fallback to OpenAI().")
            return False

    def _get_azure_client(self):
        """Return a rotated AzureOpenAI client (thread-safe)."""
        with self._azure_lock:
            resource = self._azure_resources[
                self._azure_rr_counter % len(self._azure_resources)
            ]
            self._azure_rr_counter += 1
            if resource not in self._azure_client_cache:
                self._azure_client_cache[resource] = AzureOpenAI(
                    api_version=self._azure_api_version,
                    azure_endpoint=_azure_endpoint(resource),
                    azure_ad_token_provider=self._azure_token_provider,
                )
            return self._azure_client_cache[resource]

    def respond(self, message_history: list):
        if self.model_type == "local":
            return self._call_local(message_history)
        else:
            if "claude" in self.model_name:
                return self._call_claude(message_history)
            elif "gemini" in self.model_name:
                return self._call_gemini(message_history)
            elif "gpt" in self.model_name:
                return self._call_gpt(message_history)
            elif "qwen" in self.model_name:
                return self._call_gpt(message_history)
            elif "Qwen2-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Qwen2.5-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                return self._call_llama11b(message_history)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "90b-vision-instruct" in self.model_name:
                return self._call_llama90(message_history)
            elif "OpenGVLab/InternVL" in self.model_name:
                return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-38B" in self.model_name:
            #     return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-78B" in self.model_name:
            #     return self._call_intern38b(message_history)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def _call_local(self, message_history: list):
        if self.task_type == "manip":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": (
                        llm_generation_guide_manip
                        if self.language_only
                        else vlm_generation_guide_manip
                    ),
                },
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": (
                        llm_generation_guide
                        if self.language_only
                        else vlm_generation_guide
                    ),
                },
            }
        response = self.model(
            message_history,
            gen_config=GenerationConfig(
                temperature=temperature,
                response_format=response_format,
                max_new_tokens=max_completion_tokens,
            ),
        )
        out = response.text
        out = fix_json(out)
        return out

    def _call_claude(self, message_history: list):
        if not self.language_only:
            message_history = convert_format_2claude(message_history)

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            messages=message_history,
        )

        return response.content[0].text

    def _call_gemini(self, message_history: list):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.task_type == "manip":
            response = self.model.beta.chat.completions.parse(
                model=self.model_name,
                messages=message_history,
                response_format=(
                    ActionPlan_lang_manip if self.language_only else ActionPlan_manip
                ),
                temperature=temperature,
                max_tokens=max_completion_tokens,
            )
        else:
            response = self.model.beta.chat.completions.parse(
                model=self.model_name,
                messages=message_history,
                response_format=ActionPlan_lang if self.language_only else ActionPlan,
                temperature=temperature,
                max_tokens=max_completion_tokens,
            )
        tokens = response.usage.prompt_tokens

        return str(response.choices[0].message.parsed.model_dump_json())

    def _call_gpt(self, message_history: list):
        if not self.language_only:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide
                    ),
                )
        else:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide
                    ),
                )

        # If Azure rotation is configured, obtain a rotated client; else use self.model.
        client = (
            self._get_azure_client()
            if hasattr(self, "_azure_resources")
            else self.model
        )
        response = self._chat_with_retry(
            client=client,
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )
        out = response.choices[0].message.content if response else ""

        return out

    # ---- Retry helper for GPT Azure calls ----
    def _chat_with_retry(
        self,
        client,
        model: str,
        messages: list,
        response_format: dict,
        temperature: float,
        max_tokens: int,
        stop: list[str] = None,
    ):
        """Indefinitely retry on 429 (rate limit) and selected transient errors.

        Backoff parameters via env vars:
          GPT_INITIAL_BACKOFF (seconds, default 2)
          GPT_MAX_BACKOFF (seconds, default 60)
          GPT_RETRY_JITTER (seconds, default 0.5)
          GPT_RETRY_LOG_EVERY (attempts, default 1)
        """
        import random
        import time

        from openai import APIError, RateLimitError

        initial = float(os.getenv("GPT_INITIAL_BACKOFF", 2))
        maximum = float(os.getenv("GPT_MAX_BACKOFF", 60))
        jitter = float(os.getenv("GPT_RETRY_JITTER", 0.5))
        log_every = int(os.getenv("GPT_RETRY_LOG_EVERY", 1))

        attempt = 0
        delay = initial
        while True:
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if stop is not None:
                    kwargs["stop"] = stop
                return client.chat.completions.create(**kwargs)
            except RateLimitError as e:  # 429
                attempt += 1
                if attempt % log_every == 0:
                    logging.warning(
                        f"Rate limited (429) on attempt {attempt}; retrying after {delay:.1f}s. Error: {e}"
                    )
                time.sleep(delay + random.uniform(0, jitter))
                delay = min(delay * 2, maximum)
                continue
            except APIError as e:
                # Retry on transient server errors (500/502/503/504) if status is available.
                status = getattr(e, "status_code", None)
                if status in {500, 502, 503, 504}:
                    attempt += 1
                    if attempt % log_every == 0:
                        logging.warning(
                            f"Transient API error {status} attempt {attempt}; retrying after {delay:.1f}s. Error: {e}"
                        )
                    time.sleep(delay + random.uniform(0, jitter))
                    delay = min(delay * 2, maximum)
                    continue
                logging.error(f"Non-retriable API error: {e}")
                return None
            except Exception as e:
                # Unexpected error: log and do single short retry; if persists, give up.
                attempt += 1
                if attempt > 3:
                    logging.error(f"Aborting after unexpected error attempts: {e}")
                    return None
                logging.warning(
                    f"Unexpected error '{e}' attempt {attempt}; retrying in {delay:.1f}s"
                )
                time.sleep(delay + random.uniform(0, jitter))
                delay = min(delay * 2, maximum)
                continue

    def _call_qwen7b(self, message_history: list):
        duration_logger = DurationLogger("RemoteModel#_call_qwen7b")

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide
                    ),
                )
        else:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide
                    ),
                )

        if reasoning_mode:
            response_format = None

        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # import debugpy  # todo(atupini): remove before commit
        # try:
        #     debugpy.listen(5678)
        # except RuntimeError:
        #     # Already listening
        #     pass
        # if not debugpy.is_client_connected():
        #     print('Waiting for debugger attach...', flush=True)
        #     try:
        #         debugpy.wait_for_client()
        #     except RuntimeError:
        #         pass  # Ignore if cannot wait
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------

        if os.environ.get("DEBUG_REMOTE_MODEL_INPUTS", "0") == "1":
            print(
                f"---------------------\nInput to model: \n\n"
            )
            messages_to_print = json.loads(json.dumps(message_history))
            try:
                for msgssc in messages_to_print[0]["content"]:
                    if "image_url" in msgssc:
                        print("\n[[...image url here...]]\n")
                    if "text" in msgssc:
                        print(f"\n{msgssc['text']}\n")
            except:
                print("[[...could not parse message content...]]")

            print(
                f"\n\n---------------------------------------------------"
            )

        model: Union[OpenAI, list[OpenAI]] = self.model
        if isinstance(model, list):
            model = model[self._model_rotation_index % len(model)]
            self._model_rotation_index += 1



        # todo(atupini) we should probably move all this to a new `call_vllm` method!
        with duration_logger.extend("model inference"):
            response = model.chat.completions.create(
                model="vllm-model",
                messages=message_history,
                **({"response_format": response_format} if response_format else {}),  # type: ignore
                temperature=temperature,
                max_tokens=max_completion_tokens,
                **({"stop": stop_seqs} if stop_seqs else {}),
                **({"frequency_penalty": float(vllm_frequency_penalty)} if vllm_frequency_penalty else {}),
                **({"top_p": float(vllm_top_p)} if vllm_top_p else {}),
            )  # type: ignore

        out = response.choices[0].message.content
        if os.environ.get("DEBUG_REMOTE_MODEL_OUTPUTS", "0") == "1":
            print(
                f"---------------------------------------------------\nRaw output from model: \n\n{out}\n\n---------------------------------------------------"
            )

        parsing_duration_logger = duration_logger.extend("parsing")
        parsing_duration_logger.start()

        if reasoning_mode:
            # Robust boxed JSON extraction strategy:
            # 1. Prefer the LAST boxed region (model may self-correct producing multiple boxes)
            # 2. Fallback to first boxed region (legacy behavior)
            # 3. Fallback to scanning entire output for first JSON object

            with parsing_duration_logger.extend(
                "extract_last_box_json(out) or extract_box_json(out)"
            ):
                box_json = extract_last_box_json(out) or extract_box_json(out)

            candidate = None

            if box_json:
                with parsing_duration_logger.extend("sanitize_box_payload(box_json)"):
                    candidate = sanitize_box_payload(box_json)

            if not candidate:
                # No boxed payload at all – attempt raw scan for a JSON object in full output
                with parsing_duration_logger.extend(
                    "fallback[extract_first_json_object(out)]"
                ):
                    fallback_obj = extract_first_json_object(out)
                    candidate = fallback_obj if fallback_obj else None

            if candidate:
                with parsing_duration_logger.extend("fix_json"):
                    out = fix_json(candidate)
            else:
                # Leave `out` unchanged; downstream maybe_repair will fail parse and log.
                pass

        # Always attempt plan validation / repair unless explicitly disabled.
        # Controlled by PLAN_REPAIR_MODE env (off|log|apply). Default 'apply'.
        try:
            with parsing_duration_logger.extend("maybe_repair"):
                out = maybe_repair(out)
        except Exception as e:
            # Fail-safe: never crash the call; log and return original.
            import logging as _logging

            _logging.warning(f"plan repair failed: {e}")

        parsing_duration_logger.stop()

        if os.environ.get("DEBUG_REMOTE_MODEL_OUTPUTS", "0") == "1":
            is_json_parseable = False
            try:
                _ = json.loads(out)
                is_json_parseable = True
            except:
                is_json_parseable = False

            print(
                f"---------------------------------------------------\nPost-parse candidate ({is_json_parseable=}): \n\n{out}\n\n---------------------------------------------------"
            )

        return out

    def _call_llama90(self, message_history: list):
        if self.task_type == "manip":
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={
                    "type": "json_object",
                    "schema": ActionPlan_1_manip.model_json_schema(),
                },
                temperature=temperature,
            )
            out = response.choices[0].message.content

        else:
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={
                    "type": "json_object",
                    "schema": ActionPlan_1.model_json_schema(),
                },
                temperature=temperature,
            )
            out = response.choices[0].message.content
        return out

    def _call_llama11b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide
                    ),
                )
        else:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide
                    ),
                )

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )
        out = response.choices[0].message.content
        return out

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide
                    ),
                )
        else:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide
                    ),
                )

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out

    def _call_intern38b(self, message_history):
        # if not self.language_only:
        #     message_history = convert_format_2gemini(message_history)

        # no use, lmdeploy use support json schema only if it is pytorch-backended
        if not self.language_only:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=vlm_generation_guide
                    ),
                )
        else:
            if self.task_type == "manip":
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide_manip
                    ),
                )
            else:
                response_format = dict(
                    type="json_schema",
                    json_schema=dict(
                        name="embodied_planning", schema=llm_generation_guide
                    ),
                )

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out


if __name__ == "__main__":
    model = RemoteModel(
        "Qwen/Qwen2-VL-72B-Instruct",  #'meta-llama/Llama-3.2-11B-Vision-Instruct',
        True,  # False
    )  #'claude-3-5-sonnet-20241022, Qwen/Qwen2-VL-72B-Instruct, meta-llama/Llama-3.2-11B-Vision-Instruct

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image("../../evaluator/midlevel/output.png")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    },
                },
                {
                    "type": "text",
                    "text": f"What do you think for this picture?? {template}?",
                },
            ],
        }
    ]

    response = model.respond(messages)
    print(response)
