import base64
import json
import os
from pathlib import Path
import sys

import anthropic
import google.generativeai as genai
import lmdeploy
import typing_extensions as typing
from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from openai import OpenAI

from embodiedbench.planner.planner_config.generation_guide import (
    llm_generation_guide,
    vlm_generation_guide,
)
from embodiedbench.planner.planner_config.generation_guide_manip import (
    llm_generation_guide_manip,
    vlm_generation_guide_manip,
)
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

temperature = 0
max_completion_tokens = 2048
remote_url = os.environ.get("remote_url")


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
                self.model = OpenAI()
            elif "qwen" in self.model_name:
                self.model = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif "Qwen2-VL" in self.model_name:
                self.model = OpenAI(base_url=remote_url)
            elif "Qwen2.5-VL" in self.model_name:
                self.model = OpenAI(base_url=remote_url)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url=remote_url)
            elif "OpenGVLab/InternVL" in self.model_name:
                self.model = OpenAI(base_url=remote_url)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url=remote_url)
            elif (
                "90b-vision-instruct" in self.model_name
            ):  # you can use fireworks to inference
                self.model = OpenAI(
                    base_url="https://api.fireworks.ai/inference/v1",
                    api_key=os.environ.get("firework_API_KEY"),
                )
            elif "vllm" in self.model_name:  # vLLM server with OpenAI-compatible API
                self.model = OpenAI(
                    base_url=remote_url
                    or os.getenv("REMOTE_URL", "http://localhost:8000/v1"),
                    api_key="EMPTY",  # vLLM doesn't require an API key
                )
            else:
                try:
                    self.model = OpenAI(base_url=remote_url)
                except:
                    raise ValueError(f"Unsupported model name: {model_name}")

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
            elif "vllm" in self.model_name:
                return self._call_vllm(message_history)
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

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )
        out = response.choices[0].message.content

        return out

    def _call_qwen7b(self, message_history: list):

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

    def _call_vllm(self, message_history: list):
        """Call vLLM server with OpenAI-compatible API"""

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

        out = fix_json(out)
        
        enable_debug_logs = os.environ.get("ENABLE_DEBUG_LOGS", "0") == "1"
        if enable_debug_logs:
            # Debug: Save images and message history
            debug_dir = Path.cwd() / "debug_messages___dev"
            os.makedirs(debug_dir, exist_ok=True)

            # Count how many images we're sending and save them
            image_count = 0
            for msg in message_history:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            # Save the base64 image to disk
                            if image_url.startswith("data:image"):
                                # Extract base64 data
                                base64_data = image_url.split(",", 1)[1]
                                image_bytes = base64.b64decode(base64_data)
                                image_path = os.path.join(
                                    debug_dir, f"image_{image_count}.png"
                                )
                                with open(image_path, "wb") as f:
                                    f.write(image_bytes)
                                print(f"DEBUG: Saved image to {image_path}")
                                image_count += 1

            # Build a readable text representation of the conversation
            conversation_text = "=" * 80 + "\n"
            conversation_text += "CONVERSATION HISTORY\n"
            conversation_text += "=" * 80 + "\n\n"

            for i, msg in enumerate(message_history, 1):
                role = msg.get("role", "unknown").upper()
                conversation_text += f"{'=' * 80}\n"
                conversation_text += f"TURN {i}: {role}\n"
                conversation_text += f"{'=' * 80}\n"

                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text parts and note images
                    for item in content:
                        if item.get("type") == "text":
                            conversation_text += item["text"] + "\n"
                        elif item.get("type") == "image_url":
                            conversation_text += "[IMAGE ATTACHED]\n"
                else:
                    conversation_text += str(content) + "\n"

                conversation_text += "\n"

            print(f"DEBUG: Total images sent: {image_count}")

            # Add the model's response to the conversation text
            conversation_text += f"{'=' * 80}\n"
            conversation_text += f"MODEL RESPONSE\n"
            conversation_text += f"{'=' * 80}\n"
            conversation_text += out + "\n"
            conversation_text += "\n" + "=" * 80 + "\n"

            # Save the full conversation to a text file
            conversation_file = os.path.join(debug_dir, "conversation.txt")
            with open(conversation_file, "w", encoding="utf-8") as f:
                f.write(conversation_text)
            print(f"DEBUG: Saved conversation to {conversation_file}")

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
