import os
import re
import base64
import copy
import json  # Added for optional validation after fixes
from mimetypes import guess_type
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
import typing_extensions as typing
from pydantic import BaseModel, Field

template_lang = '''\
The output json format should be {'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \
2. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \
3. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!
'''

template = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_lang_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available. 
2. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name. 
3. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. visual_state_description: Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input. 
2. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available. 
3. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name. 
4. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

def fix_json(json_str):
    """Attempt to sanitize and minimally repair model-emitted JSON.

    Pipeline (ordered):
      1. Canonicalize quotes & remove markdown fences / contraction artifacts.
      2. Escape attribute quotes inside ANY opening XML/HTML-like tags (generic – covers <points ...> and others).
      3. Regex field-level inner quote escaping for long text fields.
      4. Balance delimiters if truncated (append missing '}' or ']').
      5. Parse validation attempt; on failure perform a salvage pass that heuristically escapes
         unescaped inner quotes inside string literals without over-escaping structural quotes.

    NOTE: The salvage pass is intentionally conservative – it only transforms quotes that would
    otherwise terminate a string but are not followed by a structural boundary (comma, brace, bracket).
    This favors recovering most malformed reasoning strings while limiting collateral damage.
    """
    # -----------------------------------------------------
    # 1. Canonicalization & basic cleanup
    # -----------------------------------------------------
    json_str = json_str.replace("'",'"')
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # -----------------------------------------------------
    # 2. Generic tag attribute quote escaping
    # -----------------------------------------------------
    # Previously we only escaped <points ...>. We now generically handle ANY opening tag with attributes
    # to avoid JSON termination via raw attribute quotes inside emitted markup.
    def escape_tag_attribute_quotes(s: str) -> str:
        tag_pattern = r'<(?!/)([a-zA-Z][\w\-]*)([^<>]*?)>'  # opening tag (no leading /), capture attr region
        def repl(m: re.Match) -> str:
            tag_name = m.group(1)
            attr_region = m.group(2)
            # Escape all non-escaped quotes in attribute region only
            escaped_attrs = re.sub(r'(?<!\\)"', r'\\"', attr_region)
            return f'<{tag_name}{escaped_attrs}>'
        return re.sub(tag_pattern, repl, s)

    json_str = escape_tag_attribute_quotes(json_str)

    # Field-level escaping: for specific textual keys whose values may still contain
    # remaining unescaped quotes (outside of <points> attributes) we apply a regex-based
    # capture and escape pass.
    # We look for any of the target keys followed by a quote, capture lazily until the next
    # key boundary (another known key or end of object / executable_plan key) and escape
    # any unescaped double quotes inside that segment.

    target_keys = [
        'visual_state_description',
        'reasoning_and_reflection',
        'language_plan'
    ]

    # Build a union of possible following keys to use in the lookahead so we stop at the right boundary.
    # executable_plan (array or string) marks the end of the textual section.
    following_union = '|'.join([re.escape(k) for k in (target_keys + ['executable_plan'])])

    # Regex explanation:
    #   ("(key1|key2|...)"\s*:\s*")  -> group 1: the full prefix including key & opening quote
    #   (?P<value>.*?)                 -> lazily capture the value contents
    #   (?=",\s*"(nextKey1|nextKey2|...)"|"?\s*}\s*$) -> lookahead for "," then another key OR end of object
    # Build inner alternation separately to avoid nested quote confusion in older Python versions
    keys_alt = '|'.join(target_keys)

    # Use double escaping for backslashes in raw pattern; avoid unsupported escape warnings.
    # Build regex with explicit escaping; using single backslashes in final compiled pattern
    # We'll assemble without raw string to avoid accidental escape swallowing.
    pattern = (
        "(\"(?:" + keys_alt + ")\"\\s*:\\s*\")"  # key and opening quote
        + "(?P<value>.*?)"  # captured value
        + "(?=\\\",\\s*\\\"(?:" + following_union + ")\\\"|\\\"?\\s*})"  # lookahead boundary
    )

    def escape_inner_quotes(segment: str) -> str:
        # Escape any quote not already escaped. We avoid double-escaping by negative lookbehind for backslash.
        return re.sub(r'(?<!\\)"', r'\\"', segment)

    def field_replacer(match: re.Match) -> str:
        prefix = match.group(1)
        value = match.group('value')
        # Run general escaping as a safety net (global points escaping done earlier).
        fixed_value = escape_inner_quotes(value)
        return prefix + fixed_value

    fixed_json = re.sub(pattern, field_replacer, json_str, flags=re.DOTALL)

    # -----------------------------------------------------
    # Optional newline escaping inside string literals.
    # Some models emit multi-line reasoning / plans with raw newline characters
    # that downstream consumers may wish to treat as explicit \n tokens (e.g., when
    # piping through one-line log parsers). We conservatively replace raw newlines
    # inside JSON string literal bodies with '\n'. Structural newlines (outside of
    # strings) remain untouched.
    def escape_newlines_in_strings(s: str) -> str:
        out = []
        in_string = False
        escape = False
        for ch in s:
            if in_string:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == '\\':
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                    out.append(ch)
                    continue
                if ch == '\n':
                    out.append('\\n')
                else:
                    out.append(ch)
            else:
                if ch == '"':
                    in_string = True
                out.append(ch)
        return ''.join(out)

    fixed_json = escape_newlines_in_strings(fixed_json)

    # -----------------------------------------------------
    # 2.5 Comment stripping (non-standard JSON comments from LLM)
    # -----------------------------------------------------
    # Remove line/block/HTML-style comments occurring OUTSIDE of string literals.
    # Supported patterns:
    #   // ... \n
    #   /* ... */
    #   # ... \n  (python style)
    #   <!-- ... -->  (HTML style)
    def strip_json_comments(s: str) -> str:
        out = []
        i = 0
        in_string = False
        escape = False
        while i < len(s):
            ch = s[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            # not in string
            if ch == '"':
                in_string = True
                out.append(ch)
                i += 1
                continue
            # line comment
            if ch == '/' and i + 1 < len(s) and s[i+1] == '/':
                # skip until newline or end
                i += 2
                while i < len(s) and s[i] not in '\n\r':
                    i += 1
                # preserve newline char to keep line count stable
                continue
            # block comment
            if ch == '/' and i + 1 < len(s) and s[i+1] == '*':
                i += 2
                while i + 1 < len(s) and not (s[i] == '*' and s[i+1] == '/'):
                    i += 1
                i += 2 if i + 1 < len(s) else 1
                continue
            # python style line comment
            if ch == '#':
                i += 1
                while i < len(s) and s[i] not in '\n\r':
                    i += 1
                continue
            # html style <!-- ... -->
            if ch == '<' and i + 3 < len(s) and s[i+1:i+4] == '!--':
                i += 4
                while i + 2 < len(s) and not (s[i] == '-' and s[i+1] == '-' and s[i+2] == '>'):
                    i += 1
                i += 3 if i + 2 < len(s) else (len(s) - i)
                continue
            out.append(ch)
            i += 1
        return ''.join(out)

    fixed_json = strip_json_comments(fixed_json)

    # ---------------------------------------------------------
    # Passive structural balancing for truncated JSON objects.
    # ---------------------------------------------------------
    # Some model outputs end prematurely (e.g., missing the final '}' after the
    # executable_plan array). We attempt a conservative balance of braces and
    # brackets OUTSIDE of string literals. We do NOT try to *repair* ordering
    # errors (extra closers, mismatched types); if those occur we return as-is.
    # Heuristic: scan once, track a stack of opening delimiters. At end, append
    # required closing delimiters in reverse order.
    def balance_delimiters(s: str) -> str:
        stack: list[str] = []
        in_string = False
        escape = False
        for ch in s:
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch in '{[':
                    stack.append(ch)
                elif ch in '}]':
                    if stack:
                        top = stack[-1]
                        if (top == '{' and ch == '}') or (top == '[' and ch == ']'):
                            stack.pop()
                        else:
                            # Mismatched closer; abort balancing to avoid corrupting.
                            return s
        if not stack:
            return s
        closing_map = {'{': '}', '[': ']'}
        # Append missing closers in reverse nesting order.
        return s + ''.join(closing_map[o] for o in reversed(stack))

    balanced_json = balance_delimiters(fixed_json)

    # -----------------------------------------------------
    # 5. Validation + salvage pass
    # -----------------------------------------------------
    try:
        json.loads(balanced_json)
        return balanced_json  # Already valid
    except json.JSONDecodeError:
        pass  # proceed to salvage

    def salvage_unescaped_inner_quotes(s: str) -> str:
        """Heuristically escape inner quotes in string literals.

        Logic:
          Iterate char-by-char; when inside a JSON string and encountering an unescaped '"',
          decide if it is a closing delimiter or content. If the subsequent non-whitespace
          character is NOT a structural boundary (comma, closing brace/bracket) we treat it
          as content and escape it. This recovers cases like: "We consider "object" hidden".

        Limitations:
          - May over-escape in rare edge cases with intentional early key/value termination.
          - Does not attempt to fix mismatched braces (handled earlier).
        """
        out = []
        in_string = False
        escape = False
        for i, ch in enumerate(s):
            if ch == '"' and not escape:
                if in_string:
                    # Look ahead skipping whitespace
                    j = i + 1
                    while j < len(s) and s[j].isspace():
                        j += 1
                    # Structural boundary tokens that legitimately close the string
                    if j < len(s) and s[j] in [',', '}', ']', ':']:
                        out.append(ch)  # closing quote
                        in_string = False
                    else:
                        # Inner content quote → escape
                        out.append('\\"')
                else:
                    in_string = True
                    out.append(ch)
            else:
                out.append(ch)
            # Track escape state
            if ch == '\\' and not escape:
                escape = True
            else:
                escape = False
        return ''.join(out)

    salvaged = salvage_unescaped_inner_quotes(balanced_json)
    try:
        json.loads(salvaged)
        return salvaged
    except json.JSONDecodeError:
        # If still invalid, return the original balanced version for upstream logging/diagnostics.
        return balanced_json

# -------------------------------------------------------------
# Reasoning / boxed JSON augmentation helpers
# -------------------------------------------------------------
# When EMB_REASONING_MODE=1 is set in the environment, planners will append a
# reasoning suffix instructing the model to emit:
#   <think>...</think><answer>...<|begin_of_box|>{JSON}<|end_of_box|></answer>
# We then extract the JSON inside the box tags and feed it to existing
# json_to_action logic unchanged.

reasoning_suffix = (
    "\n\nPlease think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using natural language thought expressions. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think></think> tags, and then give your final JSON answer between the <|begin_of_box|><|end_of_box|> tags (do not include anything besides the requested JSON)."
)

_BOX_JSON_PATTERN = re.compile(r'<\|begin_of_box\|>(.*?)</?\|end_of_box\|>', re.DOTALL)

def extract_box_json(output_text: str):
    """Return only the FIRST boxed JSON segment (backwards compatible).

    Retained for compatibility with earlier imports. Prefer using
    ``extract_last_box_json`` which is more robust when the model emits
    multiple boxed regions (e.g. streaming self-corrections)."""
    box_match = _BOX_JSON_PATTERN.search(output_text)
    return box_match.group(1).strip() if box_match else None

def extract_last_box_json(output_text: str):
    """Return the LAST boxed JSON payload if multiple are present.

    Empirical failure mode observed: the model occasionally emits an
    early partial JSON inside <|begin_of_box|> ... <|end_of_box|> followed by
    a refined complete JSON later. Selecting the final occurrence reduces
    parse errors from truncated first drafts.
    """
    matches = list(_BOX_JSON_PATTERN.finditer(output_text))
    if not matches:
        return None
    return matches[-1].group(1).strip()

def sanitize_box_payload(raw: str):
    """Lightly sanitize the raw boxed region before JSON repair.

    Steps:
      1. Strip leading/trailing whitespace.
      2. If the region itself contains surrounding explanatory text plus an
         embedded JSON object, extract the FIRST complete top-level object.
      3. Return the candidate string (even if still malformed) for downstream
         ``fix_json`` which performs structural salvage.
    """
    if raw is None:
        return None
    candidate = raw.strip()
    if '{' in candidate and '}' in candidate:
        inner = extract_first_json_object(candidate)
        if inner:
            return inner.strip()
    return candidate

def extract_first_json_object(text: str):
    """Fallback: attempt to find first top-level JSON object if boxed JSON absent.
    Useful when the model did not follow boxing instructions but produced JSON.
    """
    # Simple bracket depth scan.
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


class ExecutableAction_1(typing.TypedDict):
    action_id: int
    action_name: str
class ActionPlan_1(BaseModel):
    visual_state_description: str = Field(
        description="Description of current state from the visual image"
    )
    reasoning_and_reflection: str = Field(
        description="summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task"
    )
    language_plan: str = Field(
        description="The list of actions to achieve the user instruction. Each action is started by the step number and the action name"
    )
    executable_plan: list[ExecutableAction_1] = Field(
        description="A list of actions needed to achieve the user instruction, with each action having an action ID and a name."
    )

class ActionPlan_1_manip(BaseModel):
    visual_state_description: str = Field(
        description="Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input."
    )
    reasoning_and_reflection: str = Field(
        description="Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available."
    )
    language_plan: str = Field(
        description="A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name."
    )
    executable_plan: str = Field(
        description="A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action."
    )

def convert_format_2claude(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":
            new_content = []
    
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages

def convert_format_2gemini(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":

            new_content = []
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}"
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)
        
    return new_messages



class ExecutableAction(typing.TypedDict):
    action_id: int
    action_name: str
class ActionPlan(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction]

class ActionPlan_manip(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

class ExecutableAction_lang(typing.TypedDict):
    action_id: int
    action_name: str
class ActionPlan_lang(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction_lang]

class ActionPlan_lang_manip(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def truncate_message_prompts(message_history: list):
    """
    Traverse the message list and truncate the part before "------------" in the text content of all messages except the last one
    
    Args:
        message_history: Message list, each message contains role and content
        
    Returns:
        list: Processed message list
    """
    if not message_history:
        return message_history
        
    # Create a deep copy to avoid modifying the original data
    processed_messages = []
    
    # Process all messages except the last one
    for i, message in enumerate(message_history):
        if i == len(message_history) - 1:
            # Keep the last message unchanged
            processed_messages.append(message)
        else:
            # Process current message
            processed_message = {
                "role": message.get("role", ""),
                "content": []
            }
            
            # Traverse content list
            for content_item in message.get("content", []):
                if content_item.get("type") == "text":
                    # Process text type content
                    text_content = content_item.get("text", "")
                    
                    # Look for "----------" separator
                    if "----------" in text_content:
                        # Truncate content before separator, keep content after separator
                        truncated_text = text_content.split("----------")[1]
                    else:
                        # If no separator found, keep original text
                        truncated_text = text_content
                        
                    processed_content_item = content_item.copy()
                    processed_content_item["text"] = truncated_text
                    processed_message["content"].append(processed_content_item)
                else:
                    # Directly copy non-text type content
                    processed_message["content"].append(content_item.copy())
            
            processed_messages.append(processed_message)
    
    return processed_messages