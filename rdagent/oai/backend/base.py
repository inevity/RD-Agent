from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import pytz
from pydantic import TypeAdapter

from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils import md5_hash
from pprint import pformat
import inspect, traceback

try:
    import litellm
    import openai

    openai_imported = True
except ImportError:
    openai_imported = False


def truncate(data: Any, max_length: int = 500) -> Any:
    """Helper to truncate long messages for logging"""
    if isinstance(data, (list, dict, str)):
        str_repr = str(data)
        return f"{str_repr[:max_length]}...[truncated]" if len(str_repr) > max_length else str_repr
    return data

# def fix_code_json(raw_text: str) -> str:
#     """
#     Fix a JSON string that contains only a single 'code' field
#     with possibly unescaped multiline content.
#     If the JSON doesn't match exactly this structure, return it unchanged.
#     Args:
#         raw_text (str): The raw JSON string to fix.
#     Returns:
#         str: Fixed JSON string, or original string if not match.
#     """
#     match = re.fullmatch(r'\s*\{\s*"code"\s*:\s*"((?:[^"\\]|\\.|\\n|\\r|\\t|[\n\r])*)"\s*\}\s*', raw_text, re.DOTALL)
#     if match:
#         raw_code = match.group(1)
#         try:
#             # Decode any escaped characters like \\n -> \n
#             normalized_code = bytes(raw_code, "utf-8").decode("unicode_escape")
#         except Exception:
#             normalized_code = raw_code
#         return json.dumps({"code": normalized_code}, indent=4)
#
#     # For all other cases, return original
#     return raw_text
def fix_code_json(raw_text: str) -> str:
    """
    Fix a broken JSON string with only a 'code' key and unescaped multiline value.
    Return the fixed JSON if it matches, else return original string.

    Args:
        raw_text (str): A potentially broken JSON string.

    Returns:
        str: Properly escaped JSON string if it's only {"code": ...}, else original.
    """
    # Match only JSON with exactly one "code" key and raw string value (may have real newlines)
    match = re.fullmatch(
        r'\s*\{\s*"code"\s*:\s*"((?:[^"\\]|\\.|[\n\r])*)"\s*\}\s*',
        raw_text,
        re.DOTALL
    )

    if match:
        raw_code = match.group(1)
        try:
            # Decode backslash-escaped characters like \\n into \n
            decoded_code = bytes(raw_code, "utf-8").decode("unicode_escape")
        except Exception:
            decoded_code = raw_code
        return json.dumps({"code": decoded_code}, indent=4)
    
    # Return original if doesn't match exactly {"code": ...}
    return raw_text

def sanitize_and_parse_json(json_str: str) -> Dict[str, Any]:
    """
    Fix common JSON issues (invalid control chars, trailing commas, etc.)
    and parse the string into a Python dict.
    
    Args:
        json_str: Potentially malformed JSON string
        
    Returns:
        Parsed JSON as a Python dictionary
        
    Raises:
        ValueError: If JSON cannot be repaired or parsed
    """
    # Fix 1: Remove invalid control characters (like standalone backslashes)
    sanitized = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
    
    # Fix 2: Escape newlines properly
    sanitized = sanitized.replace('\n', '\\n').replace('\r', '\\r')
    
    # Fix 3: Remove trailing commas
    sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
    
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after sanitization: {e}")


def fix_code_field_backslash(json_txt: str) -> str:
    try:
        # Match the content of the "code" field using a regex group
        def replace_in_code(match):
            code_content = match.group(1)
            # Replace comma + space + backslash with comma + \n
            fixed_code = re.sub(r",\s*\\", ", \\n", code_content)
            return f'"code": "{fixed_code}"'

        # This regex safely matches the content inside the "code" field (non-greedy)
        fixed_txt = re.sub(
            r'"code":\s*"((?:[^"\\]|\\.)*?)"',
            replace_in_code,
            json_txt,
            flags=re.DOTALL
        )

        # Try parsing the result to ensure it's valid JSON
        json.loads(fixed_txt)
        return fixed_txt
    except Exception:
        return json_txt  # fallback if any issue


def fix_formulation_slash(text: str) -> str:
    """
    Replaces escaped backslashes (\\) in the value of the 'formulation' key
    with real newline characters \n. If 'formulation' not found, return original.

    Args:
        text (str): Original JSON text.

    Returns:
        str: Modified JSON string if applicable, otherwise original text.
    """
    pattern = r'"formulation"\s*:\s*"((?:[^"\\]|\\.)*?)"'

    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return text  # 🔁 No formulation found — return original

    original_value = match.group(1)
    if "\\\\" not in original_value:
        return text  # 🔁 No double-backslash to replace — return original

    # 🔧 Replace \\ with real newline
    fixed_value = original_value.replace('\\\\', '\n')
    fixed_text = re.sub(pattern, f'"formulation": "{fixed_value}"', text, flags=re.DOTALL)
    return fixed_text

def fix_json_escaping(json_txt: str) -> str:
    try:
        # Try parsing the original JSON first
        json.loads(json_txt)
        return json_txt
    except json.JSONDecodeError:
        # Match and fix the "formulation" field
        def escape_backslashes_except_newlines(match):
            content = match.group(1)
            # Replace all \ that are not followed by 'n' (i.e., not \n)
            fixed = re.sub(r'\\(?!n)', r'\\\\', content)
            return f'"formulation": "{fixed}"'

        fixed_txt = re.sub(
            r'"formulation":\s*"((?:[^"\\]|\\.)*?)"',
            escape_backslashes_except_newlines,
            json_txt,
            flags=re.DOTALL
        )

        try:
            json.loads(fixed_txt)
            return fixed_txt
        except json.JSONDecodeError:
            return json_txt




class SQliteLazyCache(SingletonBaseClass):
    def __init__(self, cache_location: str) -> None:
        super().__init__()
        self.cache_location = cache_location
        db_file_exist = Path(cache_location).exists()
        # TODO: sqlite3 does not support multiprocessing.
        self.conn = sqlite3.connect(cache_location, timeout=20)
        self.c = self.conn.cursor()
        if not db_file_exist:
            self.c.execute(
                """
                CREATE TABLE chat_cache (
                    md5_key TEXT PRIMARY KEY,
                    chat TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE embedding_cache (
                    md5_key TEXT PRIMARY KEY,
                    embedding TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE message_cache (
                    conversation_id TEXT PRIMARY KEY,
                    message TEXT
                )
                """,
            )
            self.conn.commit()

    def chat_get(self, key: str) -> str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT chat FROM chat_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else result[0]

    def embedding_get(self, key: str) -> list | dict | str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT embedding FROM embedding_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else json.loads(result[0])

    def chat_set(self, key: str, value: str) -> None:
        md5_key = md5_hash(key)
        self.c.execute(
            "INSERT OR REPLACE INTO chat_cache (md5_key, chat) VALUES (?, ?)",
            (md5_key, value),
        )
        self.conn.commit()
        return None

    def embedding_set(self, content_to_embedding_dict: dict) -> None:
        for key, value in content_to_embedding_dict.items():
            md5_key = md5_hash(key)
            self.c.execute(
                "INSERT OR REPLACE INTO embedding_cache (md5_key, embedding) VALUES (?, ?)",
                (md5_key, json.dumps(value)),
            )
        self.conn.commit()

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        self.c.execute("SELECT message FROM message_cache WHERE conversation_id=?", (conversation_id,))
        result = self.c.fetchone()
        return [] if result is None else cast(list[dict[str, Any]], json.loads(result[0]))

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        self.c.execute(
            "INSERT OR REPLACE INTO message_cache (conversation_id, message) VALUES (?, ?)",
            (conversation_id, json.dumps(message_value)),
        )
        self.conn.commit()
        return None


class SessionChatHistoryCache(SingletonBaseClass):
    def __init__(self) -> None:
        """load all history conversation json file from self.session_cache_location"""
        self.cache = SQliteLazyCache(cache_location=LLM_SETTINGS.prompt_cache_path)

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.cache.message_get(conversation_id)

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        self.cache.message_set(conversation_id, message_value)


class ChatSession:
    def __init__(self, api_backend: Any, conversation_id: str | None = None, system_prompt: str | None = None) -> None:
        self.conversation_id = str(uuid.uuid4()) if conversation_id is None else conversation_id
        self.system_prompt = system_prompt if system_prompt is not None else LLM_SETTINGS.default_system_prompt
        self.api_backend = api_backend

    def build_chat_completion_message(self, user_prompt: str) -> list[dict[str, Any]]:
        history_message = SessionChatHistoryCache().message_get(self.conversation_id)
        messages = history_message
        if not messages:
            messages.append({"role": LLM_SETTINGS.system_prompt_role, "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_chat_completion_message_and_calculate_token(self, user_prompt: str) -> Any:
        messages = self.build_chat_completion_message(user_prompt)
        return self.api_backend._calculate_token_from_messages(messages)

    def build_chat_completion(self, user_prompt: str, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """
        this function is to build the session messages
        user prompt should always be provided
        """
        messages = self.build_chat_completion_message(user_prompt)

        with logger.tag(f"session_{self.conversation_id}"):
            response: str = self.api_backend._try_create_chat_completion_or_embedding(  # noqa: SLF001
                *args,
                messages=messages,
                chat_completion=True,
                **kwargs,
            )
            logger.log_object({"user": user_prompt, "resp": response}, tag="debug_llm")

        messages.append(
            {
                "role": "assistant",
                "content": response,
            },
        )
        SessionChatHistoryCache().message_set(self.conversation_id, messages)
        return response

    def get_conversation_id(self) -> str:
        return self.conversation_id

    def display_history(self) -> None:
        # TODO: Realize a beautiful presentation format for history messages
        pass


class APIBackend(ABC):
    """
    Abstract base class for LLM API backends
    supporting auto retry, cache and auto continue
    Inner api call should be implemented in the subclass
    """

    def __init__(
        self,
        use_chat_cache: bool | None = None,
        dump_chat_cache: bool | None = None,
        use_embedding_cache: bool | None = None,
        dump_embedding_cache: bool | None = None,
    ):
        self.dump_chat_cache = LLM_SETTINGS.dump_chat_cache if dump_chat_cache is None else dump_chat_cache
        self.use_chat_cache = LLM_SETTINGS.use_chat_cache if use_chat_cache is None else use_chat_cache
        self.dump_embedding_cache = (
            LLM_SETTINGS.dump_embedding_cache if dump_embedding_cache is None else dump_embedding_cache
        )
        self.use_embedding_cache = (
            LLM_SETTINGS.use_embedding_cache if use_embedding_cache is None else use_embedding_cache
        )
        if self.dump_chat_cache or self.use_chat_cache or self.dump_embedding_cache or self.use_embedding_cache:
            self.cache_file_location = LLM_SETTINGS.prompt_cache_path
            self.cache = SQliteLazyCache(cache_location=self.cache_file_location)

        self.retry_wait_seconds = LLM_SETTINGS.retry_wait_seconds

    def build_chat_session(
        self,
        conversation_id: str | None = None,
        session_system_prompt: str | None = None,
    ) -> ChatSession:
        """
        conversation_id is a 256-bit string created by uuid.uuid4() and is also
        the file name under session_cache_folder/ for each conversation
        """
        return ChatSession(self, conversation_id, session_system_prompt)

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> list[dict[str, Any]]:
        """
        build the messages to avoid implementing several redundant lines of code

        """
        if former_messages is None:
            former_messages = []
        # shrink multiple break will recursively remove multiple breaks(more than 2)
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            if system_prompt is not None:
                while "\n\n\n" in system_prompt:
                    system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = LLM_SETTINGS.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": LLM_SETTINGS.system_prompt_role,
                "content": system_prompt,
            },
        ]
        messages.extend(former_messages[-1 * LLM_SETTINGS.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def _build_log_messages(self, messages: list[dict[str, Any]]) -> str:
        log_messages = ""
        for m in messages:
            log_messages += (
                f"\n{LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END}"
                f"{LogColors.CYAN}{m['role']}{LogColors.END}\n"
                f"{LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END} "
                f"{LogColors.CYAN}{m['content']}{LogColors.END}\n"
            )
        return log_messages


# main portal
    def build_messages_and_create_chat_completion(  # type: ignore[no-untyped-def]
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list | None = None,
        chat_cache_prefix: str = "",
        shrink_multiple_break: bool = False,
        *args,
        **kwargs,
    ) -> str:
        # Get the frame one level up in the call stack
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame.function
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno

        logger.info(f"Called by function '{caller_name}' in {caller_file}, line {caller_line}")
        # print("Call stack:")
        # traceback.print_stack()


        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt,
            system_prompt,
            former_messages,
            shrink_multiple_break=shrink_multiple_break,
        )

        logger.info("create completion")
        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            *args,
            messages=messages,
            chat_completion=True,
            chat_cache_prefix=chat_cache_prefix,
            **kwargs,
        )
        if isinstance(resp, list):
            raise ValueError("The response of _try_create_chat_completion_or_embedding should be a string.")
        logger.log_object({"system": system_prompt, "user": user_prompt, "resp": resp}, tag="debug_llm")
        return resp

    def create_embedding(self, input_content: str | list[str], *args, **kwargs) -> list[float] | list[list[float]]:  # type: ignore[no-untyped-def]
        logger.info("create embedding")
        input_content_list = [input_content] if isinstance(input_content, str) else input_content
        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            input_content_list=input_content_list,
            embedding=True,
            *args,
            **kwargs,
        )
        if isinstance(input_content, str):
            return resp[0]  # type: ignore[return-value]
        return resp  # type: ignore[return-value]

    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: str | None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> int:
        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt, system_prompt, former_messages, shrink_multiple_break=shrink_multiple_break
        )
        return self._calculate_token_from_messages(messages)

    def _try_create_chat_completion_or_embedding(  # type: ignore[no-untyped-def]
        self,
        max_retry: int = 10,
        chat_completion: bool = False,
        embedding: bool = False,
        *args,
        **kwargs,
    ) -> str | list[list[float]]:
        assert not (chat_completion and embedding), "chat_completion and embedding cannot be True at the same time"
        max_retry = LLM_SETTINGS.max_retry if LLM_SETTINGS.max_retry is not None else max_retry
        timeout_count = 0
        violation_count = 0
        for i in range(max_retry):
            API_start_time = datetime.now()
            try:
                if embedding:
                    return self._create_embedding_with_cache(*args, **kwargs)
                if chat_completion:
                    # logger.info(f"kwargs received:\n{pformat(kwargs, indent=2)}")
                    return self._create_chat_completion_auto_continue(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if hasattr(e, "message") and (
                    "'messages' must contain the word 'json' in some form" in e.message
                    or "\\'messages\\' must contain the word \\'json\\' in some form" in e.message
                ):
                    kwargs["add_json_in_prompt"] = True
                elif hasattr(e, "message") and embedding and "maximum context length" in e.message:
                    kwargs["input_content_list"] = [
                        content[: len(content) // 2] for content in kwargs.get("input_content_list", [])
                    ]
                else:
                    RD_Agent_TIMER_wrapper.api_fail_count += 1
                    RD_Agent_TIMER_wrapper.latest_api_fail_time = datetime.now(pytz.timezone("Asia/Shanghai"))

                    if (
                        openai_imported
                        and isinstance(e, litellm.BadRequestError)
                        and (
                            isinstance(e.__cause__, litellm.ContentPolicyViolationError)
                            or "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
                            in str(e)
                        )
                    ):
                        violation_count += 1
                        if violation_count >= LLM_SETTINGS.violation_fail_limit:
                            logger.warning("Content policy violation detected.")
                            raise e

                    if (
                        openai_imported
                        and isinstance(e, openai.APITimeoutError)
                        or (
                            isinstance(e, openai.APIError)
                            and hasattr(e, "message")
                            and "Your resource has been temporarily blocked because we detected behavior that may violate our content policy."
                            in e.message
                        )
                    ):
                        timeout_count += 1
                        if timeout_count >= LLM_SETTINGS.timeout_fail_limit:
                            logger.warning("Timeout error, please check your network connection.")
                            raise e

                    recommended_wait_seconds = self.retry_wait_seconds
                    if openai_imported and isinstance(e, openai.RateLimitError) and hasattr(e, "message"):
                        match = re.search(r"Please retry after (\d+) seconds\.", e.message)
                        if match:
                            recommended_wait_seconds = int(match.group(1))
                    time.sleep(recommended_wait_seconds)
                    if RD_Agent_TIMER_wrapper.timer.started and not isinstance(e, json.decoder.JSONDecodeError):
                        RD_Agent_TIMER_wrapper.timer.add_duration(datetime.now() - API_start_time)
                logger.warning(str(e))
                logger.warning(f"Retrying {i+1}th time...")
        error_message = f"Failed to create chat completion after {max_retry} retries."
        raise RuntimeError(error_message)

    def _create_chat_completion_add_json_in_prompt(
        self,
        messages: list[dict[str, Any]],
        add_json_in_prompt: bool = False,
        json_mode: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        """
        add json related content in the prompt if add_json_in_prompt is True
        """
        logger.info(
            f"Chat completion add json in prompt called with:\n"
            f"messages: {truncate(messages)}\n"
            f"add_json_in_prompt: {add_json_in_prompt}\n"
            f"json_mode: {json_mode}\n"
        )
        
        # Log positional args if present
        if args:
            logger.info(f"Additional positional args (*args): {args}")
        
        # Log additional kwargs if present
        if kwargs:
            logger.info(f"Additional kwargs (**kwargs): {kwargs}")
        
        logger.info("Additional kwargs kwargs done")

        if json_mode and add_json_in_prompt:
            for message in messages[::-1]:
                message["content"] = message["content"] + "\nPlease respond in json format."
                if message["role"] == LLM_SETTINGS.system_prompt_role:
                    # NOTE: assumption: systemprompt is always the first message
                    break
        return self._create_chat_completion_inner_function(messages=messages, json_mode=json_mode, *args, **kwargs)  # type: ignore[misc]

    def _create_chat_completion_auto_continue(
        self,
        messages: list[dict[str, Any]], #messages is positional but can optionally be named
        *args: Any,
        json_mode: bool = False,
        chat_cache_prefix: str = "",
        seed: Optional[int] = None,
        json_target_type: Optional[str] = None,
        **kwargs: Any,
    # positional args and keyword argus 
    ) -> str:
        """
        Call the chat completion function and automatically continue the conversation if the finish_reason is length.
        """

        logger.info(
            f"Chat completion called with:\n"
            f"messages: {truncate(messages)}\n"
            f"json_mode: {json_mode}\n"
            f"chat_cache_prefix: {chat_cache_prefix}\n"
            f"seed: {seed}\n"
            f"json_target_type: {json_target_type}"
        )
        
        # Log positional args if present
        if args:
            logger.info(f"Additional positional args (*args): {args}")
        
        # Log additional kwargs if present
        if kwargs:
            logger.info(f"Additional kwargs (**kwargs): {kwargs}")
        
        logger.info("Additional kwargs kwargs done")


        if seed is None and LLM_SETTINGS.use_auto_chat_cache_seed_gen:
            seed = LLM_CACHE_SEED_GEN.get_next_seed()
        input_content_json = json.dumps(messages)
        input_content_json = (
            chat_cache_prefix + input_content_json + f"<seed={seed}/>"
        )  # FIXME this is a hack to make sure the cache represents the round index

        if self.use_chat_cache:
            cache_result = self.cache.chat_get(input_content_json)
            if cache_result is not None:
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(self._build_log_messages(messages), tag="llm_messages")
                    logger.info(f"{LogColors.CYAN}Response:{cache_result}{LogColors.END}", tag="llm_messages")
                return cache_result

        all_response = ""
        new_messages = deepcopy(messages)
        # fuck !!!!! 
        try_n = 6
        # logger.info(f"autocontine: kwargs received:\n{pformat(kwargs, indent=2)}")
        # logger.info(f"autocontine: json mode:\n{pformat(json, indent=2)}")
        for _ in range(try_n):  # for some long code, 3 times may not enough for reasoning models
            if "json_mode" in kwargs:
                del kwargs["json_mode"]
            response, finish_reason = self._create_chat_completion_add_json_in_prompt(
                new_messages, json_mode=json_mode, *args, **kwargs
            )  # type: ignore[misc]
            all_response += response
            logger.info(f"response:{all_response}")


            if finish_reason is None or finish_reason != "length":
                if json_mode:
                    #all_response = json.dumps(all_response)
                    #def normalize_python_to_json(python_str):
                    #    # Replace Python-style booleans with JSON-style
                    #    python_str = re.sub(r'\bTrue\b', 'true', python_str)
                    #    python_str = re.sub(r'\bFalse\b', 'false', python_str)
                    #    python_str = re.sub(r'\bNone\b', 'null', python_str)
                    #    
                    #    try:
                    #        # Validate that it's proper JSON now
                    #        json.loads(python_str)
                    #        return python_str
                    #    except json.JSONDecodeError as e:
                    #        return f"Error: Could not normalize to valid JSON: {e}"



                    #def parse_markdown_json(input_str):
                    #    # Remove code fence markers if present
                    #    if input_str.startswith("```"):
                    #        # Find first and last occurrence of code fences
                    #        start_idx = input_str.find("\n") + 1
                    #        end_idx = input_str.rfind("```")
                    #        if end_idx > start_idx:
                    #            input_str = input_str[start_idx:end_idx].strip()
                    #    
                    #    try:
                    #        return json.loads(input_str)
                    #    except json.JSONDecodeError as e:
                    #        print(f"JSON parse error: {e}")
                    #        return None

                    #all_response = normalize_python_to_json(all_response)

                    import re

                    def remove_thinking(text):
                        # Remove everything between <think> and </think> tags (including the tags)
                        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                        
                        # Alternative pattern if the closing tag is <think/>
                        cleaned_text = re.sub(r'<think>.*?<think/>', '', cleaned_text, flags=re.DOTALL)
                        
                        # Clean up any extra whitespace that might remain
                        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
                        
                        return cleaned_text.strip()
                    
                    all_response = all_response.replace(': False', ': false')
                    all_response = all_response.replace(': True', ': true')
                    all_response = remove_thinking(all_response)
                    #code
                    all_response = fix_code_json(all_response)

                    #formulation
                    # all_response = fix_formulation_slash(all_response)
                    # logger.info(f"fixed response formulation slash:{all_response}")
                    all_response = fix_json_escaping(all_response)
                    logger.info(f"fixed response formulation escape:{all_response}")

                    all_response = fix_code_field_backslash(all_response)
                    logger.info(f"fixed response:{all_response}")
                    try:
                        
                        logger.info("load all_response in try")
                        json.loads(all_response)
                        #parse_markdonw_json(all_response)
                    except:
                        match = re.search(r"```json(.*)```", all_response, re.DOTALL)
                        all_response = match.groups()[0] if match else all_response
                        # this line finds and captures everything between ```json and ```
                        logger.info(f"fixed response in except:{all_response}")
                        logger.info("load all_response in except")
                        json.loads(all_response)


                # logger.info("load fixed all_response")
                # json.loads(all_response)
                # try:
                #     fixed = sanitize_and_parse_json(all_response)
                #     # print("Successfully parsed:")
                #     # print(json.dumps(fixed, indent=2))
                # except ValueError as e:
                #     print(f"Error: {e}")
                # #escaped_json = json.dumps({"code": code}, indent=4)
                

                if json_target_type is not None:
                    logger.info("to valating the json using the json_target_type")
                    TypeAdapter(json_target_type).validate_json(all_response)
                    logger.info("done valating the json")

                if self.dump_chat_cache:
                    self.cache.chat_set(input_content_json, all_response)
                return all_response
            new_messages.append({"role": "assistant", "content": response})
        raise RuntimeError(f"Failed to continue the conversation after {try_n} retries.")

    def _create_embedding_with_cache(
        self, input_content_list: list[str], *args: Any, **kwargs: Any
    ) -> list[list[float]]:
        content_to_embedding_dict = {}
        filtered_input_content_list = []
        if self.use_embedding_cache:
            for content in input_content_list:
                cache_result = self.cache.embedding_get(content)
                if cache_result is not None:
                    content_to_embedding_dict[content] = cache_result
                else:
                    filtered_input_content_list.append(content)
        else:
            filtered_input_content_list = input_content_list

        if len(filtered_input_content_list) > 0:
            resp = self._create_embedding_inner_function(input_content_list=filtered_input_content_list)
            for index, data in enumerate(resp):
                content_to_embedding_dict[filtered_input_content_list[index]] = data
            if self.dump_embedding_cache:
                self.cache.embedding_set(content_to_embedding_dict)
        return [content_to_embedding_dict[content] for content in input_content_list]  # type: ignore[misc]

    @abstractmethod
    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Calculate the token count from messages
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_embedding_inner_function(  # type: ignore[no-untyped-def]
        self, input_content_list: list[str], *args, **kwargs
    ) -> list[list[float]]:  # noqa: ARG002
        """
        Call the embedding function
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        Call the chat completion function
        """
        raise NotImplementedError("Subclasses must implement this method")
