from __future__ import annotations

import logging
import os
from typing import Generator

import goemotions_local as goemotions
import memory_store as _memory_store
import session_store as _session_store
from config import BASE_DIR as CONFIG_BASE_DIR
from config import CHAT_MODEL_NAME, DEFAULT_LLM_PROVIDER, KNOWLEDGE_ENABLED, SHORT_TERM_LIMIT, STYLE_ENABLED
from config import USERS_DIR as CONFIG_USERS_DIR
from emotion import (
    detect_emotion_fluctuation,
    detect_lang,
    emotion_to_style,
    fluctuation_to_style,
    intensity_to_level,
    map_emotion_label,
    safe_analyze,
    summarize_emotion_trend,
    update_preferences,
)
from json_utils import (
    load_json,
    safe_extract_json_array,
    safe_extract_json_object,
    save_json,
)
from knowledge_store import build_knowledge_context
from llm_providers import (
    ModelRuntimeConfig,
    encode_texts,
    get_embedding_dimension,
    get_embedding_model,
    get_llm,
    make_model_config,
    stream_model_response,
)
from memory_store import (
    InterestMemoryStore,
    VectorIndexManager,
    clean_long_term,
    compact_long_memory,
    extract_long_term_interest,
    find_similar_interest,
    memory_exists,
    merge_interest_texts,
    require_faiss,
    retrieve_interests,
    save_interest,
    score_memory,
    smart_memory_filter,
    summarize_long_memories,
    update_long_term,
    update_mid_term,
    update_short_term,
)
from prompt_builder import build_messages
from style_store import build_style_context
from safety import (
    check_crisis,
    crisis_response,
    detect_crisis_keywords,
    detect_crisis_trend,
)
from session_store import SessionState, validate_user_id
from tools import (
    call_tool,
    tool_explain_code,
    tool_summarize,
    tool_translate,
    try_tool_call,
)


logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
_memory_store.logger = logger
_session_store.logger = logger

# Backward-compatible alias for older tests/imports.
_stream_model_response = stream_model_response

# Backward-compatible mutable paths. Older tests/tools set chatbot.USERS_DIR directly.
BASE_DIR = CONFIG_BASE_DIR
USERS_DIR = CONFIG_USERS_DIR


def _sync_user_paths() -> None:
    _session_store.USERS_DIR = USERS_DIR


def user_dir(user_id: str):
    _sync_user_paths()
    return _session_store.user_dir(user_id)


def user_paths(user_id: str):
    _sync_user_paths()
    return _session_store.user_paths(user_id)


def user_file_lock(user_id: str):
    _sync_user_paths()
    return _session_store.user_file_lock(user_id)


def load_state(user_id: str) -> SessionState:
    _sync_user_paths()
    return _session_store.load_state(user_id)


def persist_state(state: SessionState) -> None:
    _sync_user_paths()
    _session_store.persist_state(state)


class SessionStore(_session_store.SessionStore):
    def session(self, user_id: str):
        _sync_user_paths()
        return super().session(user_id)


def chat(
    state: SessionState,
    user_text: str,
    model_config: ModelRuntimeConfig | None = None,
    use_knowledge: bool | None = None,
    use_style: bool | None = None,
) -> Generator[str, None, None]:
    user_text = user_text.strip()
    if not user_text:
        yield "请输入需要分析或回复的内容。"
        return

    lang = detect_lang(user_text)
    update_preferences(state, user_text, lang)
    emo_label, emo_score = safe_analyze(user_text, lang)

    crisis_reply = check_crisis(state, user_text, emo_label, emo_score, lang)
    smart_memory_filter(state, user_text, emo_label, emo_score)

    if crisis_reply is not None:
        update_short_term(state, user_text, crisis_reply)
        yield crisis_reply
        return

    config = model_config or ModelRuntimeConfig()
    should_use_knowledge = KNOWLEDGE_ENABLED if use_knowledge is None else bool(use_knowledge)
    should_use_style = STYLE_ENABLED if use_style is None else bool(use_style)
    knowledge_context = build_knowledge_context(user_text) if should_use_knowledge else ""
    style_context = build_style_context(user_text) if should_use_style else ""
    messages = build_messages(
        state,
        user_text,
        emo_label,
        emo_score,
        knowledge_context=knowledge_context,
        style_context=style_context,
    )
    full_messages = [messages[0], *state.history, messages[1]]

    collected_chunks: list[str] = []
    try:
        for chunk in stream_model_response(full_messages, config):
            if chunk:
                collected_chunks.append(chunk)
                yield chunk
    except Exception as exc:
        logger.exception("模型回复生成失败。provider=%s model=%s", config.provider, config.model)
        error_reply = f"模型调用失败：{exc}"
        update_short_term(state, user_text, error_reply)
        yield error_reply
        return

    full_reply = "".join(collected_chunks).strip()
    tool_result = try_tool_call(full_reply)
    final_reply = tool_result if tool_result is not None else full_reply

    if tool_result is not None:
        yield "\n" + tool_result

    update_short_term(state, user_text, final_reply)


def chat_sync(
    state: SessionState,
    user_text: str,
    model_config: ModelRuntimeConfig | None = None,
    use_knowledge: bool | None = None,
    use_style: bool | None = None,
) -> str:
    return "".join(chat(
        state,
        user_text,
        model_config=model_config,
        use_knowledge=use_knowledge,
        use_style=use_style,
    ))


session_store = SessionStore()


def handle_user_message(
    user_id: str,
    user_text: str,
    model_config: ModelRuntimeConfig | None = None,
    use_knowledge: bool | None = None,
    use_style: bool | None = None,
) -> str:
    with session_store.session(user_id) as state:
        return chat_sync(
            state,
            user_text,
            model_config=model_config,
            use_knowledge=use_knowledge,
            use_style=use_style,
        )


def handle_user_message_stream(
    user_id: str,
    user_text: str,
    model_config: ModelRuntimeConfig | None = None,
    use_knowledge: bool | None = None,
    use_style: bool | None = None,
) -> Generator[str, None, None]:
    with session_store.session(user_id) as state:
        yield from chat(
            state,
            user_text,
            model_config=model_config,
            use_knowledge=use_knowledge,
            use_style=use_style,
        )


def main() -> None:
    user_id = os.getenv("CHATBOT_USER_ID", "local")
    print(f"对话开始（用户：{user_id}）。输入 'exit' 或按 Ctrl+C 退出。\n")

    while True:
        try:
            user_text = input("您：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break

        print("助手：", end="", flush=True)
        for chunk in handle_user_message_stream(user_id, user_text):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
