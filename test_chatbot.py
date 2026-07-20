"""
test_chatbot.py
针对 chatbot.py 中纯逻辑/接近纯逻辑函数的单元测试。

覆盖范围（按之前讨论的优先级）：
  1. safe_extract_json_object  — JSON 解析的边界情况
  2. score_memory / smart_memory_filter — 记忆评分阈值边界
  3. InterestMemoryStore — 精确查重 + 脏标记
  4. safe_analyze — 异常/空结果兜底
  5. clean_long_term — 过期清理 + 格式错误日志化
  6. 情绪辅助函数 — map_emotion_label / intensity_to_level / detect_emotion_fluctuation

运行方式：
  pytest test_chatbot.py -v
  或不装 pytest 时：
  python3 -m unittest test_chatbot -v
"""

import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import conftest  # noqa: F401  (确保 stub 在 chatbot 导入前注册)
import chatbot
import memory_store
from prompt_builder import build_messages


# ═══════════════════════════════════════════════════════════
# 1. safe_extract_json_object
# ═══════════════════════════════════════════════════════════

class TestSafeExtractJsonObject(unittest.TestCase):

    def test_plain_json_object(self):
        result = chatbot.safe_extract_json_object('{"tool": "translate"}')
        self.assertEqual(result, {"tool": "translate"})

    def test_json_with_prefix_and_suffix_text(self):
        text = '这是回复前缀 {"tool": "summarize", "args": {}} 后面还有废话'
        result = chatbot.safe_extract_json_object(text)
        self.assertEqual(result, {"tool": "summarize", "args": {}})

    def test_nested_json_takes_first_complete_object_only(self):
        # 原版贪婪正则会把两个对象之间的内容也吞进去，
        # 修复后应该只取第一个完整对象，不被后面的内容污染。
        text = '{"a": {"b": 1}, "c": 2} {"d": 3}'
        result = chatbot.safe_extract_json_object(text)
        self.assertEqual(result, {"a": {"b": 1}, "c": 2})

    def test_no_json_present_returns_none(self):
        result = chatbot.safe_extract_json_object("纯文本，没有任何 JSON 结构")
        self.assertIsNone(result)

    def test_malformed_json_returns_none(self):
        result = chatbot.safe_extract_json_object('{"tool": "translate", }')
        self.assertIsNone(result)

    def test_json_array_at_top_level_is_not_a_dict_returns_none(self):
        # 顶层是数组而非对象，应返回 None（函数只接受 dict）
        result = chatbot.safe_extract_json_object('[1, 2, 3]')
        self.assertIsNone(result)

    def test_brace_inside_string_value_does_not_break_parsing(self):
        text = '{"text": "包含 { 大括号 } 的字符串"}'
        result = chatbot.safe_extract_json_object(text)
        self.assertEqual(result, {"text": "包含 { 大括号 } 的字符串"})

    def test_multiple_candidate_starts_skips_invalid_and_finds_valid(self):
        # 第一个 '{' 起始位置解析失败，应该继续尝试下一个 '{'
        text = '{not valid json at all {"tool": "translate", "args": {}}'
        result = chatbot.safe_extract_json_object(text)
        self.assertEqual(result, {"tool": "translate", "args": {}})

    def test_empty_string_returns_none(self):
        self.assertIsNone(chatbot.safe_extract_json_object(""))


class TestChatStreamingFallback(unittest.TestCase):

    def test_empty_model_stream_yields_a_fallback_message(self):
        state = chatbot.SessionState()
        with patch.object(chatbot, "safe_analyze", return_value=("neutral", 0.0)), \
            patch.object(chatbot, "smart_memory_filter", return_value="discard"), \
            patch.object(chatbot, "stream_model_response", return_value=iter(())):
            chunks = list(chatbot.chat(state, "test message"))

        self.assertEqual(len(chunks), 1)
        self.assertIn("没有返回可显示的文本", chunks[0])
        self.assertEqual(state.history[-1]["content"], chunks[0])


# ═══════════════════════════════════════════════════════════
# 2. score_memory / smart_memory_filter — 阈值边界
# ═══════════════════════════════════════════════════════════

class TestScoreMemory(unittest.TestCase):

    def test_neutral_low_score_no_bonus(self):
        # emo_score=0.1, neutral, 无关键词 -> 0.1 * 2 = 0.2
        score = chatbot.score_memory("今天天气不错", "neutral", 0.1)
        self.assertAlmostEqual(score, 0.2)

    def test_negative_emotion_adds_bonus(self):
        # sadness 命中负面情绪 bonus(+1)：0.1*2 + 1 = 1.2
        score = chatbot.score_memory("有点难过", "sadness", 0.1)
        self.assertAlmostEqual(score, 1.2)

    def test_memory_keyword_adds_bonus(self):
        # "我想" 命中 MEMORY_KEYWORDS（+2）：0.1*2 + 2 = 2.2
        score = chatbot.score_memory("我想去旅行", "neutral", 0.1)
        self.assertAlmostEqual(score, 2.2)

    def test_personal_keyword_adds_bonus(self):
        # "我是" 命中 PERSONAL_KEYWORDS（+1.5）：0.1*2 + 1.5 = 1.7
        score = chatbot.score_memory("我是程序员", "neutral", 0.1)
        self.assertAlmostEqual(score, 1.7)

    def test_memory_query_does_not_receive_keyword_bonus(self):
        score = chatbot.score_memory("我喜欢什么？", "neutral", 0.1)
        self.assertAlmostEqual(score, 0.2)

    def test_all_bonuses_stack(self):
        # "我喜欢" 同时命中 MEMORY_KEYWORDS(+2) 和负面情绪(+1)
        # 0.5*2 + 1(负面) + 2(memory kw) = 4.0 —— 正好等于长期阈值
        score = chatbot.score_memory("我喜欢一个人难过", "sadness", 0.5)
        self.assertAlmostEqual(score, 4.0)

    def test_threshold_boundary_long_term_inclusive(self):
        # score 恰好等于阈值 4.0 时应该判定为 long（>= 是包含边界的）
        # This test validates classification boundaries, not vector-index loading.
        with patch.object(memory_store, "save_interest", return_value="added"):
            result = chatbot.smart_memory_filter(
                self._fresh_state(), "我喜欢一个人难过", "sadness", 0.5
            )
        self.assertEqual(result, "long")

    def test_threshold_boundary_mid_term_inclusive(self):
        # score 恰好等于 2.0 时应该判定为 mid
        # "我想" 命中 memory keyword(+2)，emo_score=0 -> 0*2+2 = 2.0
        result = chatbot.smart_memory_filter(
            self._fresh_state(), "我想看看", "neutral", 0.0
        )
        self.assertEqual(result, "mid")

    def test_below_mid_threshold_is_discarded(self):
        # 无关键词、neutral、低分 -> score < 2.0 -> discard
        result = chatbot.smart_memory_filter(
            self._fresh_state(), "嗯嗯好的", "neutral", 0.1
        )
        self.assertEqual(result, "discard")

    @staticmethod
    def _fresh_state():
        return chatbot.SessionState()


# ═══════════════════════════════════════════════════════════
# 3. InterestMemoryStore — O(1) 精确查重 + 脏标记
# ═══════════════════════════════════════════════════════════

class TestInterestMemoryStore(unittest.TestCase):

    def test_empty_store_is_falsy(self):
        store = chatbot.InterestMemoryStore()
        self.assertFalse(store)
        self.assertEqual(len(store), 0)

    def test_load_populates_exact_set(self):
        store = chatbot.InterestMemoryStore()
        store.load([{"text": "我喜欢编程", "time": "2026-01-01T00:00:00"}])
        self.assertTrue(store.exact_exists("我喜欢编程"))
        self.assertFalse(store.exact_exists("我喜欢音乐"))

    def test_load_marks_dirty(self):
        store = chatbot.InterestMemoryStore()
        store.load([{"text": "示例", "time": "2026-01-01T00:00:00"}])
        self.assertTrue(store.dirty)

    def test_mark_clean_resets_dirty_flag(self):
        store = chatbot.InterestMemoryStore()
        store.load([{"text": "示例", "time": "2026-01-01T00:00:00"}])
        store.mark_clean()
        self.assertFalse(store.dirty)

    def test_append_updates_exact_set_and_marks_dirty(self):
        store = chatbot.InterestMemoryStore()
        store.load([])
        store.mark_clean()
        self.assertFalse(store.dirty)

        store.append({"text": "我喜欢音乐", "time": "2026-01-01T00:00:00"})
        self.assertTrue(store.exact_exists("我喜欢音乐"))
        self.assertTrue(store.dirty)   # 追加后应该重新标脏
        self.assertEqual(len(store), 1)

    def test_append_strips_whitespace_in_text(self):
        store = chatbot.InterestMemoryStore()
        store.append({"text": "  我喜欢跑步  "})
        self.assertTrue(store.exact_exists("我喜欢跑步"))
        self.assertFalse(store.exact_exists("  我喜欢跑步  "))

    def test_append_with_empty_text_is_ignored(self):
        store = chatbot.InterestMemoryStore()
        store.append({"text": "   "})
        self.assertEqual(len(store), 0)

    def test_append_with_missing_text_key_is_ignored(self):
        store = chatbot.InterestMemoryStore()
        store.append({"time": "2026-01-01T00:00:00"})
        self.assertEqual(len(store), 0)

    def test_items_reflects_insertion_order(self):
        store = chatbot.InterestMemoryStore()
        store.append({"text": "第一条"})
        store.append({"text": "第二条"})
        texts = [item["text"] for item in store.items]
        self.assertEqual(texts, ["第一条", "第二条"])


# ═══════════════════════════════════════════════════════════
# 4. safe_analyze — 异常/空结果兜底
# ═══════════════════════════════════════════════════════════

class TestSafeAnalyze(unittest.TestCase):

    def test_normal_list_result_zh(self):
        with patch.object(chatbot.goemotions, "predict_emotion_zh",
                           return_value=("joy", 0.9)):
            label, score = chatbot.safe_analyze("我很开心", "zh-cn")
        self.assertEqual(label, "joy")
        self.assertAlmostEqual(score, 0.9)

    def test_normal_dict_result_en(self):
        with patch.object(chatbot.goemotions, "predict_emotion_en",
                           return_value={"label": "sadness", "score": 0.7}):
            label, score = chatbot.safe_analyze("I feel sad", "en")
        self.assertEqual(label, "sadness")
        self.assertAlmostEqual(score, 0.7)

    def test_empty_list_result_falls_back_to_neutral(self):
        # 原版会在这里抛 ValueError；修复后应该安全回退
        with patch.object(chatbot.goemotions, "predict_emotion_en", return_value=[]):
            label, score = chatbot.safe_analyze("test", "en")
        self.assertEqual(label, "neutral")
        self.assertEqual(score, 0.0)

    def test_empty_tuple_result_falls_back_to_neutral(self):
        with patch.object(chatbot.goemotions, "predict_emotion_en", return_value=()):
            label, score = chatbot.safe_analyze("test", "en")
        self.assertEqual(label, "neutral")
        self.assertEqual(score, 0.0)

    def test_unknown_format_falls_back_to_neutral(self):
        with patch.object(chatbot.goemotions, "predict_emotion_en", return_value=42):
            label, score = chatbot.safe_analyze("test", "en")
        self.assertEqual(label, "neutral")
        self.assertEqual(score, 0.0)

    def test_model_raises_exception_falls_back_to_neutral(self):
        with patch.object(chatbot.goemotions, "predict_emotion_en",
                           side_effect=RuntimeError("model crashed")):
            label, score = chatbot.safe_analyze("test", "en")
        self.assertEqual(label, "neutral")
        self.assertEqual(score, 0.0)

    def test_raw_label_gets_mapped_to_primary_category(self):
        # "grief" 应该被映射成 "sadness"
        with patch.object(chatbot.goemotions, "predict_emotion_en",
                           return_value=("grief", 0.6)):
            label, _ = chatbot.safe_analyze("test", "en")
        self.assertEqual(label, "sadness")

    def test_none_score_defaults_to_zero(self):
        with patch.object(chatbot.goemotions, "predict_emotion_en",
                           return_value={"label": "joy", "score": None}):
            _, score = chatbot.safe_analyze("test", "en")
        self.assertEqual(score, 0.0)


# ═══════════════════════════════════════════════════════════
# 5. clean_long_term — 过期清理 + 格式错误处理
# ═══════════════════════════════════════════════════════════

class TestCleanLongTerm(unittest.TestCase):

    def test_recent_item_is_kept(self):
        recent_time = datetime.now().isoformat()
        items = [{"text": "最近的记忆", "time": recent_time}]
        result = chatbot.clean_long_term(items)
        self.assertEqual(len(result), 1)

    def test_expired_item_is_removed(self):
        old_time = (datetime.now() - timedelta(days=40)).isoformat()
        items = [{"text": "过期的记忆", "time": old_time}]
        result = chatbot.clean_long_term(items)
        self.assertEqual(len(result), 0)

    def test_item_at_exact_boundary_is_kept(self):
        # delta 略小于 30 天，应该保留（边界用 < expiry，不是 <=）
        boundary_time = (datetime.now() - timedelta(days=29, hours=23)).isoformat()
        items = [{"text": "边界记忆", "time": boundary_time}]
        result = chatbot.clean_long_term(items)
        self.assertEqual(len(result), 1)

    def test_missing_time_field_is_kept_with_warning(self):
        items = [{"text": "缺少时间字段"}]
        with self.assertLogs(chatbot.logger, level="WARNING") as log_ctx:
            result = chatbot.clean_long_term(items)
        self.assertEqual(len(result), 1)
        self.assertTrue(any("time" in msg for msg in log_ctx.output))

    def test_malformed_time_field_is_kept_with_warning(self):
        items = [{"text": "时间格式错误", "time": "not-a-valid-date"}]
        with self.assertLogs(chatbot.logger, level="WARNING") as log_ctx:
            result = chatbot.clean_long_term(items)
        self.assertEqual(len(result), 1)
        self.assertTrue(any("格式非法" in msg for msg in log_ctx.output))

    def test_mixed_valid_expired_and_malformed_items(self):
        recent_time = datetime.now().isoformat()
        old_time = (datetime.now() - timedelta(days=100)).isoformat()
        items = [
            {"text": "有效", "time": recent_time},
            {"text": "过期", "time": old_time},
            {"text": "格式错误"},  # 缺 time，保留
        ]
        result = chatbot.clean_long_term(items)
        texts = {item["text"] for item in result}
        self.assertEqual(texts, {"有效", "格式错误"})

    def test_empty_list_returns_empty_list(self):
        self.assertEqual(chatbot.clean_long_term([]), [])


# ═══════════════════════════════════════════════════════════
# 6. 情绪辅助函数
# ═══════════════════════════════════════════════════════════

class TestEmotionHelpers(unittest.TestCase):

    def test_map_emotion_label_known(self):
        self.assertEqual(chatbot.map_emotion_label("grief"), "sadness")
        self.assertEqual(chatbot.map_emotion_label("annoyance"), "anger")
        self.assertEqual(chatbot.map_emotion_label("gratitude"), "joy")

    def test_map_emotion_label_unknown_defaults_neutral(self):
        self.assertEqual(chatbot.map_emotion_label("some_unmapped_label"), "neutral")

    def test_map_emotion_label_none_defaults_neutral(self):
        self.assertEqual(chatbot.map_emotion_label(None), "neutral")

    def test_intensity_to_level_boundaries(self):
        self.assertEqual(chatbot.intensity_to_level(0.75), "high")
        self.assertEqual(chatbot.intensity_to_level(0.749), "medium")
        self.assertEqual(chatbot.intensity_to_level(0.40), "medium")
        self.assertEqual(chatbot.intensity_to_level(0.399), "low")
        self.assertEqual(chatbot.intensity_to_level(0.0), "low")

    def test_detect_emotion_fluctuation_no_history_is_stable_unknown(self):
        level, direction = chatbot.detect_emotion_fluctuation([], "joy", 0.5)
        self.assertEqual((level, direction), ("stable", "unknown"))

    def test_detect_emotion_fluctuation_small_delta_is_stable(self):
        history = [{"label": "joy", "score": 0.5}]
        level, _ = chatbot.detect_emotion_fluctuation(history, "joy", 0.55)
        self.assertEqual(level, "stable")

    def test_detect_emotion_fluctuation_large_delta_is_severe(self):
        history = [{"label": "joy", "score": 0.1}]
        level, _ = chatbot.detect_emotion_fluctuation(history, "sadness", 0.9)
        self.assertEqual(level, "severe")

    def test_detect_emotion_fluctuation_negative_emotion_direction(self):
        # sadness 分数升高 => 情况变差 (worse)
        history = [{"label": "sadness", "score": 0.2}]
        _, direction = chatbot.detect_emotion_fluctuation(history, "sadness", 0.8)
        self.assertEqual(direction, "worse")

    def test_detect_emotion_fluctuation_positive_emotion_direction(self):
        # joy 分数升高 => 情况变好 (better)
        history = [{"label": "joy", "score": 0.2}]
        _, direction = chatbot.detect_emotion_fluctuation(history, "joy", 0.8)
        self.assertEqual(direction, "better")


# ═══════════════════════════════════════════════════════════
# 7. memory_exists / extract_long_term_interest — 轻量集成
# ═══════════════════════════════════════════════════════════

class TestMemoryExistsAndInterestExtraction(unittest.TestCase):

    def test_memory_exists_false_on_empty_store(self):
        state = chatbot.SessionState()
        self.assertFalse(chatbot.memory_exists("随便什么", state))

    def test_memory_exists_true_on_exact_match_without_touching_faiss(self):
        # 关键：精确匹配命中时应直接返回 True，
        # 不应该走到 FAISS 检索那一步（即使 FAISS 不可用也不报错）
        state = chatbot.SessionState()
        state.interest_store.append({"text": "我喜欢爬山"})
        self.assertTrue(chatbot.memory_exists("我喜欢爬山", state))

    def test_extract_long_term_interest_matches_pattern(self):
        result = chatbot.extract_long_term_interest("我喜欢弹吉他")
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "我喜欢弹吉他")

    def test_extract_long_term_interest_no_pattern_returns_none(self):
        result = chatbot.extract_long_term_interest("今天天气怎么样")
        self.assertIsNone(result)

    def test_interest_question_is_not_extracted_as_a_preference(self):
        result = chatbot.extract_long_term_interest("我喜欢什么？")
        self.assertIsNone(result)

    def test_explicit_identity_is_saved_as_stable_profile(self):
        state = chatbot.SessionState()

        outcome = chatbot.smart_memory_filter(state, "我是学生", "neutral", 0.0)

        self.assertEqual(outcome, "stable")
        self.assertEqual(state.stable_profile[0]["text"], "我是学生")
        self.assertEqual(state.stable_profile[0]["kind"], "profile")
        self.assertEqual(state.memory_events[-1]["section"], "stable")
        self.assertEqual(state.memory_events[-1]["action"], "added")
        self.assertIn("身份", state.memory_events[-1]["reason"])

    def test_identity_question_is_not_saved_as_profile(self):
        state = chatbot.SessionState()

        outcome = chatbot.smart_memory_filter(state, "你觉得我是学生吗？", "neutral", 0.0)

        self.assertEqual(outcome, "discard")
        self.assertEqual(state.stable_profile, [])
        self.assertEqual(state.memory_events[-1]["action"], "skipped")
        self.assertIn("回忆查询", state.memory_events[-1]["reason"])

    def test_identity_fact_before_question_is_saved_as_profile(self):
        state = chatbot.SessionState()
        text = "我是一名学生，请问您觉得我应该如何为未来做准备？"

        outcome = chatbot.smart_memory_filter(state, text, "neutral", 0.0)

        self.assertEqual(outcome, "stable")
        self.assertEqual(state.stable_profile[0]["text"], "我是一名学生")
        self.assertEqual(state.memory_events[-1]["text"], "我是一名学生")
        self.assertIn("混合陈述与提问", state.memory_events[-1]["reason"])

    def test_profile_query_with_declarative_prefix_is_still_rejected(self):
        state = chatbot.SessionState()

        outcome = chatbot.smart_memory_filter(state, "我来自哪里？", "neutral", 0.0)

        self.assertEqual(outcome, "discard")
        self.assertEqual(state.stable_profile, [])

    def test_interest_fact_before_question_is_saved_without_question(self):
        state = chatbot.SessionState()
        text = "我喜欢编程，请问我应该学习哪些方向？"

        with patch.object(memory_store, "memory_exists", return_value=False), \
             patch.object(memory_store, "save_interest", return_value="added"):
            outcome = chatbot.smart_memory_filter(state, text, "neutral", 0.0)

        self.assertEqual(outcome, "long")
        self.assertEqual(state.long_memory[0]["text"], "我喜欢编程")
        self.assertEqual(state.memory_events[-1]["text"], "我喜欢编程")

    def test_english_profile_fact_before_question_is_saved(self):
        profile = memory_store.extract_personal_profile(
            "I am a student, how should I prepare for my future?"
        )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["text"], "I am a student")

    def test_latest_memory_receipt_describes_write_decision(self):
        state = chatbot.SessionState()
        chatbot.smart_memory_filter(state, "我是学生", "neutral", 0.0)

        receipt = chatbot.latest_memory_receipt(state)

        self.assertIn("稳定资料已新增", receipt)
        self.assertIn("我是学生", receipt)

    def test_extract_long_term_interest_english_pattern(self):
        result = chatbot.extract_long_term_interest("I love hiking on weekends")
        self.assertIsNotNone(result)

    def test_stable_profile_is_included_in_prompt_context(self):
        state = chatbot.SessionState()
        state.stable_profile.append({"text": "我是学生", "kind": "profile"})

        messages = build_messages(state, "帮我规划学习", "neutral", 0.0)

        assert "稳定资料" in messages[0]["content"]
        assert "我是学生" in messages[0]["content"]


class TestModelRuntimeConfig(unittest.TestCase):

    def test_local_provider_defaults_to_chat_model_name(self):
        config = chatbot.make_model_config(provider="local_hf", model="")
        self.assertEqual(config.normalized_provider(), "local_hf")
        self.assertEqual(config.resolved_model(), chatbot.CHAT_MODEL_NAME)

    def test_deepseek_defaults(self):
        config = chatbot.make_model_config(provider="deepseek", model="")
        self.assertEqual(config.resolved_model(), "deepseek-chat")
        self.assertEqual(config.resolved_base_url(), "https://api.deepseek.com")

    def test_explicit_api_key_takes_precedence(self):
        config = chatbot.make_model_config(provider="deepseek", api_key="  explicit-key  ")
        self.assertEqual(config.resolved_api_key(), "explicit-key")

    def test_unknown_provider_rejected_by_stream_router(self):
        config = chatbot.make_model_config(provider="unknown")
        with self.assertRaises(ValueError):
            list(chatbot._stream_model_response([], config))


if __name__ == "__main__":
    unittest.main(verbosity=2)
