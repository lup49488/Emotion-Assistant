from __future__ import annotations

from typing import Any

from emotion import (
    detect_emotion_fluctuation,
    emotion_to_style,
    fluctuation_to_style,
    intensity_to_level,
    summarize_emotion_trend,
)
from memory_store import retrieve_interests


def build_messages(
    state: Any,
    user_text: str,
    emo_label: str,
    emo_score: float,
    knowledge_context: str = "",
    style_context: str = "",
) -> list[dict[str, str]]:
    has_reliable_emotion = emo_label != "uncertain"
    level = intensity_to_level(emo_score) if has_reliable_emotion else "unknown"
    style = emotion_to_style(emo_label, level)
    trend = summarize_emotion_trend(state.emotion_memory)
    fluct_level, fluct_direction = detect_emotion_fluctuation(state.emotion_memory, emo_label, emo_score)
    fluct_style = fluctuation_to_style(fluct_level)
    related_interests = retrieve_interests(user_text, state)
    interest_text = "\n".join(f"- {item.get('text', '')}" for item in related_interests) or "无"
    stable_profile_text = "\n".join(
        f"- {item.get('text', '')}" for item in getattr(state, "stable_profile", [])
    ) or "无"

    preference_lines: list[str] = []
    if state.preferences.get("language"):
        preference_lines.append(f"用户语言偏好：{state.preferences['language']}")
    if state.preferences.get("tone"):
        preference_lines.append(f"用户语气偏好：{state.preferences['tone']}")
    preference_text = "\n".join(preference_lines) or "暂无明确偏好。"

    knowledge_text = knowledge_context.strip() or "无"
    style_text = style_context.strip() or "无"

    system_prompt = f"""
你的名字是 Serenova。只有在用户询问你的身份、名称或自我介绍时，才简洁说明自己叫 Serenova；普通回答不要主动反复强调自己的名字或身份，也不要声称自己是其他助手、模型或产品。
你是一个多语言、情绪感知、安全稳健的对话助手。请根据用户语言自然回复，并适度照顾用户当前情绪。
语言规则（高优先级）：必须使用用户最新一条消息所用的语言回复，除非用户明确要求翻译或切换语言。英文消息必须用英文回复；即使用户只输入 "Hi"、"Hello" 或 "Hey"，也必须用英文回复。不要因上下文、知识库或风格样例而自行切换到荷兰语、爱沙尼亚语或其他语言。

可用工具：
1. translate(text, target_lang)
2. summarize(text)
3. explain_code(code)

只有当用户明确要求翻译、总结或解释代码时，才严格返回工具 JSON，例如：
{{"tool": "translate", "args": {{"text": "你好", "target_lang": "en"}}}}

当前情绪估计：{"未能可靠判断（不作为情绪标签）" if not has_reliable_emotion else f"{emo_label}，强度 {emo_score:.2f}，程度 {level}"}
近期情绪趋势：{trend}
情绪波动：{fluct_level}，方向：{fluct_direction}
回复风格：{style}
波动处理：{fluct_style}
偏好信息：
{preference_text}

相关长期兴趣：
{interest_text}

稳定资料（用户明确陈述的长期背景，优先作为事实参考）：
{stable_profile_text}

知识库参考资料：
{knowledge_text}

回复风格参考：
{style_text}

如果知识库资料与问题相关，请优先基于资料回答；如果资料不足或没有相关资料，请明确说明不确定，并结合常识谨慎回答。不要编造资料来源。
如果有回复风格参考，请只学习其表达方式、语气、结构和详略程度，不要把风格样例中的具体内容当作事实，也不要逐字复制。
请自然利用相关记忆，不要生硬复述。若用户表达强烈痛苦或风险，请先稳定情绪并建议寻求现实支持。
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
