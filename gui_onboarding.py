from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr

from gui_auth import authorize_or_message
from onboarding_store import mark_onboarding_completed, onboarding_completed


ONBOARDING_TITLE = "首次使用引导"
ONBOARDING_GUIDE_TEXT = (
    "1. 在下方设置用户名和至少 8 位访问密码。\n\n"
    "2. 选择提供者和模型；想要使用自己的 API 时可在高级模式填写服务地址和 API Key。\n\n"
    "3. 开始使用。"
)
ONBOARDING_COMPLETE_LABEL = "完成引导"


@dataclass
class OnboardingComponents:
    guide: Any
    complete_button: Any


def build_onboarding(
    tr: Callable[[str], Any], *, show_on_first_use: bool
) -> OnboardingComponents:
    """Render the lightweight, session-only first-use guide."""
    with gr.Accordion(
        tr(ONBOARDING_TITLE), open=show_on_first_use, visible=show_on_first_use
    ) as guide:
        gr.Markdown(tr(ONBOARDING_GUIDE_TEXT))
        complete_button = gr.Button(tr(ONBOARDING_COMPLETE_LABEL))
    return OnboardingComponents(guide=guide, complete_button=complete_button)


def onboarding_visibility_after_login(user_id: str, access_key: str) -> Any:
    authorized_user, error = authorize_or_message(user_id, access_key)
    if error or not authorized_user:
        return gr.update()
    return gr.update(visible=not onboarding_completed(authorized_user))


def dismiss_onboarding(user_id: str, access_key: str) -> Any:
    authorized_user, error = authorize_or_message(user_id, access_key)
    if error or not authorized_user:
        return gr.update(visible=True)
    mark_onboarding_completed(authorized_user)
    return gr.update(visible=False)
