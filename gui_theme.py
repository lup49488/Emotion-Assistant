from __future__ import annotations


THEME_CSS = """
#theme-mode {
    max-width: 360px;
    margin-bottom: 8px;
}
#theme-mode .wrap {
    gap: 6px;
}

#chat-workspace {
    align-items: stretch;
    gap: 0;
    min-height: 720px;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    overflow: hidden;
}
#chat-sidebar {
    background: var(--background-fill-secondary);
    border-right: 1px solid var(--border-color-primary);
    padding: 14px;
    gap: 10px;
}
#chat-sidebar h3 {
    margin: 0 0 2px;
    font-size: 16px;
}
#conversation-list {
    min-height: 260px;
    max-height: 420px;
    overflow-y: auto;
}
#conversation-list .wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
#conversation-list label {
    width: 100%;
    min-height: 38px;
    padding: 8px 10px;
    border: 1px solid transparent;
    border-radius: 6px;
    background: transparent;
}
#conversation-list label:hover {
    background: var(--background-fill-primary);
}
#conversation-list label:has(input:checked) {
    border-color: var(--border-color-primary);
    background: var(--background-fill-primary);
}
#conversation-list input {
    display: none;
}
#chat-main {
    min-width: 0;
    padding: 14px;
}
#conversation-chat {
    border: 0;
}
#chat-main .chatbot {
    min-height: 520px;
}
@media (max-width: 760px) {
    #chat-workspace {
        min-height: 0;
        flex-direction: column;
    }
    #chat-sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--border-color-primary);
    }
    #conversation-list {
        min-height: 140px;
        max-height: 220px;
    }
    #chat-main {
        padding: 10px;
    }
    #chat-main .chatbot {
        min-height: 420px;
    }
}
"""

REFRESH_CHART_THEME_JS = """
(userId, accessKey, endDate, theme, locale) => {
    const dark = theme === "dark";
    document.documentElement.classList.toggle("dark", dark);
    document.body.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    localStorage.setItem("chatbot-theme", theme);
    const detected = document.documentElement.lang || navigator.language || "zh-CN";
    return [userId, accessKey, endDate, theme, detected];
}
"""

LOAD_THEME_JS = """
(userId, accessKey, endDate, theme, locale) => {
    const saved = localStorage.getItem("chatbot-theme");
    const selected = saved === "dark" ? "dark" : "light";
    const dark = selected === "dark";
    document.documentElement.classList.toggle("dark", dark);
    document.body.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    const detected = document.documentElement.lang || navigator.language || "zh-CN";
    return [userId, accessKey, endDate, selected, detected];
}
"""
