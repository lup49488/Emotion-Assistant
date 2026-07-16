from __future__ import annotations


THEME_CSS = """
#theme-mode {
    max-width: 360px;
    margin-bottom: 8px;
}
#theme-mode .wrap {
    gap: 6px;
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
