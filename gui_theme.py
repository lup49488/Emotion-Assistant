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
(userId, accessKey, endDate, theme) => {
    const dark = theme === "dark";
    document.documentElement.classList.toggle("dark", dark);
    document.body.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    localStorage.setItem("chatbot-theme", theme);
    return [userId, accessKey, endDate, theme];
}
"""

LOAD_THEME_JS = """
(userId, accessKey, endDate, theme) => {
    const saved = localStorage.getItem("chatbot-theme");
    const selected = saved === "dark" ? "dark" : "light";
    const dark = selected === "dark";
    document.documentElement.classList.toggle("dark", dark);
    document.body.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    return [userId, accessKey, endDate, selected];
}
"""
