import logging

from app_logging import clear_logs, get_log_text, setup_gui_logging


def test_gui_logging_captures_and_clears_messages():
    setup_gui_logging()
    logger = logging.getLogger("test.gui")

    logger.info("hello log panel")

    assert "hello log panel" in get_log_text()
    assert clear_logs() == "暂无日志"
    assert get_log_text() == "暂无日志"
