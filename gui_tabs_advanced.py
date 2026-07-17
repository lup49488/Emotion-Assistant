from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import gradio as gr


@dataclass
class AdvancedTabComponents:
    advanced_tab: Any
    status_box: Any
    refresh_button: Any
    connection_button: Any
    save_model_config_button: Any
    connection_result: Any
    connection_detail_result: Any
    save_model_config_result: Any
    status_timer: Any
    local_dtype_input: Any
    local_attention_input: Any
    local_low_cpu_mem_input: Any
    local_compile_input: Any
    local_cpu_threads_input: Any
    save_local_runtime_button: Any
    refresh_local_runtime_button: Any
    local_runtime_status: Any
    local_runtime_save_result: Any
    log_box: Any
    refresh_logs_button: Any
    clear_logs_button: Any
    log_timer: Any


def build_advanced_tab(
    tr: Callable[[str], Any],
    *,
    initial_provider: str,
    initial_model: str,
    initial_base_url: str,
    local_dtype_choices: Sequence[str],
    local_attention_choices: Sequence[tuple[str, str]],
    local_dtype: str,
    local_attention: str,
    local_low_cpu_memory: bool,
    local_compile: bool,
    local_cpu_threads: int,
    build_status_text: Callable[[str, str, str], str],
    build_local_runtime_text: Callable[[str, str, bool, bool, int], str],
    get_log_text: Callable[[], str],
) -> AdvancedTabComponents:
    """Render the advanced runtime and diagnostic controls."""
    with gr.Tab(tr("tab_advanced"), id="advanced", visible=False) as advanced_tab:
        with gr.Accordion(tr("运行状态"), open=True):
            status_box = gr.Textbox(
                label=tr("状态"),
                value=build_status_text(initial_provider, initial_model, initial_base_url),
                lines=9,
                interactive=False,
            )
            with gr.Row():
                refresh_button = gr.Button(tr("刷新状态"))
                connection_button = gr.Button(tr("测试连接"))
                save_model_config_button = gr.Button(tr("保存模型配置"))
            connection_result = gr.Textbox(label=tr("连接测试摘要"), interactive=False)
            connection_detail_result = gr.Textbox(
                label=tr("连接测试详情"), value=tr("尚未测试连接。"), lines=9, interactive=False
            )
            save_model_config_result = gr.Textbox(label=tr("模型配置保存结果"), interactive=False)
            status_timer = gr.Timer(value=2.0)

        with gr.Accordion(tr("本地模型运行配置"), open=False):
            with gr.Row():
                local_dtype_input = gr.Dropdown(
                    label=tr("模型精度"), choices=local_dtype_choices,
                    value=local_dtype if local_dtype in local_dtype_choices else "auto",
                )
                local_attention_input = gr.Dropdown(
                    label=tr("注意力实现"), choices=local_attention_choices,
                    value=local_attention or "", allow_custom_value=True,
                )
            with gr.Row():
                local_low_cpu_mem_input = gr.Checkbox(label=tr("低内存加载"), value=local_low_cpu_memory)
                local_compile_input = gr.Checkbox(label=tr("启用 torch.compile"), value=local_compile)
                local_cpu_threads_input = gr.Number(
                    label=tr("CPU 线程数（0 为默认）"), value=local_cpu_threads,
                    minimum=0, maximum=512, precision=0,
                )
            with gr.Row():
                save_local_runtime_button = gr.Button(tr("保存本地运行配置"))
                refresh_local_runtime_button = gr.Button(tr("查看当前配置"))
            local_runtime_status = gr.Textbox(
                label=tr("本地模型配置状态"),
                value=lambda: build_local_runtime_text(
                    local_dtype, local_attention or "", local_low_cpu_memory,
                    local_compile, local_cpu_threads,
                ),
                lines=8,
                interactive=False,
            )
            local_runtime_save_result = gr.Textbox(
                label=tr("保存结果"), value=tr("尚未保存本地运行配置。"), lines=3, interactive=False
            )

        with gr.Accordion(tr("日志面板"), open=False):
            log_box = gr.Textbox(label=tr("最近日志"), value=get_log_text, lines=14, interactive=False)
            with gr.Row():
                refresh_logs_button = gr.Button(tr("刷新日志"))
                clear_logs_button = gr.Button(tr("清空日志"))
            log_timer = gr.Timer(value=3.0)

    return AdvancedTabComponents(
        advanced_tab, status_box, refresh_button, connection_button, save_model_config_button,
        connection_result, connection_detail_result, save_model_config_result, status_timer,
        local_dtype_input, local_attention_input, local_low_cpu_mem_input, local_compile_input,
        local_cpu_threads_input, save_local_runtime_button, refresh_local_runtime_button,
        local_runtime_status, local_runtime_save_result, log_box, refresh_logs_button,
        clear_logs_button, log_timer,
    )
