from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr


@dataclass
class DataTabComponents:
    new_access_key_input: Any
    change_access_key_button: Any
    export_user_data_button: Any
    export_user_data_status: Any
    export_user_data_file: Any
    admin_recovery_group: Any
    admin_recovery_key_input: Any
    admin_new_access_key_input: Any
    admin_recovery_button: Any
    admin_recovery_status_box: Any
    stable_profile_input: Any
    add_stable_profile_button: Any
    load_stable_profile_button: Any
    stable_profile_box: Any
    stable_advanced_controls: Any
    stable_profile_editor: Any
    save_stable_profile_button: Any
    clear_stable_profile_button: Any
    memory_section: Any
    load_memory_button: Any
    memory_box: Any
    memory_advanced_controls: Any
    save_memory_button: Any
    clear_memory_button: Any
    refresh_memory_events_button: Any
    memory_editor: Any
    memory_event_box: Any
    backup_memory_button: Any
    restore_memory_mode: Any
    restore_memory_file: Any
    restore_memory_button: Any
    memory_backup_status: Any
    memory_backup_file: Any
    memory_safety_backup_file: Any


def build_data_tab(
    tr: Callable[[str], Any], admin_recovery_status: Callable[[], str]
) -> DataTabComponents:
    """Render the personal-data tab; event wiring stays in Web_GUI."""
    with gr.Tab(tr("tab_data"), id="data"):
        with gr.Accordion(tr("用户访问"), open=True):
            new_access_key_input = gr.Textbox(
                label=tr("新访问密码"),
                value="",
                type="password",
                placeholder=tr("修改密码时填写，至少 8 位，可包含特殊符号"),
            )
            with gr.Row():
                change_access_key_button = gr.Button(tr("修改密码"))
                export_user_data_button = gr.Button(tr("导出用户数据"))
            export_user_data_status = gr.Textbox(
                label=tr("数据导出状态"), value=tr("尚未导出。"), lines=3, interactive=False
            )
            export_user_data_file = gr.File(label=tr("导出的用户数据"), interactive=False)
            with gr.Column(visible=False) as admin_recovery_group:
                with gr.Accordion(tr("管理员恢复"), open=False):
                    admin_recovery_key_input = gr.Textbox(
                        label=tr("管理员恢复密钥"), value="", type="password",
                        placeholder=tr("仅从服务器环境变量读取的恢复密钥"),
                    )
                    admin_new_access_key_input = gr.Textbox(
                        label=tr("恢复后的新密码"), value="", type="password",
                        placeholder=tr("修改密码时填写，至少 8 位，可包含特殊符号"),
                    )
                    admin_recovery_button = gr.Button(tr("重置访问密码"), variant="stop")
                    admin_recovery_status_box = gr.Textbox(
                        label=tr("管理员恢复状态"), value=tr(admin_recovery_status()),
                        lines=3, interactive=False,
                    )

        with gr.Accordion(tr("稳定资料"), open=False):
            stable_profile_input = gr.Textbox(
                label=tr("新增稳定资料"),
                placeholder=tr("例如：我是学生；我喜欢被简洁地称呼；我来自北京"),
                lines=2,
            )
            with gr.Row():
                add_stable_profile_button = gr.Button(tr("添加资料"))
                load_stable_profile_button = gr.Button(tr("查看资料"))
            stable_profile_box = gr.Textbox(
                label=tr("稳定资料状态"), value=tr("点击“查看资料”加载当前用户的稳定资料。"),
                lines=10, interactive=False,
            )
            with gr.Column(visible=False) as stable_advanced_controls:
                stable_profile_editor = gr.Textbox(
                    label=tr("稳定资料 JSON 编辑区"), value="[]", lines=10, interactive=True
                )
                with gr.Row():
                    save_stable_profile_button = gr.Button(tr("保存编辑"))
                    clear_stable_profile_button = gr.Button(tr("清空稳定资料"), variant="stop")

        with gr.Accordion(tr("记忆管理"), open=False):
            memory_section = gr.Radio(
                label=tr("清理范围"),
                choices=[
                    (tr("短期对话"), "history"), (tr("情绪记忆"), "emotion"),
                    (tr("长期记忆"), "long"), (tr("兴趣记忆"), "interest"),
                    (tr("稳定资料"), "stable"), (tr("全部记忆"), "all"),
                ],
                value="history",
            )
            load_memory_button = gr.Button(tr("查看记忆"))
            memory_box = gr.Textbox(
                label=tr("当前用户记忆"), value=tr("点击“查看记忆”加载当前用户的记忆。"),
                lines=18, interactive=False,
            )
            with gr.Column(visible=False) as memory_advanced_controls:
                with gr.Row():
                    save_memory_button = gr.Button(tr("保存修改"))
                    clear_memory_button = gr.Button(tr("清理所选记忆"))
                    refresh_memory_events_button = gr.Button(tr("查看写入记录"))
                memory_editor = gr.Textbox(label=tr("记忆 JSON 编辑区"), value="[]", lines=14, interactive=True)
                memory_event_box = gr.Textbox(
                    label=tr("记忆写入记录"), value=tr("点击“查看写入记录”加载最近的记忆判断。"),
                    lines=14, interactive=False,
                )
                with gr.Accordion(tr("记忆备份与恢复"), open=False):
                    with gr.Row():
                        backup_memory_button = gr.Button(tr("备份全部记忆"))
                        restore_memory_mode = gr.Radio(
                            choices=[(tr("合并并去重"), "merge"), (tr("覆盖当前记忆"), "replace")],
                            value="merge", label=tr("恢复模式"),
                        )
                    restore_memory_file = gr.File(
                        label=tr("选择记忆备份 JSON"), file_types=[".json"], type="filepath"
                    )
                    restore_memory_button = gr.Button(tr("恢复记忆"), variant="primary")
                    memory_backup_status = gr.Textbox(
                        label=tr("备份 / 恢复状态"), value=tr("尚未执行备份或恢复。"), lines=4, interactive=False
                    )
                    memory_backup_file = gr.File(label=tr("生成的备份文件"), interactive=False)
                    memory_safety_backup_file = gr.File(label=tr("恢复前安全备份"), interactive=False)

    return DataTabComponents(
        new_access_key_input, change_access_key_button, export_user_data_button,
        export_user_data_status, export_user_data_file, admin_recovery_group,
        admin_recovery_key_input, admin_new_access_key_input, admin_recovery_button,
        admin_recovery_status_box, stable_profile_input, add_stable_profile_button,
        load_stable_profile_button, stable_profile_box, stable_advanced_controls,
        stable_profile_editor, save_stable_profile_button, clear_stable_profile_button,
        memory_section, load_memory_button, memory_box, memory_advanced_controls,
        save_memory_button, clear_memory_button, refresh_memory_events_button,
        memory_editor, memory_event_box, backup_memory_button, restore_memory_mode,
        restore_memory_file, restore_memory_button, memory_backup_status,
        memory_backup_file, memory_safety_backup_file,
    )
