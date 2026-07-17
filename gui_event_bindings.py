from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def bind_gui_events(
    ui: Any,
    callbacks: Mapping[str, Callable[..., Any]],
    *,
    sync_locale_js: str,
    refresh_chart_theme_js: str,
    load_theme_js: str,
) -> None:
    """Attach all Gradio events after every UI component has been assembled."""
    callback = callbacks.__getitem__

    ui.interface_mode_input.change(
        fn=callback("interface_mode_visibility"),
        inputs=ui.interface_mode_input,
        outputs=[
            ui.advanced_chat_settings, ui.memory_advanced_controls,
            ui.stable_advanced_controls, ui.knowledge_advanced_controls,
            ui.style_advanced_controls, ui.admin_recovery_group, ui.advanced_tab,
            ui.main_tabs,
        ],
        show_progress="hidden",
    )
    ui.provider_input.change(
        fn=callback("provider_changed"),
        inputs=[ui.provider_input, ui.api_key_input, ui.locale_probe_input],
        outputs=[ui.model_input, ui.base_url_input, ui.status_box],
        js=sync_locale_js,
    )
    ui.refresh_button.click(
        fn=callback("refresh_status"),
        inputs=[ui.provider_input, ui.model_input, ui.base_url_input, ui.api_key_input, ui.locale_probe_input],
        outputs=ui.status_box,
        js=sync_locale_js,
    )
    ui.status_timer.tick(
        fn=callback("refresh_status"),
        inputs=[ui.provider_input, ui.model_input, ui.base_url_input, ui.api_key_input, ui.locale_probe_input],
        outputs=ui.status_box,
        js=sync_locale_js,
    )
    ui.locale_status_timer.tick(
        fn=callback("relocalize_status_values_and_locale"),
        inputs=[
            ui.status_box, ui.connection_result, ui.connection_detail_result,
            ui.save_model_config_result, ui.local_runtime_status, ui.local_runtime_save_result,
            ui.login_status_box, ui.access_key_status, ui.export_user_data_status,
            ui.admin_recovery_status_box, ui.locale_probe_input,
        ],
        outputs=[
            ui.status_box, ui.connection_result, ui.connection_detail_result,
            ui.save_model_config_result, ui.local_runtime_status, ui.local_runtime_save_result,
            ui.login_status_box, ui.access_key_status, ui.export_user_data_status,
            ui.admin_recovery_status_box, ui.locale_probe_input,
        ],
        js=sync_locale_js,
        show_progress="hidden",
    )
    ui.connection_button.click(
        fn=callback("test_model_connection_and_refresh"),
        inputs=[
            ui.provider_input, ui.model_input, ui.base_url_input, ui.api_key_input,
            ui.temperature_input, ui.top_p_input, ui.max_new_tokens_input, ui.locale_probe_input,
        ],
        outputs=[ui.connection_result, ui.connection_detail_result, ui.status_box],
        js=sync_locale_js,
    )
    ui.save_model_config_button.click(
        fn=callback("save_model_config_and_refresh"),
        inputs=[
            ui.provider_input, ui.model_input, ui.base_url_input, ui.api_key_input,
            ui.temperature_input, ui.top_p_input, ui.max_new_tokens_input, ui.locale_probe_input,
        ],
        outputs=[ui.save_model_config_result, ui.status_box],
        js=sync_locale_js,
    )
    ui.save_local_runtime_button.click(
        fn=callback("save_local_runtime_config_and_refresh"),
        inputs=[
            ui.local_dtype_input, ui.local_attention_input, ui.local_low_cpu_mem_input,
            ui.local_compile_input, ui.local_cpu_threads_input, ui.locale_probe_input,
        ],
        outputs=[ui.local_runtime_save_result, ui.local_runtime_status],
        js=sync_locale_js,
    )
    ui.refresh_local_runtime_button.click(
        fn=callback("build_local_runtime_config_text_localized"),
        inputs=[
            ui.local_dtype_input, ui.local_attention_input, ui.local_low_cpu_mem_input,
            ui.local_compile_input, ui.local_cpu_threads_input, ui.locale_probe_input,
        ],
        outputs=ui.local_runtime_status,
        js=sync_locale_js,
    )
    ui.save_access_key_button.click(
        fn=callback("save_access_key_and_status"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.locale_probe_input],
        outputs=[ui.access_key_status, ui.login_status_box, ui.onboarding_guide],
        js=sync_locale_js,
    )
    ui.onboarding_complete_button.click(
        fn=callback("dismiss_onboarding"),
        inputs=[ui.user_id_input, ui.access_key_input],
        outputs=ui.onboarding_guide,
        show_progress="hidden",
    )
    ui.change_access_key_button.click(
        fn=callback("change_access_key_and_status"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.new_access_key_input, ui.locale_probe_input],
        outputs=[ui.access_key_status, ui.access_key_input, ui.new_access_key_input, ui.login_status_box],
        js=sync_locale_js,
    )
    ui.admin_recovery_button.click(
        fn=callback("admin_recover_access_key_and_status"),
        inputs=[ui.user_id_input, ui.admin_recovery_key_input, ui.admin_new_access_key_input, ui.locale_probe_input],
        outputs=[ui.admin_recovery_status_box, ui.access_key_input, ui.admin_recovery_key_input, ui.login_status_box],
        js=sync_locale_js,
    )
    ui.export_user_data_button.click(
        fn=callback("export_user_data_from_gui"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.locale_probe_input],
        outputs=[ui.export_user_data_status, ui.export_user_data_file],
        js=sync_locale_js,
    )

    ui.refresh_logs_button.click(fn=callback("get_log_text"), outputs=ui.log_box)
    ui.clear_logs_button.click(fn=callback("clear_logs"), outputs=ui.log_box)
    ui.log_timer.tick(fn=callback("get_log_text"), outputs=ui.log_box)

    ui.save_mood_button.click(
        fn=callback("submit_mood_checkin_and_refresh_dashboard"),
        inputs=[
            ui.user_id_input, ui.access_key_input, ui.mood_date_input, ui.mood_choice_input,
            ui.mood_intensity_input, ui.mood_note_input, ui.weekly_mood_end_date_input,
            ui.theme_mode_input, ui.locale_probe_input,
        ],
        outputs=[
            ui.mood_box, ui.weekly_mood_end_date_input, ui.weekly_mood_chart,
            ui.weekly_mood_summary, ui.weekly_mood_analysis,
        ],
    )
    ui.refresh_mood_button.click(
        fn=callback("refresh_mood_panel"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.locale_probe_input],
        outputs=ui.mood_box,
    )
    ui.delete_mood_button.click(
        fn=callback("delete_mood_checkin_and_refresh_dashboard"),
        inputs=[
            ui.user_id_input, ui.access_key_input, ui.mood_date_input,
            ui.weekly_mood_end_date_input, ui.theme_mode_input, ui.locale_probe_input,
        ],
        outputs=[
            ui.mood_box, ui.weekly_mood_end_date_input, ui.weekly_mood_chart,
            ui.weekly_mood_summary, ui.weekly_mood_analysis,
        ],
    )
    ui.refresh_weekly_mood_button.click(
        fn=callback("refresh_weekly_mood_dashboard"),
        inputs=[
            ui.user_id_input, ui.access_key_input, ui.weekly_mood_end_date_input,
            ui.theme_mode_input, ui.locale_probe_input,
        ],
        outputs=[ui.weekly_mood_chart, ui.weekly_mood_summary, ui.weekly_mood_analysis],
    )
    ui.theme_mode_input.change(
        fn=callback("refresh_weekly_mood_dashboard"),
        inputs=[
            ui.user_id_input, ui.access_key_input, ui.weekly_mood_end_date_input,
            ui.theme_mode_input, ui.locale_probe_input,
        ],
        outputs=[ui.weekly_mood_chart, ui.weekly_mood_summary, ui.weekly_mood_analysis],
        js=refresh_chart_theme_js,
        show_progress="hidden",
    )
    ui.demo.load(
        fn=callback("sync_locale_value"),
        inputs=ui.locale_probe_input,
        outputs=ui.locale_probe_input,
        js=sync_locale_js,
        show_progress="hidden",
    )
    ui.demo.load(
        fn=callback("load_theme_and_weekly_dashboard"),
        inputs=[
            ui.user_id_input, ui.access_key_input, ui.weekly_mood_end_date_input,
            ui.theme_mode_input, ui.locale_probe_input,
        ],
        outputs=[ui.theme_mode_input, ui.weekly_mood_chart, ui.weekly_mood_summary, ui.weekly_mood_analysis],
        js=load_theme_js,
        show_progress="hidden",
    )

    ui.load_memory_button.click(
        fn=callback("load_memory_editor"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.memory_section, ui.locale_probe_input],
        outputs=[ui.memory_editor, ui.memory_box],
    )
    ui.save_memory_button.click(
        fn=callback("save_memory_editor"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.memory_section, ui.memory_editor, ui.locale_probe_input],
        outputs=[ui.memory_editor, ui.memory_box],
    )
    ui.clear_memory_button.click(
        fn=callback("clear_memory_section_and_reload"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.memory_section, ui.locale_probe_input],
        outputs=[ui.memory_editor, ui.memory_box],
    )
    ui.refresh_memory_events_button.click(
        fn=callback("load_memory_event_log"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.locale_probe_input],
        outputs=ui.memory_event_box,
    )
    ui.backup_memory_button.click(
        fn=callback("backup_memory_from_gui"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.locale_probe_input],
        outputs=[ui.memory_backup_status, ui.memory_backup_file],
    )
    ui.restore_memory_button.click(
        fn=callback("restore_memory_from_gui"),
        inputs=[
            ui.user_id_input, ui.access_key_input, ui.restore_memory_file,
            ui.restore_memory_mode, ui.locale_probe_input,
        ],
        outputs=[ui.memory_backup_status, ui.memory_safety_backup_file, ui.memory_box, ui.memory_event_box],
    )
    ui.add_stable_profile_button.click(
        fn=callback("add_stable_profile"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.stable_profile_input, ui.locale_probe_input],
        outputs=[ui.stable_profile_input, ui.stable_profile_editor, ui.stable_profile_box],
    )
    ui.load_stable_profile_button.click(
        fn=callback("load_stable_profile_editor"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.locale_probe_input],
        outputs=[ui.stable_profile_editor, ui.stable_profile_box],
    )
    ui.save_stable_profile_button.click(
        fn=callback("save_stable_profile_editor"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.stable_profile_editor, ui.locale_probe_input],
        outputs=[ui.stable_profile_editor, ui.stable_profile_box],
    )
    ui.clear_stable_profile_button.click(
        fn=callback("clear_stable_profile"),
        inputs=[ui.user_id_input, ui.access_key_input, ui.locale_probe_input],
        outputs=[ui.stable_profile_editor, ui.stable_profile_box],
    )

    ui.import_knowledge_button.click(
        fn=callback("import_knowledge_files_and_refresh"),
        inputs=ui.knowledge_files,
        outputs=[ui.knowledge_box, ui.knowledge_document_selector, ui.knowledge_document_box],
    )
    ui.rebuild_knowledge_button.click(fn=callback("rebuild_knowledge_panel"), outputs=ui.knowledge_box)
    ui.refresh_knowledge_button.click(fn=callback("knowledge_status"), outputs=ui.knowledge_box)
    ui.preview_knowledge_button.click(
        fn=callback("preview_knowledge_search"),
        inputs=[ui.knowledge_query, ui.knowledge_top_k, ui.knowledge_threshold, ui.knowledge_candidate_multiplier],
        outputs=ui.knowledge_box,
    )
    ui.inspect_knowledge_quality_button.click(fn=callback("format_knowledge_quality_report"), outputs=ui.knowledge_box)
    ui.refresh_knowledge_documents_button.click(
        fn=callback("refresh_knowledge_documents_from_gui"),
        outputs=[ui.knowledge_document_selector, ui.knowledge_document_box],
    )
    ui.delete_knowledge_document_button.click(
        fn=callback("delete_knowledge_document_from_gui"),
        inputs=ui.knowledge_document_selector,
        outputs=[ui.knowledge_document_selector, ui.knowledge_document_box, ui.knowledge_box],
    )
    ui.clear_knowledge_documents_button.click(
        fn=callback("clear_knowledge_documents_from_gui"),
        outputs=[ui.knowledge_document_selector, ui.knowledge_document_box, ui.knowledge_box],
    )
    ui.import_style_button.click(fn=callback("import_style_files"), inputs=ui.style_files, outputs=ui.style_box)
    ui.rebuild_style_button.click(fn=callback("rebuild_style_panel"), outputs=ui.style_box)
    ui.refresh_style_button.click(fn=callback("style_status"), outputs=ui.style_box)
    ui.preview_style_button.click(fn=callback("preview_style_search"), inputs=ui.style_query, outputs=ui.style_box)
