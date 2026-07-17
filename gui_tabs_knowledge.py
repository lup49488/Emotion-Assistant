from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr


@dataclass
class KnowledgeTabComponents:
    knowledge_files: Any
    import_knowledge_button: Any
    refresh_knowledge_button: Any
    knowledge_query: Any
    knowledge_box: Any
    knowledge_advanced_controls: Any
    knowledge_top_k: Any
    knowledge_threshold: Any
    knowledge_candidate_multiplier: Any
    rebuild_knowledge_button: Any
    preview_knowledge_button: Any
    inspect_knowledge_quality_button: Any
    knowledge_document_selector: Any
    refresh_knowledge_documents_button: Any
    delete_knowledge_document_button: Any
    clear_knowledge_documents_button: Any
    knowledge_document_box: Any
    style_files: Any
    import_style_button: Any
    refresh_style_button: Any
    style_query: Any
    preview_style_button: Any
    style_box: Any
    style_advanced_controls: Any
    rebuild_style_button: Any


def build_knowledge_tab(
    tr: Callable[[str], Any],
    *,
    knowledge_top_k: int,
    knowledge_threshold: float,
    knowledge_candidate_multiplier: int,
    knowledge_status: Callable[[], str],
    style_status: Callable[[], str],
    refresh_document_panel: Callable[[], tuple[list[str], str]],
    format_document_list: Callable[[], str],
) -> KnowledgeTabComponents:
    """Render RAG and style-library controls; event wiring stays external."""
    with gr.Tab(tr("tab_knowledge"), id="knowledge"):
        with gr.Accordion(tr("知识库 / RAG"), open=True):
            knowledge_files_component = gr.File(
                label=tr("导入资料"), file_count="multiple",
                file_types=[".txt", ".md", ".markdown", ".csv", ".json", ".pdf", ".docx"],
            )
            with gr.Row():
                import_knowledge_button = gr.Button(tr("导入并重建索引"))
                refresh_knowledge_button = gr.Button(tr("查看知识库状态"))
            knowledge_query = gr.Textbox(
                label=tr("检索预览问题"),
                placeholder=tr("输入一个问题，查看会被检索进 Prompt 的资料片段"),
            )
            knowledge_box = gr.Textbox(
                label=tr("知识库状态 / 检索结果"), value=knowledge_status, lines=16, interactive=False
            )
            with gr.Column(visible=False) as knowledge_advanced_controls:
                with gr.Row():
                    knowledge_top_k_input = gr.Slider(
                        1, 10, value=knowledge_top_k, step=1, label=tr("返回片段数")
                    )
                    knowledge_threshold_input = gr.Slider(
                        0, 1, value=knowledge_threshold, step=0.05, label=tr("相关度阈值")
                    )
                    knowledge_candidate_multiplier_input = gr.Slider(
                        1, 8, value=knowledge_candidate_multiplier, step=1, label=tr("候选池倍数")
                    )
                with gr.Row():
                    rebuild_knowledge_button = gr.Button(tr("重建索引"))
                    preview_knowledge_button = gr.Button(tr("检索质量诊断"))
                    inspect_knowledge_quality_button = gr.Button(tr("检查知识库质量"))
                with gr.Accordion(tr("RAG 文档管理"), open=False):
                    knowledge_document_selector = gr.Dropdown(
                        label=tr("已入库文档"), choices=refresh_document_panel()[0],
                        value=None, interactive=True,
                    )
                    with gr.Row():
                        refresh_knowledge_documents_button = gr.Button(tr("刷新文档列表"))
                        delete_knowledge_document_button = gr.Button(tr("删除所选文档"), variant="stop")
                        clear_knowledge_documents_button = gr.Button(tr("清空全部文档"), variant="stop")
                    knowledge_document_box = gr.Textbox(
                        label=tr("文档清单"), value=format_document_list, lines=10, interactive=False
                    )

        with gr.Accordion(tr("风格库 / Style RAG"), open=False):
            style_files = gr.File(
                label=tr("导入风格样例"), file_count="multiple",
                file_types=[".txt", ".md", ".markdown", ".csv", ".json", ".jsonl"],
            )
            with gr.Row():
                import_style_button = gr.Button(tr("导入并重建风格索引"))
                refresh_style_button = gr.Button(tr("查看风格库状态"))
            style_query = gr.Textbox(
                label=tr("风格检索预览"),
                placeholder=tr("输入当前问题或场景，查看会被参考的回复风格样例"),
            )
            preview_style_button = gr.Button(tr("风格检索预览"))
            style_box = gr.Textbox(
                label=tr("风格库状态 / 检索结果"), value=style_status, lines=16, interactive=False
            )
            with gr.Column(visible=False) as style_advanced_controls:
                rebuild_style_button = gr.Button(tr("重建风格索引"))

    return KnowledgeTabComponents(
        knowledge_files_component, import_knowledge_button, refresh_knowledge_button,
        knowledge_query, knowledge_box, knowledge_advanced_controls, knowledge_top_k_input,
        knowledge_threshold_input, knowledge_candidate_multiplier_input, rebuild_knowledge_button,
        preview_knowledge_button, inspect_knowledge_quality_button, knowledge_document_selector,
        refresh_knowledge_documents_button, delete_knowledge_document_button,
        clear_knowledge_documents_button, knowledge_document_box, style_files,
        import_style_button, refresh_style_button, style_query, preview_style_button,
        style_box, style_advanced_controls, rebuild_style_button,
    )
