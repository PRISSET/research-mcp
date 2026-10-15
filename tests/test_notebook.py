"""RESEARCH.md: sections, papers without duplicates, conclusions, TOC, and hand edits left alone."""

import pytest

from kb import notebook as nb

PAPER = {"id": "arxiv:2302.01318", "title": "Accelerating [LLM] Decoding", "url": "https://arxiv.org/abs/2302.01318",
         "year": 2023, "venue": None, "grade": "B", "citation_count": 900, "authors": ["A", "B", "C", "D"]}


@pytest.fixture(autouse=True)
def notebook_file(tmp_path, monkeypatch):
    path = tmp_path / "RESEARCH.md"
    monkeypatch.setenv("RESEARCH_NOTEBOOK", str(path))
    monkeypatch.setenv("RESEARCH_LANG", "ru")
    return path


def test_start_add_write_and_toc(notebook_file):
    assert nb.start("Ускорение LLM", "Как ускорить декодирование?", ["speculative decoding"])["section"] == "ускорение-llm"
    assert nb.start("Ускорение LLM", "другой вопрос")["status"] == "exists"           # builds on, never duplicates
    assert nb.add_paper("ускорение-llm", PAPER, "- **Суть:** черновик")["status"] == "added"
    again = nb.add_paper("Ускорение LLM", PAPER, "- другое")                            # by title works too
    assert again["status"] == "exists" and "черновик" in again["current_notes"]
    assert nb.add_paper("ускорение-llm", PAPER, "- **Суть:** 2-2.5x", replace=True)["status"] == "replaced"
    nb.write("ускорение-llm", summary="Спекулятивное декодирование даёт 2-2.5x.", add_queries=["draft model"])
    text = notebook_file.read_text()
    assert text.count('research:paper id="arxiv:2302.01318"') == 1 and "черновик" not in text
    assert "[Accelerating (LLM) Decoding](https://arxiv.org/abs/2302.01318)" in text and "A, B, C et al." in text
    assert "| [Ускорение LLM](#ускорение-llm) | Как ускорить декодирование? | 1 |" in text
    assert "`speculative decoding`, `draft model`" in text
    sec = nb.read_section("ускорение-llm")
    assert sec["papers"] == ["arxiv:2302.01318"] and "2-2.5x" in sec["markdown"]
    assert nb.overview()["sections"][0]["has_summary"]
    assert nb.search("спекулятивное")[0]["section"] == "ускорение-llm"
    assert nb.sections_with("arxiv:2302.01318") == ["ускорение-llm"]


def test_placeholder_is_not_a_summary_and_append(notebook_file):
    nb.start("Тема", "Вопрос?")
    assert not nb.overview()["sections"][0]["has_summary"]
    nb.write("тема", summary="Первое.", append=True)
    nb.write("тема", summary="Второе.", append=True)
    assert "Первое.\n\nВторое." in notebook_file.read_text()


def test_hand_edits_outside_markers_survive(notebook_file):
    nb.start("Тема", "Вопрос?")
    notebook_file.write_text(notebook_file.read_text() + "\n## Мои заметки\nне трогать\n")
    nb.add_paper("тема", PAPER, "заметка")
    nb.write("тема", remove_papers=["arxiv:2302.01318"])
    text = notebook_file.read_text()
    assert "## Мои заметки\nне трогать" in text and "arxiv:2302.01318" not in text.split("<!-- /research:toc -->")[1]


def test_question_with_comment_end_does_not_break_the_file(notebook_file):
    nb.start("Тема", "Что если --> сломать?")
    assert nb.overview()["sections"][0]["question"] == "Что если --> сломать?"


def test_unknown_section_lists_existing():
    nb.start("Тема", "Вопрос?")
    with pytest.raises(nb.NotebookError, match='"тема"'):
        nb.add_paper("нет такой", PAPER, "x")
