import fitz
import pytest

import TeleOCR.config as config
from TeleOCR import engine


def make_pdf(page_count=3):
    document = fitz.open()
    for index in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index}")
    payload = document.tobytes()
    document.close()
    return payload


@pytest.mark.parametrize("pdf_backend", ["PyMuPDF", "pypdfium2"])
def test_page_subset_preserves_requested_order(monkeypatch, pdf_backend):
    monkeypatch.setattr(config, "PDF_TOOLS", pdf_backend)
    prepared, mappings = engine._prepare_pdf_bytes([make_pdf()], [[2, 0]])
    subset = fitz.open(stream=prepared[0], filetype="pdf")
    try:
        assert len(subset) == 2
    finally:
        subset.close()
    assert mappings == [[2, 0]]

    result = {"pdf_info": [{"page_idx": 0}, {"page_idx": 1}]}
    engine.inplace_change_page_ids([result], mappings)
    assert [page["page_idx"] for page in result["pdf_info"]] == [2, 0]


@pytest.mark.parametrize("page_ids", [[], [0, 0], [True], ["0"]])
def test_invalid_page_selections_fail_before_inference(monkeypatch, page_ids):
    monkeypatch.setattr(config, "PDF_TOOLS", "PyMuPDF")
    with pytest.raises(ValueError):
        engine._prepare_pdf_bytes([make_pdf()], [page_ids])


def test_out_of_range_page_selection_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "PDF_TOOLS", "PyMuPDF")
    with pytest.raises(ValueError, match="out of range"):
        engine._prepare_pdf_bytes([make_pdf()], [[4]])


def test_staged_output_protects_existing_results(tmp_path):
    final = tmp_path / "document"
    final.mkdir()
    (final / "old.txt").write_text("old")

    with pytest.raises(FileExistsError):
        engine._stage_output(tmp_path, "document", overwrite=False)

    stage, target = engine._stage_output(tmp_path, "document", overwrite=True)
    (stage / "new.txt").write_text("new")
    engine._commit_output(stage, target)
    assert (final / "new.txt").read_text() == "new"
    assert not (final / "old.txt").exists()


def test_pipeline_failure_leaves_existing_output_intact(tmp_path, monkeypatch):
    final = tmp_path / "document"
    final.mkdir()
    (final / "old.txt").write_text("old")
    monkeypatch.setattr(engine, "doc_analyze", lambda *a, **k: {"pdf_info": []})

    def fail_output(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(engine, "_process_output", fail_output)
    with pytest.raises(OSError, match="disk full"):
        engine._process_vlm(tmp_path, ["document"], [b"pdf"], [None], overwrite=True)
    assert (final / "old.txt").read_text() == "old"
    assert not list(tmp_path.glob(".document-*"))


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".", ".."])
def test_output_name_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError):
        engine._stage_output(tmp_path, name, overwrite=False)
