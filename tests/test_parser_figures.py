from src.ingestion.parser import config
from src.ingestion.parser.figures import FigureRef, describe_figures, ocr_image


def fake_describer(image_bytes: bytes) -> str:
    return f"A test image, {len(image_bytes)} bytes."


def failing_describer(image_bytes: bytes) -> str:
    raise RuntimeError("vision API unavailable")


def test_descriptions_off_by_default_skips_figures(monkeypatch, caplog):
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", False)
    figs = [FigureRef(image_bytes=b"x" * 100, page=1, width=200, height=200)]
    with caplog.at_level("INFO"):
        pieces = describe_figures(figs, fake_describer)
    assert pieces == []
    assert "skipped" in caplog.text


def test_no_describer_given_skips_figures_even_if_enabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    figs = [FigureRef(image_bytes=b"x" * 100, page=1, width=200, height=200)]
    assert describe_figures(figs, None) == []


def test_enabled_with_describer_produces_pieces(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    figs = [FigureRef(image_bytes=b"x" * 50, page=2, width=200, height=200)]
    pieces = describe_figures(figs, fake_describer)
    assert len(pieces) == 1
    assert pieces[0].page == 2
    assert "test image" in pieces[0].text


def test_tiny_images_are_skipped(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    figs = [
        FigureRef(image_bytes=b"x" * 10, page=1, width=10, height=10),   # icon, too small
        FigureRef(image_bytes=b"x" * 50, page=1, width=200, height=200),
    ]
    pieces = describe_figures(figs, fake_describer)
    assert len(pieces) == 1


def test_caption_is_prepended_when_present(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    figs = [FigureRef(image_bytes=b"x" * 50, page=1, width=200, height=200,
                       caption="Figure 1: Revenue over time.")]
    (piece,) = describe_figures(figs, fake_describer)
    assert piece.text.startswith("Figure 1: Revenue over time.")


def test_one_failed_description_does_not_stop_the_rest(monkeypatch, caplog):
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    calls = {"n": 0}

    def flaky(image_bytes: bytes) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "ok description"

    figs = [
        FigureRef(image_bytes=b"x" * 50, page=1, width=200, height=200),
        FigureRef(image_bytes=b"y" * 50, page=2, width=200, height=200),
    ]
    with caplog.at_level("WARNING"):
        pieces = describe_figures(figs, flaky)
    assert len(pieces) == 1
    assert pieces[0].page == 2
    assert "could not be described" in caplog.text


def test_max_figures_per_doc_is_enforced(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FIGURE_DESCRIPTIONS", True)
    monkeypatch.setattr(config, "MAX_FIGURES_PER_DOC", 2)
    figs = [FigureRef(image_bytes=b"x" * 50, page=i, width=200, height=200) for i in range(5)]
    pieces = describe_figures(figs, fake_describer)
    assert len(pieces) == 2


def test_ocr_reads_real_text_from_an_image():
    from PIL import Image, ImageDraw, ImageFont
    import io as _io

    img = Image.new("RGB", (600, 100), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 30), "OpenAI released GPT-4", fill="black", font=font)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")

    text = ocr_image(buf.getvalue())
    # OCR isn't perfect (e.g. it commonly reads "I" as "l" in some fonts) — check the
    # part of the text that OCR engines reliably get right, not an exact match.
    assert "released" in text
    assert "GPT-4" in text or "GPT" in text
