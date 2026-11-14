"""Тесты поиска/резолва картинок и кэша описаний (Фаза 5B, без API)."""

from rag.ingest.images import (
    CaptionCache,
    caption_image,
    caption_markdown,
    find_image_refs,
)
from rag.ingest.markdown import load_wiki


def _make_png(path):
    # содержимое не важно для резолва/хэша — важны расширение и наличие файла
    path.write_bytes(b"\x89PNG\r\n\x1a\n fake bytes")


def test_find_resolves_local_image_and_strips_size(tmp_path):
    files = tmp_path / ".files"
    files.mkdir()
    _make_png(files / "image.png")
    text = "см. схему ![Схема.png](.files/image.png =1049x538) ниже"
    refs = find_image_refs(text, tmp_path)
    assert len(refs) == 1
    assert refs[0].is_local
    assert refs[0].path == ".files/image.png"  # суффикс =ШхВ срезан
    assert refs[0].abs_path == (files / "image.png").resolve()


def test_external_and_missing_are_not_local(tmp_path):
    text = (
        "![favicon](https://static.figma.com/favicon.ico)\n"
        "![нет файла](.files/absent.png)\n"
    )
    refs = find_image_refs(text, tmp_path)
    assert len(refs) == 2
    assert all(not r.is_local for r in refs)


def test_non_image_extension_ignored(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF")
    refs = find_image_refs("![doc](doc.pdf)", tmp_path)
    assert not refs[0].is_local


def test_caption_cache_roundtrip(tmp_path):
    cache = CaptionCache(tmp_path / "captions")
    assert cache.get("k") is None
    cache.put("k", "описание")
    assert cache.get("k") == "описание"


def test_caption_image_uses_cache_and_calls_captioner_once(tmp_path):
    img = tmp_path / "img.png"
    _make_png(img)
    cache = CaptionCache(tmp_path / "captions")
    calls = {"n": 0}

    def captioner(_path):
        calls["n"] += 1
        return "  описание схемы  "

    first = caption_image(img, captioner, cache)
    second = caption_image(img, captioner, cache)
    assert first == "описание схемы"  # strip применён
    assert second == "описание схемы"
    assert calls["n"] == 1  # второй раз — из кэша, captioner не вызван


def test_caption_markdown_replaces_local_and_removes_external(tmp_path):
    files = tmp_path / ".files"
    files.mkdir()
    _make_png(files / "d.png")
    text = "до ![d](.files/d.png =10x10) и ![ext](https://x/y.png) после"
    out, n = caption_markdown(text, tmp_path, lambda _p: "диаграмма потока")
    assert n == 1
    assert "[Изображение: диаграмма потока]" in out
    assert "https://x/y.png" not in out  # внешняя убрана
    assert "до" in out and "после" in out


def test_caption_markdown_skips_image_when_captioner_fails(tmp_path):
    files = tmp_path / ".files"
    files.mkdir()
    _make_png(files / "d.png")

    def bad_captioner(_path):
        raise RuntimeError("image too large")

    out, n = caption_markdown("до ![d](.files/d.png =10x10) после", tmp_path, bad_captioner)
    assert n == 0  # картинка не описана
    assert "до" in out and "после" in out  # остальной текст цел
    assert "[Изображение" not in out  # битую картинку не вставили


def test_build_vision_captioner_requires_api_key(monkeypatch):
    import pytest

    from rag.ingest import images

    monkeypatch.setattr(images.settings, "anthropic_api_key", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        images.build_vision_captioner()


def test_load_wiki_injects_captions_into_text(tmp_path):
    files = tmp_path / ".files"
    files.mkdir()
    _make_png(files / "scheme.png")
    (tmp_path / "page.md").write_text(
        "# Страница\n\nСхема: ![s](.files/scheme.png =20x20)\n", encoding="utf-8"
    )
    docs = load_wiki(tmp_path, captioner=lambda _p: "описание схемы авторизации")
    assert len(docs) == 1
    assert "описание схемы авторизации" in docs[0].text
    # и в секции (structure-чанкинг берёт из sections)
    assert any("описание схемы авторизации" in s.text for s in docs[0].sections)
