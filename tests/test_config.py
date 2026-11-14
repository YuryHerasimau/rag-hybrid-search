"""Тесты конфигурации: разрешение путей к данным."""

from pathlib import Path

from rag.config import PROJECT_ROOT, settings


def test_raw_dir_relative_resolves_under_project_root(monkeypatch):
    monkeypatch.setattr(settings, "data_raw_dir", Path("data/raw"))
    assert settings.raw_dir == PROJECT_ROOT / "data/raw"


def test_raw_dir_absolute_is_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_raw_dir", tmp_path)
    assert settings.raw_dir == tmp_path


def test_images_dir_relative_resolves_under_project_root(monkeypatch):
    monkeypatch.setattr(settings, "data_images_dir", Path("data/images"))
    assert settings.images_dir == PROJECT_ROOT / "data/images"
