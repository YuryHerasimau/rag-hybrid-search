"""Caption локальных картинок вики через Claude vision (Фаза 5B).

В вике есть PNG-схемы/скриншоты, часть информации — только на них. На этапе
инжеста прогоняем КАЖДУЮ локальную картинку через Claude vision → текстовое
описание, которое индексируется вместе с текстом страницы.

Формат ссылки Yandex Wiki: `![alt](sub/.files/name.png =ШИРИНАxВЫСОТА)` — путь
относителен каталогу .md-файла, суффикс размера срезаем. Внешние `http(s)://`
и `data:` — не картинки вики (favicon, аватары), пропускаем.

Описания кэшируются по хэшу содержимого файла: повторный инжест не платит за
vision заново. Модуль не импортирует anthropic на уровне модуля (фазовость).
"""

from __future__ import annotations

import base64
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from rag.config import PROJECT_ROOT, settings

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_CACHE_DIR = PROJECT_ROOT / "storage" / "captions"

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

CAPTION_PROMPT = (
    "Это схема, макет или скриншот из технической вики продукта. Опиши по делу, "
    "что на ней: элементы интерфейса/блоки, связи и потоки, подписи и значения. "
    "Описание будет проиндексировано для поиска — пиши на русском, без воды."
)


@dataclass
class ImageRef:
    """Ссылка на картинку из markdown."""

    alt: str
    path: str  # как в markdown, без суффикса размера
    abs_path: Path | None  # разрешённый локальный файл или None (внешний/отсутствует)

    @property
    def is_local(self) -> bool:
        return self.abs_path is not None


def _strip_size(raw: str) -> str:
    """'sub/.files/image.png =1049x538' -> 'sub/.files/image.png'."""
    raw = raw.strip()
    return raw.split()[0] if raw else ""


def find_image_refs(text: str, base_dir: Path) -> list[ImageRef]:
    """Находит `![]()` в тексте и резолвит локальные пути относительно base_dir.

    Внешние ссылки и отсутствующие файлы дают abs_path=None (is_local=False).
    """
    refs: list[ImageRef] = []
    for m in _IMG_RE.finditer(text):
        alt, path = m.group(1), _strip_size(m.group(2))
        abs_path: Path | None = None
        if path and not path.startswith(("http://", "https://", "data:")):
            cand = (base_dir / path).resolve()
            if cand.is_file() and cand.suffix.lower() in _MEDIA_TYPES:
                abs_path = cand
        refs.append(ImageRef(alt=alt, path=path, abs_path=abs_path))
    return refs


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class CaptionCache:
    """Кэш описаний на диске: ключ — хэш содержимого картинки, значение — текст."""

    def __init__(self, directory: Path = _CACHE_DIR) -> None:
        self.dir = directory

    def get(self, key: str) -> str | None:
        f = self.dir / f"{key}.txt"
        return f.read_text(encoding="utf-8") if f.exists() else None

    def put(self, key: str, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{key}.txt").write_text(text, encoding="utf-8")


def caption_image(abs_path: Path, captioner, cache: CaptionCache | None = None) -> str:
    """Описание картинки. При наличии кэша — не вызывает captioner повторно.

    captioner — callable(Path) -> str (реальный использует Claude vision;
    в тестах подменяется заглушкой).
    """
    key = _content_hash(abs_path)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    text = captioner(abs_path).strip()
    if cache is not None:
        cache.put(key, text)
    return text


def caption_markdown(
    text: str, base_dir: Path, captioner, cache: CaptionCache | None = None
) -> tuple[str, int]:
    """Заменяет локальные `![]()` на текстовые описания; внешние/битые — убирает.

    Возвращает (новый_текст, число_описанных_картинок). Применяется к СЫРОМУ
    markdown до разбора на секции — тогда описание попадает и в structure-, и в
    simple-чанки.
    """
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        path = _strip_size(m.group(2))
        if path and not path.startswith(("http://", "https://", "data:")):
            cand = (base_dir / path).resolve()
            if cand.is_file() and cand.suffix.lower() in _MEDIA_TYPES:
                try:
                    caption = caption_image(cand, captioner, cache)
                except Exception as exc:  # noqa: BLE001 — любой сбой по одной
                    # картинке (vision-API, размер, битый файл) не должен ронять
                    # инжест всего корпуса; пропускаем и идём дальше
                    print(f"[captions] пропуск {cand.name}: {exc}", file=sys.stderr)
                    return ""
                count += 1
                return f"[Изображение: {caption}]"
        return ""  # внешняя ссылка или нет файла — убираем шум

    return _IMG_RE.sub(repl, text), count


def build_vision_captioner(model: str | None = None):
    """Фабрика реального captioner'а на Claude vision (Haiku по умолчанию)."""
    import anthropic

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "Не задан ANTHROPIC_API_KEY — он нужен для --captions (описание картинок). "
            "Впиши ключ в .env или собирай индекс без --captions."
        )
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model = model or settings.cheap_model

    def captioner(path: Path) -> str:
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        message = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        },
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }
            ],
        )
        return message.content[0].text

    return captioner
