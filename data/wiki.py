"""Small Wikipedia helpers: a polite API client and an HTML -> plain-text extractor."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

# Wikimedia asks automated clients to identify themselves; put your contact in WIKI_USER_AGENT.
USER_AGENT = os.environ.get("WIKI_USER_AGENT", "wiki-corpus-fetch/0.1 (small educational demo)")
MIN_PARAGRAPH_CHARS = 40
_SKIP_TAGS = {"table", "style", "script", "sup", "figure", "ol", "ul"}
_SKIP_CLASSES = ("mw-editsection", "reference", "noprint", "hatnote", "navbox", "thumb", "geo")
_VOID_TAGS = {"br", "img", "hr", "input", "link", "meta", "wbr", "area", "col", "source"}


def api_get(lang: str, params: dict[str, str], pause_s: float = 1.0) -> dict[str, object]:
    """GET the MediaWiki API with a pause between calls and backoff on HTTP 429."""
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(
        f"https://{lang}.wikipedia.org/w/api.php?{query}", headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(6):
        time.sleep(pause_s)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body: dict[str, object] = json.load(response)
                return body
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(float(error.headers.get("Retry-After") or 5 * (attempt + 1)))
    raise RuntimeError(f"Wikipedia kept rate-limiting the request: {params}")


class _ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._stack: list[bool] = []  # one entry per open tag: does it start a skipped region?
        self._skip_depth = 0
        self._buffer: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            return
        classes = dict(attrs).get("class") or ""
        skip = tag in _SKIP_TAGS or any(name in classes for name in _SKIP_CLASSES)
        self._stack.append(skip)
        self._skip_depth += skip
        if tag == "p" and not self._skip_depth:
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or not self._stack:
            return
        self._skip_depth -= self._stack.pop()
        if tag == "p" and self._buffer is not None:
            text = re.sub(r"\[\d+\]", "", "".join(self._buffer))  # stray citation markers
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= MIN_PARAGRAPH_CHARS:
                self.paragraphs.append(text)
            self._buffer = None

    def handle_data(self, data: str) -> None:
        if self._buffer is not None and not self._skip_depth:
            self._buffer.append(data)


def html_to_text(html: str, max_chars: int) -> str:
    """Article paragraphs as plain text, capped at `max_chars` on a paragraph boundary."""
    parser = _ParagraphExtractor()
    parser.feed(html)
    kept: list[str] = []
    total = 0
    for paragraph in parser.paragraphs:
        if kept and total + len(paragraph) > max_chars:
            break
        kept.append(paragraph)
        total += len(paragraph) + 2
    return "\n\n".join(kept)


def fetch_revision_text(lang: str, revid: int, max_chars: int) -> str:
    body = api_get(lang, {"action": "parse", "oldid": str(revid), "prop": "text"})
    parse = body["parse"]
    assert isinstance(parse, dict)
    return html_to_text(str(parse["text"]), max_chars)
