"""`make data`: download the pinned Wikipedia revisions in manifest.json into corpus.jsonl.

Text is CC BY-SA 4.0 (Wikipedia contributors); each row keeps a permalink to its source revision.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

from data.wiki import fetch_revision_text

HERE = Path(__file__).parent
MANIFEST = HERE / "manifest.json"
CORPUS = HERE / "corpus.jsonl"
MAX_DOC_CHARS = 6000  # lead plus the first sections of each article
MIN_DOC_CHARS = 1200


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    done: dict[str, dict[str, object]] = {}
    if CORPUS.exists():  # resume after an interrupted run
        for line in CORPUS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    short: list[str] = []
    with CORPUS.open("w", encoding="utf-8") as out:
        for n, entry in enumerate(manifest, start=1):
            row = done.get(entry["id"])
            if row is None or row["revid"] != entry["revid"]:
                text = fetch_revision_text(entry["lang"], entry["revid"], MAX_DOC_CHARS)
                title = urllib.parse.quote(entry["title"].replace(" ", "_"))
                url = (
                    f"https://{entry['lang']}.wikipedia.org/w/index.php"
                    f"?title={title}&oldid={entry['revid']}"
                )
                row = {**entry, "url": url, "text": text}
                print(f"[{n}/{len(manifest)}] {entry['id']} {entry['title']} ({len(text)} chars)")
            if len(str(row["text"])) < MIN_DOC_CHARS:
                short.append(entry["id"])
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    if short:
        raise SystemExit(f"too short, remove from manifest.json: {' '.join(short)}")
    print(f"corpus ready: {len(manifest)} docs in {CORPUS}")


if __name__ == "__main__":
    main()
