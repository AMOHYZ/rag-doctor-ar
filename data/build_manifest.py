"""One-off: pick articles from Wikipedia categories and pin their revisions in manifest.json.

`make data` does NOT run this. It only reads the committed manifest, so every clone gets the
same text and the gold answers in evals/questions.jsonl stay valid. Re-run by hand to rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.wiki import api_get

MANIFEST = Path(__file__).parent / "manifest.json"
PER_CATEGORY = 28
MIN_BYTES = {"ar": 9000, "en": 6000}  # wikitext size; Arabic is ~2 bytes per character
CATEGORIES = {
    "ar": [  # Saudi Arabia, in Modern Standard Arabic
        "تصنيف:مدن السعودية",
        "تصنيف:محافظات السعودية",
        "تصنيف:مطارات السعودية",
        "تصنيف:مطبخ سعودي",
        "تصنيف:جبال السعودية",
        "تصنيف:جامعات السعودية",
        "تصنيف:مواقع أثرية في السعودية",
        "تصنيف:أندية كرة قدم سعودية",
        "تصنيف:مواقع التراث العالمي في السعودية",
        "تصنيف:جزر السعودية",
        "تصنيف:أودية السعودية",
    ],
    "en": [  # the other Gulf states, in English
        "Category:Cities in the United Arab Emirates",
        "Category:Tourist attractions in Dubai",
        "Category:Skyscrapers in Dubai",
        "Category:Tourist attractions in Abu Dhabi",
        "Category:Populated places in Qatar",
        "Category:Museums in Qatar",
        "Category:Forts in Oman",
        "Category:Populated places in Oman",
        "Category:Islands of Bahrain",
        "Category:Tourist attractions in Bahrain",
        "Category:Buildings and structures in Kuwait City",
        "Category:Islands of Kuwait",
        "Category:Arab cuisine",
    ],
}


def category_pages(lang: str, category: str) -> list[dict[str, object]]:
    body = api_get(
        lang,
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmnamespace": "0",
            "gcmlimit": "500",
            "prop": "info",
        },
    )
    query = body.get("query")
    pages = query.get("pages", []) if isinstance(query, dict) else []
    big = [page for page in pages if int(page.get("length", 0)) >= MIN_BYTES[lang]]
    big.sort(key=lambda page: -int(page["length"]))
    return big[:PER_CATEGORY]


def main() -> None:
    entries: dict[str, dict[str, object]] = {}
    for lang, categories in CATEGORIES.items():
        for category in categories:
            pages = category_pages(lang, category)
            print(f"{lang} {category}: {len(pages)} pages")
            for page in pages:
                doc_id = f"{lang}-{page['pageid']}"
                entries[doc_id] = {
                    "id": doc_id,
                    "lang": lang,
                    "title": page["title"],
                    "revid": page["lastrevid"],
                }
    ordered = sorted(entries.values(), key=lambda entry: str(entry["id"]))
    MANIFEST.write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(ordered)} entries to {MANIFEST}")


if __name__ == "__main__":
    main()
