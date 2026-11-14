"""Smoke test: embed three tiny docs, retrieve, and answer one question with the local LLM."""

from __future__ import annotations

from rag.common import Chunk, OllamaEmbedder, VectorStore, build_prompt, make_llm

CHUNKS = [
    Chunk("en-1", "en", "Riyadh is the capital and largest city of Saudi Arabia."),
    Chunk("ar-1", "ar", "جدة مدينة سعودية تقع على ساحل البحر الأحمر، وهي بوابة الحرمين الشريفين."),
    Chunk("en-2", "en", "Kabsa is a rice and meat dish popular across the Arabian Peninsula."),
]
QUESTION = "على أي بحر تقع مدينة جدة؟"


def main() -> None:
    embedder = OllamaEmbedder()
    store = VectorStore(CHUNKS, embedder.embed([chunk.text for chunk in CHUNKS]))
    hits = store.search(embedder.embed([QUESTION])[0], k=2)
    completion = make_llm().complete(build_prompt(QUESTION, hits))
    print(f"question : {QUESTION}")
    print(f"top hit  : {hits[0].chunk.doc_id} (score {hits[0].score:.2f})")
    print(f"answer   : {completion.text}")
    if hits[0].chunk.doc_id != "ar-1" or not completion.text:
        raise SystemExit("smoke test failed")
    print("smoke test OK")


if __name__ == "__main__":
    main()
