# Data

Public data only: 313 Wikipedia articles about the Gulf region.

- 182 in Modern Standard Arabic (Saudi cities, governorates, airports, mountains, wadis,
  heritage sites, food), from ar.wikipedia.org
- 131 in English (UAE, Oman, Qatar, Bahrain, Kuwait places and landmarks, Arab cuisine), from
  en.wikipedia.org

`manifest.json` pins the exact revision of every article, so `make data` rebuilds the same
`corpus.jsonl` on every machine and the gold answers in `evals/questions.jsonl` stay valid.
Each article is cut to its first ~6,000 characters of body text.

There are no Gulf-dialect documents: public, openly licensed Gulf-dialect text is scarce. The
dialect lives in the questions instead (people ask in dialect about documents written in MSA),
which is the realistic case for a Gulf customer-facing RAG.

Wikipedia throttles anonymous clients, so `make data` takes roughly 30 minutes. Setting
`WIKI_USER_AGENT` to a string with your contact details, as Wikimedia asks, may speed it up.

## Licence

Article text is by Wikipedia contributors, licensed CC BY-SA 4.0. Every row of `corpus.jsonl`
has a `url` pointing at the source revision. The text is downloaded, not redistributed here.
`build_manifest.py` shows how the article list was chosen.
