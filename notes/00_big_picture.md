# The big picture

**Goal:** a user types a news claim; the system returns `SUPPORTED`, `REFUTED` or
`UNVERIFIED`, a confidence score, and the ranked evidence behind the decision.

A verdict is **not** a truth oracle. It means: "given the articles we could retrieve just now,
and what a pretrained NLI model thinks of their snippets, this is the weighted balance of
support and refutation." If there is too little evidence, the honest answer is `UNVERIFIED`.

## Pipeline

```
claim
  -> 1. claim analysis: entities + search queries
  -> 2. retrieval: live news articles (cache, provider fallback)
  -> 3. ranking: best evidence sentence per article + recency
  -> 4. NLI: does each snippet support / refute / stay neutral?
  -> 5. credibility: weight each source
  -> 6. decision: combine weighted probabilities into a verdict
  -> JSON API + one HTML page
```

Status after Phase 1: only the **baseline** and the **NLI smoke test** exist. Stages 1-6 are
planned, not yet built.

## Worked example

Claim: **"The Eiffel Tower is located in Paris."**

1. **Claim analysis (Phase 2).** spaCy finds the entity `Eiffel Tower` (a landmark, likely
   `ORG`/`FAC`) and `Paris` (`GPE`). Rule-based query generation produces:
   `Eiffel Tower Paris`, `Eiffel Tower location`, `Paris Eiffel Tower`.
2. **Retrieval (Phase 3).** NewsAPI is asked for each query. Articles are normalized, merged,
   and de-duplicated by URL and near-identical title. If NewsAPI fails, GNews is tried. The
   result is cached in `.cache/` so a demo can run offline.
3. **Ranking (Phase 4).** Each article body is split into sentences and embedded with MiniLM.
   The sentence most similar to the claim becomes the snippet. A very recent article keeps
   full score; an old one is discounted. The top six snippets above the similarity floor survive.
4. **NLI (Phase 5).** For each snippet, the model reads premise = snippet and hypothesis =
   claim. A Paris tourism article gives high `p_entail`; an article saying the tower is in
   Berlin would give high `p_contra`; an article about ticket prices gives high `p_neutral`.
5. **Credibility (Phase 6).** `reuters.com` or `bbc.com` gets a high weight; an unknown blog
   gets a low weight; a blocklisted satire site gets the lowest weight. Each evidence weight is
   `similarity x credibility`.
6. **Decision (Phase 6).** Weighted support and refutation are summed. If there are fewer than
   two useful sources, total weight is too low, or neither side clearly wins, the verdict is
   `UNVERIFIED`. Otherwise the larger side wins with a confidence based on the margin.
7. **API and page (Phase 7).** FastAPI returns the full `Verdict` JSON; `index.html` shows the
   badge, confidence bar, explanation and evidence cards.

For this claim, the expected result is `SUPPORTED` with high confidence if reputable outlets
are retrieved. If retrieval returns only unrelated articles, the expected result is
`UNVERIFIED`, not a guess.

## ISOT baseline vs the real pipeline

The Phase 1 baseline classifies an article as Fake or True from its writing style. That is a
different task from verifying a claim against live evidence. Its 98.0% deduped score is useful
as a sanity check and as a warning about shortcuts such as the Reuters dateline, but it is
**not** the accuracy of VeriNews.
