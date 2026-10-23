# Phase 2 - Claim analysis

## What we built

`app/schemas.py` now defines the shared `Claim`, `Article`, `Evidence` and `Verdict` dataclasses
that every later stage will use. `app/claim_analysis.py` loads spaCy's `en_core_web_sm` once,
extracts the allowed entity types, cleans them, and builds up to three deduplicated search
queries. `tests/test_claim_analysis.py` covers five sample claims, a no-entity claim, query
deduplication, and empty/short/non-string input. **11 tests pass** with
`python -m pytest tests/test_claim_analysis.py -q`.

## Worked output

Claim: `Narendra Modi met Joe Biden in New Delhi on Friday.`

- Entities: `Narendra Modi` (PERSON), `Joe Biden` (PERSON), `New Delhi` (GPE), `Friday` (DATE)
- Queries:
  1. `Narendra Modi met Joe Biden New Delhi Friday` — claim with stopwords removed, 8 words
  2. `Narendra Modi Joe Biden New Delhi Friday` — entities joined, people/places first
  3. `Narendra Modi Joe Biden met` — top entities plus the main verb

A claim with no named entity still works: `Inflation rose sharply.` gives
`Inflation rose sharply`, `rose Inflation`, `Inflation rose`.

## Concepts, simply

- **Named entity recognition (NER)** finds names and numeric facts in text. Analogy: highlighting
  the who, where, when and how much before searching a newspaper archive.
- **Entity labels**: `PERSON` = people, `ORG` = companies/institutions, `GPE` = countries,
  cities and states, `DATE` = dates, `MONEY` = amounts, `CARDINAL` = numbers.
- **Stopwords** are common words such as “the”, “is” and “in”. Removing them makes a search
  query shorter and more focused, but can make it ungrammatical.
- **Noun chunk** is a noun plus its modifiers, e.g. `the fourth quarter`. It is a useful search
  phrase when no clear verb exists.
- **Rule-based generation** means fixed rules, not a trained model. It is predictable and cheap;
  the cost is that it cannot paraphrase creatively.
- We cap at three queries because free news APIs have daily quotas and each query costs a call.

## How the code works

1. `_get_nlp()` loads `en_core_web_sm` lazily and raises a clear install message if it is missing.
2. `_normalize_claim()` trims whitespace and rejects empty, under-3-word, too-long or non-string input.
3. `_extract_entities()` keeps only `ENTITY_LABELS`, cleans trailing punctuation, removes duplicates.
4. `_ordered_entities()` sorts by usefulness: organization, person, place, event, then dates/numbers.
5. `_build_queries()` creates the three rule-based candidates, adds fallbacks, then
   `_add_query()` removes duplicates, drops queries under two words, caps each at 12 words and
   stops at three.
6. `analyze()` wires these together and returns a `Claim`.
7. `build_queries()` is the public single-purpose entry point used by tests.

## Why these choices

- The fixed small spaCy model is fast on CPU and accurate enough for headline entities.
- Rule-based queries are easy to defend in a viva and have no API cost.
- Excluding `LOC` follows the project contract, but it means `Eiffel Tower` is not extracted;
  `Paris` and `France` are. This is a real limitation, not a bug.
- Trade-offs: no coreference (“he” → a person), no semantic expansion (“economy” → “GDP”),
  and the small model can mislabel unusual names. The fallback queries reduce empty output.

## Likely viva questions

1. **Why use NER before retrieval?** Search APIs perform better with names and places than with
   a full sentence, and entities anchor the claim to the right articles.
2. **Why rule-based queries instead of an LLM?** No extra model cost, no hallucinated searches,
   and the rules are explainable.
3. **What happens if spaCy finds no entities?** The keyword and noun-chunk fallbacks still build
   queries; retrieval quality may be lower.
4. **Why remove stopwords only from the first query?** To keep it short and keyword-like; the
   entity queries already avoid most stopwords.
5. **What does `GPE` mean?** Geo-political entity: a country, city, state or similar place.

## Check yourself

1. Why does `Eiffel Tower` not appear in the entity list for the Paris claim?
2. What is the difference between a stopword and a named entity?
3. Why do we deduplicate queries before calling the news API?
