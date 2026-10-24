# Retrieval test fixtures

These are sanitized real API responses used by `tests/test_retrieval.py`. The
tests do not need a network connection or an API key.

## Files

- `newsapi_tom_skerritt_1.json` - NewsAPI response for query
  `Tom Skerritt actor known Gun Alien died age 93`, captured on 2026-10-24.
  Kept 3 of the 4 returned articles.
- `newsapi_tom_skerritt_2.json` - NewsAPI response for query
  `Tom Skerritt the age of 93`, captured on 2026-10-24. Kept 3 of the 15
  returned articles.
- `newsapi_tom_skerritt_3.json` - NewsAPI response for query
  `Tom Skerritt the age of 93 died`, captured on 2026-10-24. Kept 3 of the
  12 returned articles.
- `gnews_tom_skerritt.json` - GNews response for query `Tom Skerritt`, captured
  on 2026-10-24. Kept 3 of the 10 returned articles. The response reported
  `totalArticles` as 128.

All four responses concern the claim: "Tom Skerritt, the actor known for Top Gun
and Alien, has died at the age of 93."

## Sanitization

- Kept only title, source name, URL, published date, and the short description
  and content text fields.
- Removed authors, images, article IDs, language metadata, source IDs, source
  URLs, source countries, and response metadata other than the article count.
- Removed request headers, request parameters, and every API key. These files
  contain response bodies only.
- Normalized dash variants to plain hyphens and line endings to `\n`.
- Kept the real truncation markers in the content fields. The NewsAPI fixtures
  use the `[+N chars]` form and the GNews fixture uses the `[N chars]` form.
  The provider tests verify that both forms are stripped.
- Some articles are syndicated copies with near-identical titles. The raw
  provider tests return all selected articles, while `retrieve()` deduplicates
  them. The fallback test exercises that path with the GNews fixture.

These fixtures are response snapshots, not a live fallback. Tests always
monkeypatch HTTP, so an offline test run never reaches NewsAPI or GNews.
