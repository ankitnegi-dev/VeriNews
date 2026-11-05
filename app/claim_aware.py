"""Claim-aware helpers for the v3 experiment.

The module is deliberately independent of the pipeline. It provides:

* a number-free first retrieval query (component A);
* claim features (numbers, entities, ordinals, content terms, clauses);
* 1- and 2-sentence passage candidates and a coverage-aware ranker (B);
* the two evidence guards and their change counters (C).

No function in this module fetches a URL, loads a model, or changes a verdict by
itself. The pipeline decides when to use the results.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

import config
from app.claim_analysis import _extract_entities, _get_nlp, _ordered_entities
from app.schemas import NLIResult as SchemaNLIResult

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
WHITESPACE = re.compile(r"\s+")
TOKEN = re.compile(r"[a-z0-9]+")

ORDINAL_WORDS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
        "thirtieth",
        "fortieth",
        "fiftieth",
        "sixtieth",
        "seventieth",
        "eightieth",
        "ninetieth",
        "hundredth",
    }
)
ORDINAL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|"
    r"seventieth|eightieth|ninetieth|hundredth|"
    r"\d{1,2}(?:st|nd|rd|th))\b",
    re.IGNORECASE,
)

NAMED_ENTITY_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "LAW", "EVENT", "NORP"}
)
NUMERIC_ENTITY_LABELS = frozenset(
    {"DATE", "MONEY", "PERCENT", "CARDINAL", "QUANTITY", "ORDINAL"}
)
AMOUNT_UNITS = frozenset(
    {
        "billion",
        "million",
        "trillion",
        "thousand",
        "percent",
        "cent",
        "cents",
        "dollar",
        "dollars",
        "euro",
        "euros",
        "rupee",
        "rupees",
        "pound",
        "pounds",
    }
)
UNIT_WORDS = frozenset(
    {
        "billion",
        "million",
        "trillion",
        "thousand",
        "percent",
        "cent",
        "cents",
        "basis",
        "point",
        "points",
        "bps",
        "people",
        "workers",
        "employees",
    }
)

SCORE_RE = re.compile(r"(?<![\w.])(\d{1,3})\s*[-–]\s*(\d{1,3})(?![\w.])")
NUMBER_RE = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(billion|million|trillion|thousand|percent|per\s+cent|%|cents?|"
    r"basis\s+points?|bps|points?|people|workers|employees)?",
    re.IGNORECASE,
)
DATE_NUMBER_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}(?:,?\s+\d{4})?\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class ClaimFeatures:
    """The claim facts used by retrieval, ranking, and the guards."""

    text: str
    numbers: tuple[str, ...]
    entities: tuple[str, ...]
    main_entity: str | None
    ordinals: tuple[str, ...]
    content_terms: tuple[str, ...]
    clauses: tuple[str, ...]
    coordinated: bool


@dataclass
class GuardStats:
    """Counts of passages whose stance a guard changed."""

    guard1_changed: int = 0
    guard1_refute_to_neutral: int = 0
    guard1_kept: int = 0
    guard2_changed: int = 0
    guard2_neutral_to_support: int = 0
    guard2_weak_refute_to_support: int = 0
    guard2_kept: int = 0
    passages_seen: int = 0

    def merge(self, other: "GuardStats") -> None:
        self.guard1_changed += other.guard1_changed
        self.guard1_refute_to_neutral += other.guard1_refute_to_neutral
        self.guard1_kept += other.guard1_kept
        self.guard2_changed += other.guard2_changed
        self.guard2_neutral_to_support += other.guard2_neutral_to_support
        self.guard2_weak_refute_to_support += other.guard2_weak_refute_to_support
        self.guard2_kept += other.guard2_kept
        self.passages_seen += other.passages_seen

    def as_dict(self) -> dict[str, int]:
        return {
            "guard1_changed": self.guard1_changed,
            "guard1_refute_to_neutral": self.guard1_refute_to_neutral,
            "guard1_kept": self.guard1_kept,
            "guard2_changed": self.guard2_changed,
            "guard2_neutral_to_support": self.guard2_neutral_to_support,
            "guard2_weak_refute_to_support": self.guard2_weak_refute_to_support,
            "guard2_kept": self.guard2_kept,
            "passages_seen": self.passages_seen,
        }


def _clean(text: str) -> str:
    return WHITESPACE.sub(" ", str(text)).strip()


def _is_ordinal_text(text: str) -> bool:
    return bool(ORDINAL_RE.fullmatch(text.strip())) or text.strip().lower() in ORDINAL_WORDS


def _is_year(value: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", value))


def _normalize_number(value: str, unit: str | None) -> str:
    value = value.replace(",", "").strip()
    normalized_unit = _clean(unit or "").lower()
    normalized_unit = re.sub(r"\s+", " ", normalized_unit)
    if normalized_unit in {"percent", "per cent", "%"}:
        return f"{value}%"
    if normalized_unit in {"basis point", "basis points", "bps"}:
        return f"{value} bps"
    if normalized_unit in {"cent", "cents"}:
        return f"{value} cents"
    if normalized_unit in {"billion", "million", "trillion", "thousand"}:
        return f"{value} {normalized_unit}"
    if normalized_unit in {"point", "points"}:
        return f"{value} points"
    if normalized_unit in {"people", "workers", "employees"}:
        return f"{value} {normalized_unit}"
    return value


def extract_numbers(text: str) -> tuple[str, ...]:
    """Extract substantive claim numbers, excluding calendar years and days.

    ``12.87 billion``, ``5.50%`` and ``87 cents`` are kept.  ``2026`` and the
    day in ``October 9, 2026`` are not: they are calendar coordinates, not the
    quantitative facts the guards are designed to rescue.
    """
    cleaned = _clean(text)
    if not cleaned:
        return ()

    date_spans = [match.span() for match in DATE_NUMBER_RE.finditer(cleaned)]
    year_spans = [match.span() for match in YEAR_RE.finditer(cleaned)]

    def _inside(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
        return any(start <= span[0] and span[1] <= end for start, end in spans)

    found: list[str] = []
    seen: set[str] = set()

    for match in SCORE_RE.finditer(cleaned):
        if _inside(match.span(), date_spans):
            continue
        value = f"{match.group(1)}-{match.group(2)}"
        if value not in seen:
            seen.add(value)
            found.append(value)

    for match in NUMBER_RE.finditer(cleaned):
        if _inside(match.span(), date_spans) or _inside(match.span(), year_spans):
            continue
        raw_value = match.group(1)
        unit = match.group(2)
        if unit is None and _is_year(raw_value):
            continue
        # A score such as 2-2 was already captured above; skip its components.
        if any(
            score.startswith(f"{raw_value}-") or score.endswith(f"-{raw_value}")
            for score in found
        ):
            continue
        value = _normalize_number(raw_value, unit)
        if value not in seen:
            seen.add(value)
            found.append(value)

    return tuple(found)


def extract_ordinals(text: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for match in ORDINAL_RE.finditer(_clean(text)):
        value = match.group(0).lower()
        if value not in seen:
            seen.add(value)
            found.append(value)
    return tuple(found)


def _named_entities(doc) -> tuple[str, ...]:
    entities: list[str] = []
    seen: set[str] = set()
    for entity in _ordered_entities(_extract_entities(doc)):
        if entity["label"] not in NAMED_ENTITY_LABELS:
            continue
        key = entity["text"].lower()
        if key not in seen:
            seen.add(key)
            entities.append(entity["text"])
    return tuple(entities)


def _content_terms(doc) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space or token.like_num:
            continue
        if token.pos_ not in {"NOUN", "PROPN", "VERB", "ADJ"}:
            continue
        lemma = token.lemma_.lower().strip()
        if len(lemma) < 2 or not any(char.isalpha() for char in lemma):
            continue
        if lemma not in seen:
            seen.add(lemma)
            terms.append(lemma)
    return tuple(terms)


def _split_clauses(text: str) -> tuple[tuple[str, ...], bool]:
    """Split a coordinated claim into clauses.

    A split happens only when spaCy sees a conjunction between two verbs, or
    when the literal text contains ``, and``.  Noun-phrase coordination such as
    ``Bayern Munich and Arsenal`` is deliberately left alone.
    """
    cleaned = _clean(text)
    if not cleaned:
        return (), False

    doc = _get_nlp()(cleaned)
    verb_conj = any(
        token.dep_ == "conj"
        and token.pos_ == "VERB"
        and token.head.pos_ == "VERB"
        for token in doc
    )
    comma_and = bool(re.search(r",\s*and\s+", cleaned, flags=re.IGNORECASE))

    if not verb_conj and not comma_and:
        return (cleaned,), False

    if comma_and:
        parts = re.split(r",\s*and\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)
    else:
        # Split at the conjunction token. The right clause starts after "and".
        and_tokens = [
            token
            for token in doc
            if token.text.lower() == "and" and token.dep_ == "cc"
        ]
        if not and_tokens:
            return (cleaned,), False
        split_at = and_tokens[0].idx
        parts = [cleaned[:split_at], cleaned[split_at:]]

    clauses = [_clean(part).strip(" ,;") for part in parts]
    clauses = [clause for clause in clauses if clause]
    if len(clauses) < 2 or any(len(clause.split()) < 2 for clause in clauses):
        return (cleaned,), False
    return tuple(clauses), True


def claim_features(text: str) -> ClaimFeatures:
    """Return the claim facts used by all three components."""
    cleaned = _clean(text)
    doc = _get_nlp()(cleaned)
    entities = _named_entities(doc)
    clauses, coordinated = _split_clauses(cleaned)
    return ClaimFeatures(
        text=cleaned,
        numbers=extract_numbers(cleaned),
        entities=entities,
        main_entity=entities[0] if entities else None,
        ordinals=extract_ordinals(cleaned),
        content_terms=_content_terms(doc),
        clauses=clauses,
        coordinated=coordinated,
    )


def _protected_token_indices(doc) -> set[int]:
    protected: set[int] = set()
    for entity in doc.ents:
        if entity.label_ not in config.ENTITY_LABELS:
            continue
        if entity.label_ in NUMERIC_ENTITY_LABELS:
            continue
        protected.update(token.i for token in entity)
    for chunk in doc.noun_chunks:
        chunk_tokens = [token for token in chunk if not token.is_punct]
        if any(token.pos_ == "PROPN" for token in chunk_tokens):
            protected.update(token.i for token in chunk_tokens)
    return protected


def number_free_query(text: str) -> str:
    """Build a query from entities and event words, with numbers removed.

    Ordinals such as ``second`` and place names are retained.  Amount units
    (``billion``, ``percent``, ``cents``) are removed with their numeral.
    """
    cleaned = _clean(text)
    if not cleaned:
        return ""

    doc = _get_nlp()(cleaned)
    protected = _protected_token_indices(doc)
    tokens: list[str] = []
    previous_numeric = False

    for token in doc:
        if token.is_punct or token.is_space:
            previous_numeric = False
            continue

        is_ordinal = _is_ordinal_text(token.text)
        if token.like_num and not is_ordinal:
            previous_numeric = True
            continue

        lower = token.text.lower()
        if lower in {"$", "€", "£", "¥", "%"}:
            previous_numeric = False
            continue
        if previous_numeric and lower in AMOUNT_UNITS:
            previous_numeric = False
            continue
        previous_numeric = False

        if token.is_stop and token.i not in protected:
            continue
        if token.pos_ not in {"NOUN", "PROPN", "VERB", "ADJ", "NUM"} and token.i not in protected:
            continue
        if token.pos_ == "NUM" and not is_ordinal:
            continue
        tokens.append(token.text)

    if len(tokens) < 2:
        # Fallback: remove digit-bearing tokens from the original text.
        fallback = [
            token.text
            for token in doc
            if not token.like_num and not token.is_punct and not token.is_space
        ]
        tokens = fallback

    query = " ".join(tokens[: config.MAX_QUERY_WORDS]).strip()
    return query


def number_free_first_query(claim: str, queries: Sequence[str]) -> list[str]:
    """Replace the first v1 query with the number-free query."""
    existing = [str(query).strip() for query in queries if str(query).strip()]
    replacement = number_free_query(claim)
    if not replacement:
        return existing

    result = [replacement]
    seen = {replacement.lower()}
    for query in existing:
        if query.lower() in seen:
            continue
        seen.add(query.lower())
        result.append(query)
        if len(result) >= config.MAX_QUERIES:
            break
    return result


def entity_aliases(entity: str) -> tuple[str, ...]:
    """Return surface forms used to check whether a passage names an entity."""
    cleaned = _clean(entity).strip(" .,;:!?\"'()[]{}")
    if not cleaned:
        return ()
    aliases = [cleaned]
    without_article = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    if without_article and without_article != cleaned:
        aliases.append(without_article)

    words = re.findall(r"[A-Za-z][A-Za-z0-9&.-]*", cleaned)
    if len(words) >= 3 and all(word[:1].isupper() for word in words):
        acronym = "".join(word[0] for word in words).upper()
        if len(acronym) >= 3:
            aliases.append(acronym)
    return tuple(dict.fromkeys(aliases))


def contains_entity(text: str, entity: str | None) -> bool:
    if not entity:
        return False
    for alias in entity_aliases(entity):
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
        if pattern.search(text):
            return True
    return False


def _passage_tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN.findall(text.lower()))


def _term_present(term: str, tokens: Sequence[str]) -> bool:
    if term in tokens:
        return True
    if len(term) >= 4:
        return any(
            len(token) >= 4 and (term in token or token in term)
            for token in tokens
        )
    return False


def content_term_coverage(features: ClaimFeatures, passage: str) -> float:
    terms = features.content_terms
    if not terms:
        return 1.0
    tokens = _passage_tokens(passage)
    matched = sum(1 for term in terms if _term_present(term, tokens))
    return matched / len(terms)


def numbers_present(features: ClaimFeatures, passage: str) -> bool:
    if not features.numbers:
        return False
    passage_numbers = set(extract_numbers(passage))
    return all(number in passage_numbers for number in features.numbers)


def guard_entity_and_terms(
    features: ClaimFeatures,
    passage: str,
    *,
    min_term_coverage: float,
) -> bool:
    """Return True when a contradiction passes guard (i)."""
    if features.main_entity and not contains_entity(passage, features.main_entity):
        return False
    return content_term_coverage(features, passage) >= min_term_coverage


def apply_guards(
    features: ClaimFeatures,
    passage: str,
    result: SchemaNLIResult,
    *,
    min_term_coverage: float,
    contradiction_max: float,
    support_prob: float,
    stats: GuardStats | None = None,
) -> SchemaNLIResult:
    """Apply guards (i) and (ii) to one passage result.

    Guard (ii) runs first, but only on passages NLI already called neutral, so
    it cannot silently undo guard (i)'s contradiction filter.  Guard (i) then
    removes contradictions that do not share the claim's main entity and enough
    of its content terms.
    """
    active = stats or GuardStats()
    active.passages_seen += 1
    stance = result.stance
    p_entail = result.p_entail
    p_neutral = result.p_neutral
    p_contra = result.p_contra

    if (
        stance == "NEUTRAL"
        and numbers_present(features, passage)
        and contains_entity(passage, features.main_entity)
        and p_contra < contradiction_max
    ):
        stance = "SUPPORTS"
        p_entail = max(p_entail, support_prob)
        p_neutral = max(0.0, 1.0 - p_entail - p_contra)
        active.guard2_changed += 1
        active.guard2_neutral_to_support += 1

    if stance == "REFUTES":
        if guard_entity_and_terms(
            features, passage, min_term_coverage=min_term_coverage
        ):
            active.guard1_kept += 1
        else:
            stance = "NEUTRAL"
            p_contra = 0.0
            p_neutral = max(p_neutral, 1.0 - p_entail)
            active.guard1_changed += 1
            active.guard1_refute_to_neutral += 1

    return SchemaNLIResult(
        p_entail=p_entail,
        p_neutral=p_neutral,
        p_contra=p_contra,
        stance=stance,
        cached=result.cached,
    )


def split_sentences(text: str) -> list[str]:
    cleaned = _clean(text)
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def sliding_windows(sentences: Sequence[str], size: int) -> list[str]:
    cleaned = [_clean(sentence) for sentence in sentences if _clean(sentence)]
    if not cleaned:
        return []
    size = max(1, int(size))
    if len(cleaned) <= size:
        return [" ".join(cleaned)]
    return [
        " ".join(cleaned[index : index + size])
        for index in range(len(cleaned) - size + 1)
    ]


def build_passages(
    text: str,
    *,
    title: str | None = None,
    sizes: Sequence[int] | None = None,
) -> list[str]:
    """Build title, 1-sentence, and 2-sentence passage candidates."""
    active_sizes = tuple(sizes or config.V3_PASSAGE_WINDOW_SIZES)
    passages: list[str] = []
    title_text = _clean(title or "")
    if title_text:
        passages.append(title_text)

    sentences = split_sentences(text)
    for size in active_sizes:
        for window in sliding_windows(sentences, size):
            cleaned = _clean(window)
            if cleaned and cleaned not in passages:
                passages.append(cleaned)

    if not passages and _clean(text):
        passages.append(_clean(text))
    return passages


def _normalize_rows(values: object) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _coverage_score(features: ClaimFeatures, passage: str) -> float:
    components: list[tuple[float, float]] = []
    if features.numbers:
        passage_numbers = set(extract_numbers(passage))
        components.append(
            (
                0.5,
                sum(1 for number in features.numbers if number in passage_numbers)
                / len(features.numbers),
            )
        )
    if features.entities:
        components.append(
            (
                0.3,
                sum(1 for entity in features.entities if contains_entity(passage, entity))
                / len(features.entities),
            )
        )
    if features.ordinals:
        passage_tokens = set(_passage_tokens(passage))
        components.append(
            (
                0.2,
                sum(1 for ordinal in features.ordinals if ordinal in passage_tokens)
                / len(features.ordinals),
            )
        )
    if not components:
        return 0.0
    total_weight = sum(weight for weight, _ in components)
    return sum(weight * score for weight, score in components) / total_weight


def rank_passages(
    claim: str,
    passages: Sequence[str],
    *,
    features: ClaimFeatures | None = None,
    embedder: object | None = None,
    top_k: int | None = None,
    coverage_weight: float | None = None,
) -> list[str]:
    """Rank passages by cosine similarity plus a claim-coverage bonus."""
    cleaned_claim = _clean(claim)
    cleaned_passages = [_clean(passage) for passage in passages if _clean(passage)]
    if not cleaned_claim or not cleaned_passages:
        return []

    limit = config.V3_PASSAGE_TOP_K if top_k is None else int(top_k)
    if limit <= 0:
        return []
    if len(cleaned_passages) <= limit:
        return cleaned_passages

    if embedder is None:
        from app.ranking import _get_embedder

        embedder = _get_embedder()

    texts = [cleaned_claim, *cleaned_passages]
    try:
        encoded = embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except TypeError:
        encoded = embedder.encode(texts)
    matrix = _normalize_rows(encoded)
    claim_vector = matrix[0]
    similarities = matrix[1:] @ claim_vector

    active_features = features or claim_features(claim)
    weight = (
        config.V3_PASSAGE_COVERAGE_WEIGHT
        if coverage_weight is None
        else float(coverage_weight)
    )
    scores = [
        float(similarity) + weight * _coverage_score(active_features, passage)
        for similarity, passage in zip(similarities, cleaned_passages)
    ]
    order = sorted(range(len(cleaned_passages)), key=lambda i: scores[i], reverse=True)
    return [cleaned_passages[i] for i in order[:limit]]


def aggregate_results(results: Sequence[SchemaNLIResult]) -> SchemaNLIResult:
    """Aggregate passage results without letting neutral passages add weight."""
    cleaned = [result for result in results if result is not None]
    if not cleaned:
        return SchemaNLIResult(0.0, 1.0, 0.0, "NEUTRAL", cached=False)

    supports = [result for result in cleaned if result.stance == "SUPPORTS"]
    refutes = [result for result in cleaned if result.stance == "REFUTES"]

    if supports and refutes:
        return SchemaNLIResult(0.0, 1.0, 0.0, "NEUTRAL", cached=all(r.cached for r in cleaned))
    if supports:
        best = max(supports, key=lambda result: result.p_entail)
        return SchemaNLIResult(
            best.p_entail,
            best.p_neutral,
            best.p_contra,
            "SUPPORTS",
            cached=all(r.cached for r in cleaned),
        )
    if refutes:
        best = max(refutes, key=lambda result: result.p_contra)
        return SchemaNLIResult(
            best.p_entail,
            best.p_neutral,
            best.p_contra,
            "REFUTES",
            cached=all(r.cached for r in cleaned),
        )

    best = max(cleaned, key=lambda result: result.p_neutral)
    return SchemaNLIResult(
        best.p_entail,
        best.p_neutral,
        best.p_contra,
        "NEUTRAL",
        cached=all(r.cached for r in cleaned),
    )


def clause_combine(
    clause_counts: Sequence[dict[str, int]],
) -> str:
    """Apply the coordinated-clause rule.

    Any clause refuted by at least two articles wins.  Otherwise every clause
    must be supported by at least two articles.  The two-article rule is the
    same safety floor used by the decision engine.
    """
    if any(counts.get("refutes", 0) >= 2 for counts in clause_counts):
        return "REFUTED"
    if clause_counts and all(counts.get("supports", 0) >= 2 for counts in clause_counts):
        return "SUPPORTED"
    return "UNVERIFIED"


def passage_diagnostics(
    passages: Sequence[str],
    results: Sequence[SchemaNLIResult],
) -> list[dict[str, object]]:
    return [
        {
            "passage": passage,
            "stance": result.stance,
            "p_entail": result.p_entail,
            "p_neutral": result.p_neutral,
            "p_contra": result.p_contra,
        }
        for passage, result in zip(passages, results)
    ]


def best_passage(
    passages: Sequence[str],
    results: Sequence[SchemaNLIResult],
) -> tuple[str, SchemaNLIResult]:
    if not passages or not results:
        return "", SchemaNLIResult(0.0, 1.0, 0.0, "NEUTRAL", cached=False)
    pairs = list(zip(passages, results))
    pairs.sort(key=lambda pair: max(pair[1].p_entail, pair[1].p_contra), reverse=True)
    return pairs[0]


def support_probability(result: NLIResult | SchemaNLIResult) -> float:
    return float(result.p_entail)


def is_finite(value: float) -> bool:
    return math.isfinite(float(value))
