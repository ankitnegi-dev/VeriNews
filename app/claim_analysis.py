"""Turn a free-text claim into entities and short search queries.

This is the first pipeline stage. It uses spaCy's small English model only for
named entity recognition and basic syntax (verbs / noun chunks). Query building
is deliberately rule-based: it is easy to explain, fast, and does not need an
LLM call.
"""

from __future__ import annotations

import logging
import re

import spacy
from spacy.tokens import Doc, Token

import config
from app.schemas import Claim

logger = logging.getLogger(__name__)

# Organizations and people make better search anchors than dates or numbers.
# Places and landmarks come next, then context-like labels.
ENTITY_PRIORITY = {
    "ORG": 0,
    "PERSON": 1,
    "GPE": 2,
    "LOC": 3,
    "FAC": 4,
    "EVENT": 5,
    "NORP": 6,
    "PRODUCT": 7,
    "LAW": 8,
    "DATE": 9,
    "MONEY": 10,
    "PERCENT": 11,
    "CARDINAL": 12,
}

_NLP = None


def _get_nlp():
    """Load spaCy once, on first use, so importing the module stays cheap."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(config.SPACY_MODEL_NAME)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {config.SPACY_MODEL_NAME!r} is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP


def _normalize_claim(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Claim must be a string.")

    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError("Claim is empty. Please provide a news claim.")
    if len(cleaned) < config.MIN_CLAIM_CHARS or len(cleaned.split()) < config.MIN_CLAIM_WORDS:
        raise ValueError(
            "Claim is too short. Please provide at least "
            f"{config.MIN_CLAIM_WORDS} words."
        )
    if len(cleaned) > config.MAX_CLAIM_CHARS:
        raise ValueError(
            f"Claim is too long. Maximum length is {config.MAX_CLAIM_CHARS} characters."
        )
    return cleaned


def _clean_entity_text(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    return cleaned.strip(".,;:!?\"'()[]{}").strip()


def _extract_entities(doc) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for entity in doc.ents:
        if entity.label_ not in config.ENTITY_LABELS:
            continue
        cleaned = _clean_entity_text(entity.text)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        entities.append({"text": cleaned, "label": entity.label_})
    return entities


def _ordered_entities(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort by search usefulness, preserving original order for ties."""
    indexed = list(enumerate(entities))
    indexed.sort(key=lambda pair: (ENTITY_PRIORITY.get(pair[1]["label"], 99), pair[0]))
    return [entity for _, entity in indexed]


def _keyword_tokens(doc: Doc) -> list[Token]:
    return [
        token
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]


def _join_source_tokens(doc: Doc, tokens: list[Token]) -> str:
    """Join tokens using the original text, so '$2.5' stays '$2.5'.

    The gap between two selected tokens is copied when it has no whitespace
    (for example the hyphen in 'COVID-19'), and reduced to one space otherwise.
    """
    if not tokens:
        return ""

    parts = [tokens[0].text]
    previous_end = tokens[0].idx + len(tokens[0].text)
    for token in tokens[1:]:
        gap = doc.text[previous_end : token.idx]
        separator = " " if any(character.isspace() for character in gap) else gap
        parts.append(f"{separator}{token.text}")
        previous_end = token.idx + len(token.text)
    return "".join(parts)


def _main_verb_or_chunk(doc: Doc) -> str:
    root = doc[:].root
    if root.pos_ == "VERB":
        return root.text
    for token in doc:
        if token.pos_ == "VERB":
            return token.text
    for chunk in doc.noun_chunks:
        return chunk.text
    return ""


def _clean_query(query: str) -> str:
    # Search engines do not need commas, brackets or quotes inside a query.
    query = re.sub(r"[,;:!?\"'()\[\]{}]", " ", query)
    query = " ".join(query.split())
    return query.strip(" .")


def _query_words(query: str) -> list[str]:
    """Tokenize a query for order-insensitive duplicate detection."""
    return re.findall(r"[a-z0-9$%]+(?:[.\-][a-z0-9$%]+)*", query.lower())


def _query_key(query: str) -> str:
    return " ".join(sorted(_query_words(query)))


def _add_query(queries: list[str], seen: set[str], candidate: str) -> None:
    candidate = _clean_query(candidate)
    words = candidate.split()
    if len(words) < 2:
        return
    candidate = " ".join(words[: config.MAX_QUERY_WORDS])
    key = _query_key(candidate)
    if not key or key in seen:
        return
    seen.add(key)
    queries.append(candidate)


def _build_queries(doc: Doc, entities: list[dict[str, str]]) -> list[str]:
    keyword_tokens = _keyword_tokens(doc)
    ordered = _ordered_entities(entities)
    context = _main_verb_or_chunk(doc)
    entity_keys = {entity["text"].lower() for entity in ordered}

    # Do not build a query such as "Paris Paris" when the noun chunk is itself
    # an entity. The fallbacks below will still fill the third slot.
    if context.lower() in entity_keys:
        context = ""

    candidates: list[str] = []

    # Rule 1: the claim itself, stopwords removed, capped at 12 words.
    candidates.append(_join_source_tokens(doc, keyword_tokens))

    # Rule 2: entities joined, with organizations/people/places first.
    candidates.append(" ".join(entity["text"] for entity in ordered[:4]))

    # Rule 3: top entities plus the main verb or noun chunk.
    if ordered:
        top_entities = [entity["text"] for entity in ordered[:2]]
        candidates.append(" ".join([*top_entities, context]) if context else "")
    elif context and keyword_tokens:
        candidates.append(f"{context} {keyword_tokens[0].text}")
    else:
        candidates.append(context)

    # Fallbacks for unusual claims with few named entities.
    chunks = [chunk.text for chunk in doc.noun_chunks]
    if len(chunks) >= 2:
        candidates.append(" ".join(chunks[:2]))
    if len(keyword_tokens) >= 2:
        candidates.append(_join_source_tokens(doc, keyword_tokens[:2]))
    candidates.append(_join_source_tokens(doc, list(doc)))

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        _add_query(queries, seen, candidate)
        if len(queries) >= config.MAX_QUERIES:
            break
    return queries


def build_queries(text: str, entities: list[dict[str, str]]) -> list[str]:
    """Build up to three deduplicated, multi-word search queries."""
    cleaned = " ".join(text.split())
    doc = _get_nlp()(cleaned)
    return _build_queries(doc, entities)


def analyze(text: str) -> Claim:
    """Extract entities and generate search queries for one claim."""
    cleaned = _normalize_claim(text)
    doc = _get_nlp()(cleaned)
    entities = _extract_entities(doc)
    queries = _build_queries(doc, entities)

    if len(queries) < 2:
        raise ValueError(
            "Could not build at least two useful search queries from this claim. "
            "Please make it more specific."
        )

    logger.debug("analyzed claim %r -> %s entities, %s queries", cleaned, len(entities), len(queries))
    return Claim(text=cleaned, entities=entities, queries=queries)
