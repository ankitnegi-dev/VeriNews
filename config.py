"""Central configuration for VeriNews.

Every model name, path, threshold and weight lives here so experiments can be
compared and tuned without hunting through the code.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ISOT_DIR = DATA_DIR / "raw" / "ISOT"
ISOT_FAKE_PATH = ISOT_DIR / "Fake.csv"
ISOT_TRUE_PATH = ISOT_DIR / "True.csv"
FEVER_DIR = DATA_DIR / "raw" / "fever"
FEVER_CSV_PATH = FEVER_DIR / "fever.csv"
FEVER_JSONL_PATH = FEVER_DIR / "train.jsonl"

CACHE_DIR = BASE_DIR / ".cache"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
CREDIBILITY_PATH = BASE_DIR / "credibility" / "domains.yaml"
ENV_PATH = BASE_DIR / ".env"

BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"

# ---------------------------------------------------------------------------
# Secrets (loaded from .env; never print these values)
# ---------------------------------------------------------------------------
load_dotenv(ENV_PATH)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

# ---------------------------------------------------------------------------
# Models and hardware
# ---------------------------------------------------------------------------
SPACY_MODEL_NAME = "en_core_web_sm"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_FALLBACK_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
NLI_MAX_TOKENS = 512

# Transformers uses 0 for the first GPU and -1 for CPU.
DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# ISOT baseline (Phase 1)
# ---------------------------------------------------------------------------
BASELINE_TEST_SIZE = 0.2
BASELINE_RANDOM_STATE = 42
BASELINE_TFIDF_MAX_FEATURES = 50_000
BASELINE_TFIDF_STOP_WORDS = "english"
BASELINE_LOGREG_MAX_ITER = 1_000

# ---------------------------------------------------------------------------
# Claim analysis (Phase 2)
# ---------------------------------------------------------------------------
MAX_CLAIM_CHARS = 500
MIN_CLAIM_CHARS = 3
MIN_CLAIM_WORDS = 3
MAX_QUERIES = 3
MAX_QUERY_WORDS = 12

# Entity types worth searching for. LOC, FAC and PRODUCT cover landmarks and
# goods; LAW covers bills, acts and court rulings.
ENTITY_LABELS = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "FAC",
        "PRODUCT",
        "LAW",
        "DATE",
        "EVENT",
        "NORP",
        "MONEY",
        "PERCENT",
        "CARDINAL",
    }
)

# ---------------------------------------------------------------------------
# Retrieval (Phase 3)
# ---------------------------------------------------------------------------
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
GNEWS_BASE_URL = "https://gnews.io/api/v4/search"
NEWS_LANGUAGE = "en"
NEWSAPI_PAGE_SIZE = 20
GNEWS_MAX_RESULTS = 10
MAX_RESULTS_PER_QUERY = 20
# Keep evaluation results cached through the final demo. The free NewsAPI search
# window is about one month, so a 14-day TTL avoids re-fetching the same claims.
CACHE_TTL_HOURS = 14 * 24
CACHE_ONLY = os.getenv("CACHE_ONLY", "false").strip().lower() == "true"
DAILY_REQUEST_LIMIT = 80
RETRIEVAL_TIMEOUT_SECONDS = 20
RETRIEVAL_CACHE_DIR = CACHE_DIR / "retrieval"
REQUEST_COUNT_PATH = CACHE_DIR / "request_count.json"

# ---------------------------------------------------------------------------
# Ranking (Phase 4)
# ---------------------------------------------------------------------------
TOP_K = 6
# PROVISIONAL: this came from the early Skerritt check, and that claim is in
# split=test. Recalibrate later using only rows with split=dev in
# tests/data/claims.csv. Never tune on split=test.
MIN_SIMILARITY = 0.35
SNIPPET_MAX_CHARS = 400
RECENCY_FULL_CREDIT_DAYS = 3
RECENCY_ZERO_CREDIT_DAYS = 30
RECENCY_MIN_FACTOR = 0.6
RECENCY_UNKNOWN_DATE_FACTOR = 0.8

# ---------------------------------------------------------------------------
# NLI (Phase 5)
# ---------------------------------------------------------------------------
NLI_SUPPORT_THRESHOLD = 0.5
NLI_REFUTE_THRESHOLD = 0.5
NLI_CACHE_DIR = CACHE_DIR / "nli"
NLI_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Credibility and decision (Phase 6)
# ---------------------------------------------------------------------------
CREDIBILITY_HIGH = 1.0
CREDIBILITY_MEDIUM = 0.7
CREDIBILITY_LOW = 0.4
CREDIBILITY_BLOCKED = 0.2

MIN_EVIDENCE = 2
MIN_TOTAL_WEIGHT = 0.5
DECISION_RATIO = 0.4
MARGIN = 1.5

# ---------------------------------------------------------------------------
# API (Phase 7)
# ---------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 8000
