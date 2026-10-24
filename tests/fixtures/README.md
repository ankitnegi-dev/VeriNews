# Retrieval test fixtures

These are offline response-shape fixtures used by `tests/test_retrieval.py`.
They contain no API keys and no real request metadata.

They are placeholders created without a live key. Once `.env` contains a
NewsAPI or GNews key, they should be replaced with sanitized live responses
from 3-5 real searches. The tests themselves do not need the network.
