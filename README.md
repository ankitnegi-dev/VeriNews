# VeriNews

A real-time, evidence-based news claim verification prototype built with Python and FastAPI.

Given a claim, the planned pipeline extracts entities, retrieves recent news articles, ranks
evidence, runs a pretrained NLI model, weights sources by credibility, and returns a verdict
(`SUPPORTED`, `REFUTED`, or `UNVERIFIED`) with ranked evidence.

Status: **Phase 3 - news retrieval (offline tests pass; live check completed for one claim and cached)**.
See `WORK_MAP.md` for the full build plan.
