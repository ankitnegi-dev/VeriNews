"""Feasibility gate: can we fetch full article text for the cached articles?

For every kept article in the cached retrieval results, check robots.txt, fetch
the page with trafilatura (8 second timeout, one request per second per domain),
and record the extracted text length. The gate passes at a 50% fetch success
rate. Results are written to ``notes/fulltext_feasibility.json``.

Usage:
    python scripts/fulltext_feasibility.py
"""

from pathlib import Path
import csv, hashlib, json, re, sys, time, urllib.robotparser, statistics
from urllib.parse import urlparse

import requests
import trafilatura

sys.path.insert(0, str(Path.cwd()))
from app.claim_analysis import analyze
from app.ranking import rank_articles
from app.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "fulltext"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UA = "VeriNews/1.0 (+https://github.com/ankitnegi-dev/VeriNews; passage-NLI feasibility; contact: local research)"
TIMEOUT = 8
STOPWORDS = {
    "the","a","an","and","or","but","of","to","in","on","at","for","with","by",
    "from","as","is","was","were","be","been","being","this","that","these","those",
    "it","its","he","she","they","them","their","his","her","we","our","you","your",
    "will","would","could","should","may","might","can","not","no","yes","about",
    "into","over","after","before","than","then","so","if","when","where","who",
    "what","which","how","why","all","any","some","more","most","other","such",
    "new","one","two","three","said","says","report","reported",
}

def cache_path(url):
    return CACHE_DIR / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")

def key_terms(claim):
    terms=set()
    for token in re.findall(r"[A-Za-z0-9$][A-Za-z0-9$.,-]*", claim):
        low=token.lower().strip(".,")
        if len(low) >= 3 and low not in STOPWORDS:
            terms.add(low)
    return terms

def domain_of(url):
    return urlparse(url).netloc.lower()

robots_cache = {}
last_request = {}

def rate_limit(domain):
    last = last_request.get(domain, 0.0)
    wait = 1.0 - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait)
    last_request[domain] = time.monotonic()

def robots_allowed(domain, url):
    if domain not in robots_cache:
        rate_limit(domain)
        robots_url = f"https://{domain}/robots.txt"
        try:
            r = requests.get(robots_url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            parser = urllib.robotparser.RobotFileParser()
            if r.status_code == 200:
                parser.parse(r.text.splitlines())
            else:
                parser = None
            robots_cache[domain] = parser
        except Exception:
            robots_cache[domain] = None
    parser = robots_cache[domain]
    if parser is None:
        return True
    try:
        return parser.can_fetch(UA, url)
    except Exception:
        return True

def fetch(url):
    cp = cache_path(url)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    domain = domain_of(url)
    allowed = robots_allowed(domain, url)
    record = {
        "url": url,
        "domain": domain,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robots_allowed": allowed,
        "status_code": None,
        "error": None,
        "text": "",
        "text_length": 0,
    }
    if not allowed:
        record["error"] = "robots_disallowed"
        cp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return record

    rate_limit(domain)
    try:
        response = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)
        record["status_code"] = response.status_code
        if response.status_code >= 400:
            record["error"] = f"http_{response.status_code}"
        else:
            text = trafilatura.extract(
                response.text,
                url=response.url,
                output_format="txt",
                include_comments=False,
                include_tables=False,
                include_images=False,
                include_links=False,
                no_fallback=False,
            )
            if text:
                record["text"] = " ".join(text.split())
                record["text_length"] = len(record["text"])
            else:
                record["error"] = "extraction_empty"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    cp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record

claims = list(csv.DictReader(open(ROOT / "tests/data/claims.csv", encoding="utf-8", newline="")))
kept = []
for row in claims:
    claim = " ".join((row["claim"] or "").split())
    analysis = analyze(claim)
    articles = retrieve(analysis.queries, cache_only=True)
    ranked = rank_articles(analysis.text, articles)
    for r in ranked:
        kept.append((row, r))

print(f"claims={len(claims)} kept_articles={len(kept)} domains={len(set(r.source_domain for _,r in kept))}", flush=True)

success = 0
lengths = []
keyterm_articles = 0
results = []
for i, (row, article) in enumerate(kept, 1):
    rec = fetch(article.url)
    text = rec.get("text", "")
    ok = bool(text) and len(text) >= 100
    if ok:
        success += 1
        lengths.append(len(text))
    terms = key_terms(row["claim"])
    full_low = text.lower()
    snippet_low = article.snippet.lower()
    new_terms = [t for t in terms if t in full_low and t not in snippet_low]
    if new_terms:
        keyterm_articles += 1
    results.append({
        "claim": row["claim"],
        "split": row["split"],
        "url": article.url,
        "domain": article.source_domain,
        "ok": ok,
        "length": len(text),
        "error": rec.get("error"),
        "robots_allowed": rec.get("robots_allowed"),
        "new_key_terms": new_terms[:10],
    })
    print(f"[{i}/{len(kept)}] {article.source_domain} ok={ok} len={len(text)} err={rec.get('error')} new_terms={len(new_terms)}", flush=True)

summary = {
    "claims": len(claims),
    "kept_articles": len(kept),
    "domains": len(set(r.source_domain for _,r in kept)),
    "fetch_success": success,
    "fetch_success_rate": success / len(kept) if kept else 0.0,
    "median_text_length": statistics.median(lengths) if lengths else 0,
    "mean_text_length": statistics.mean(lengths) if lengths else 0,
    "articles_with_new_key_terms": keyterm_articles,
    "articles_with_new_key_terms_rate": keyterm_articles / len(kept) if kept else 0.0,
    "errors": {},
}
for r in results:
    err = r["error"] or "ok"
    summary["errors"][err] = summary["errors"].get(err, 0) + 1

(ROOT / "notes").mkdir(exist_ok=True)
(ROOT / "notes" / "fulltext_feasibility.json").write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
