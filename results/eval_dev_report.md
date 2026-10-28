# VeriNews evaluation - dev

Generated: 2026-10-28T04:45:41.144080+00:00

Retrieval: cache-only. Models were warmed before timing.

## Metrics

| Metric | Value |
| --- | ---: |
| Claims | 10 |
| Exact accuracy | 0.800 (8/10) |
| Wrong-verdict rate (SUPPORTED vs REFUTED flips) | 0.000 (0/6) |
| Coverage on directional claims | 0.667 (4/6) |
| Mean latency | 4153 ms |
| Max latency | 6265 ms |

Wrong-verdict rate counts only claims whose expected label is SUPPORTED or REFUTED. Coverage uses the same denominator and counts predictions that committed to either direction.

## Per-class precision and recall

| Class | Precision | Recall | Gold support | Predicted |
| --- | ---: | ---: | ---: | ---: |
| SUPPORTED | 1.000 | 0.667 | 3 | 2 |
| REFUTED | 1.000 | 0.667 | 3 | 2 |
| UNVERIFIED | 0.667 | 1.000 | 4 | 6 |

## Confusion matrix

Rows are the expected label. Columns are the predicted label.

| Expected \ Predicted | SUPPORTED | REFUTED | UNVERIFIED |
| --- | ---: | ---: | ---: |
| SUPPORTED | 2 | 0 | 1 |
| REFUTED | 0 | 2 | 1 |
| UNVERIFIED | 0 | 0 | 4 |

## Mistakes

### 1. Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cents per share.

- Expected: `SUPPORTED`
- Predicted: `UNVERIFIED`
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.000
- Total weight: 0.000
- Latency: 4995 ms
- Explanation: UNVERIFIED: no relevant article supports or refutes the claim; neutral evidence is treated as no information; 6 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | biztoc.com | relevance 0.889 | credibility 0.40 | P(entail) 0.002 | P(contra) 0.000 | [The Truth About Netflix’s Massive Stock Buybacks](https://biztoc.com/x/d68bcd9bdbccda85)
  - `NEUTRAL` | thewrap.com | relevance 0.797 | credibility 0.40 | P(entail) 0.013 | P(contra) 0.015 | [Netflix Warns of Disappointing Q4 Revenue As Growth Questions Remain](https://www.thewrap.com/industry-news/business/netflix-earnings-q3-2026/)
  - `NEUTRAL` | variety.com | relevance 0.782 | credibility 0.70 | P(entail) 0.001 | P(contra) 0.000 | [Netflix Q3 Revenue Meets Wall Street Estimates, Q4 Earnings Will Be Crimped by Costs Associated With Mass Layoffs](https://variety.com/2026/tv/news/netflix-q3-2026-earnings-wall-street-layoffs-exec-firings-1236917438/)
  - `NEUTRAL` | theinformation.com | relevance 0.762 | credibility 0.40 | P(entail) 0.005 | P(contra) 0.042 | [Netflix Revenue Growth Hits Three-Year Low](https://www.theinformation.com/briefings/netflix-revenue-growth-hits-three-year-low)
  - `NEUTRAL` | economictimes.indiatimes.com | relevance 0.743 | credibility 0.70 | P(entail) 0.002 | P(contra) 0.005 | [Netflix issues weak fourth-quarter forecast, shares drop](https://economictimes.indiatimes.com/tech/technology/netflix-issues-weak-fourth-quarter-forecast-shares-drop/articleshow/135079840.cms)
  - `NEUTRAL` | gizmodo.com | relevance 0.706 | credibility 0.40 | P(entail) 0.008 | P(contra) 0.050 | [Netflix Promises To ‘Do Better’](https://gizmodo.com/netflix-promises-to-do-better-2000828357)

### 2. A magnitude 7.7 earthquake struck southern Panama on October 9, 2026, and the Panama Canal suspended all ship passage.

- Expected: `REFUTED`
- Predicted: `UNVERIFIED`
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.265
- Total weight: 0.311
- Latency: 2515 ms
- Explanation: UNVERIFIED: only one relevant article refutes the claim; a single article cannot decide a verdict; top refuting: cbc.ca; 5 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | hurriyetdailynews.com | relevance 0.867 | credibility 0.40 | P(entail) 0.002 | P(contra) 0.002 | [Powerful 7.7 quake hits Panama, damages buildings](https://www.hurriyetdailynews.com/powerful-7-7-quake-hits-panama-damages-buildings-227958)
  - `NEUTRAL` | israelnationalnews.com | relevance 0.809 | credibility 0.40 | P(entail) 0.004 | P(contra) 0.016 | [Panama rocked by 7.7 magnitude earthquake](https://www.israelnationalnews.com/news/434384)
  - `NEUTRAL` | freerepublic.com | relevance 0.818 | credibility 0.40 | P(entail) 0.002 | P(contra) 0.003 | [Panama struck by magnitude 7.7 earthquake, U.S. Geological Survey says](https://freerepublic.com/focus/f-news/4398607/posts)
  - `NEUTRAL` | dw.com | relevance 0.802 | credibility 1.00 | P(entail) 0.003 | P(contra) 0.023 | [Strong earthquakes shake Panama, prompting tsunami alert](https://www.dw.com/en/strong-earthquakes-shake-panama-prompting-tsunami-alert/a-79623168)
  - `REFUTES` | cbc.ca | relevance 0.779 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.852 | [Powerful 7.7 magnitude earthquake rocks Panama](https://www.cbc.ca/news/world/panama-earthquake-9.7376897)
  - `NEUTRAL` | business-standard.com | relevance 0.770 | credibility 0.70 | P(entail) 0.004 | P(contra) 0.110 | [Panama hit by strongest quake in 75 years, 7.7-magnitude tremor strikes](https://www.business-standard.com/world-news/panama-hit-by-strongest-quake-in-75-years-7-7-magnitude-tremor-strikes-126101000149_1.html)

## Credibility ablation

All evidence credibility scores were set to 1.0 and the decision was re-run with the same NLI probabilities. Retrieval, ranking, and NLI were not re-run.

| Metric | Real tiers | All weights 1.0 |
| --- | ---: | ---: |
| Exact accuracy | 0.800 (8/10) | 0.800 (8/10) |
| Wrong-verdict rate | 0.000 (0/6) | 0.000 (0/6) |
| Coverage | 0.667 (4/6) | 0.667 (4/6) |

| Claim | Expected | Real tiers | All weights 1.0 | Changed |
| --- | --- | --- | --- | --- |
| South African human rights lawyer Navi Pillay won the 2026 Nobel Peace Prize. | SUPPORTED | SUPPORTED | SUPPORTED | no |
| Canadian poet Anne Carson won the 2026 Nobel Prize in Literature. | SUPPORTED | SUPPORTED | SUPPORTED | no |
| Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cent... | SUPPORTED | UNVERIFIED | UNVERIFIED | no |
| Manchester United beat Como 4-1 in a Champions League match on October 21, 2026. | REFUTED | REFUTED | REFUTED | no |
| SpaceX launched a Dragon cargo spacecraft to the International Space Station from Van... | REFUTED | REFUTED | REFUTED | no |
| A magnitude 7.7 earthquake struck southern Panama on October 9, 2026, and the Panama... | REFUTED | UNVERIFIED | UNVERIFIED | no |
| Netflix will cut its subscription prices for every customer next week, according to a... | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| Bitcoin will trade above $250,000 before the end of 2027. | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| Several unnamed engineers at a major chip company are secretly planning to quit next... | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| An anonymous source says a famous Bollywood actor will announce retirement next week. | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |

