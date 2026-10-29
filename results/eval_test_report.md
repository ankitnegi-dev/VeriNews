# VeriNews evaluation - test

Generated: 2026-10-29T07:36:17.003576+00:00

Retrieval: cache-only. Models were warmed before timing.

## Metrics

| Metric | Value |
| --- | ---: |
| Claims | 20 |
| Exact accuracy | 0.600 (12/20) |
| Wrong-verdict rate (SUPPORTED vs REFUTED flips) | 0.071 (1/14) |
| Coverage on directional claims | 0.500 (7/14) |
| Mean latency | 4692 ms |
| Max latency | 7899 ms |

Wrong-verdict rate counts only claims whose expected label is SUPPORTED or REFUTED. Coverage uses the same denominator and counts predictions that committed to either direction.

## Post-test review

The test split was evaluated once, cache-only, after the `v0.8-frozen` tag. No
ranking, NLI, credibility, or decision code was changed after seeing these
results. The only post-run edits are reporting metadata: one category per
mistake and this statement. No crash or code-level wrong-output bug was found,
so no before/after rerun was needed.

## Per-class precision and recall

| Class | Precision | Recall | Gold support | Predicted |
| --- | ---: | ---: | ---: | ---: |
| SUPPORTED | 1.000 | 0.143 | 7 | 1 |
| REFUTED | 0.833 | 0.714 | 7 | 6 |
| UNVERIFIED | 0.462 | 1.000 | 6 | 13 |

## Confusion matrix

Rows are the expected label. Columns are the predicted label.

| Expected \ Predicted | SUPPORTED | REFUTED | UNVERIFIED |
| --- | ---: | ---: | ---: |
| SUPPORTED | 1 | 1 | 5 |
| REFUTED | 0 | 5 | 2 |
| UNVERIFIED | 0 | 0 | 6 |

## Mistakes

### 1. Max Verstappen won the 2026 Singapore Grand Prix, finishing ahead of Charles Leclerc.

- Expected: `SUPPORTED`
- Predicted: `UNVERIFIED`
- Category: NLI too strict
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.000
- Total weight: 0.000
- Latency: 6791 ms
- Explanation: UNVERIFIED: no relevant article supports or refutes the claim; neutral evidence is treated as no information; 6 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | sports.yahoo.com | relevance 0.839 | credibility 0.40 | P(entail) 0.003 | P(contra) 0.005 | [Verstappen’s Singapore Win Left the Title Alone and Blew Up the Fight for Second](https://sports.yahoo.com/articles/verstappen-singapore-win-left-title-170000937.html)
  - `NEUTRAL` | motorsport.com | relevance 0.723 | credibility 0.40 | P(entail) 0.003 | P(contra) 0.007 | [Charles Leclerc slams Isack Hadjar's Max Verstappen Ferrari claim as "nonsense"](https://www.motorsport.com/f1/news/charles-leclerc-slams-isack-hadjars-max-verstappen-ferrari-claim-as-nonsense/10867179/)
  - `NEUTRAL` | tribune.newspackstaging.com | relevance 0.857 | credibility 0.40 | P(entail) 0.005 | P(contra) 0.073 | [Resurgent Verstappen wins Singapore Grand Prix for first time](https://tribune.newspackstaging.com/2026/10/12/resurgent-verstappen-wins-singapore-grand-prix-for-first-time/)
  - `NEUTRAL` | bbc.co.uk | relevance 0.859 | credibility 1.00 | P(entail) 0.005 | P(contra) 0.022 | [Verstappen perfect and Antonelli hugely mature - driver ratings](https://www.bbc.co.uk/sport/formula1/articles/c64g10qd44gjo)
  - `NEUTRAL` | gpblog.com | relevance 0.877 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.000 | [Verstappen wins Sprint ahead of Hamilton as Antonelli capitalizes on Russell's DNF](https://www.gpblog.com/en/breaking-news/verstappen-wins-sprint-ahead-of-hamilton-as-antonelli-capitalizes-on-russells-dnf)
  - `NEUTRAL` | leadlapracing.com | relevance 0.847 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.000 | [Max Verstappen Dominates Singapore Grand Prix for First Marina Bay Victory](https://leadlapracing.com/max-verstappen-dominates-singapore-grand-prix-for-first-marina-bay-victory/)

### 2. West Indies beat India by six wickets in the second T20I in Ranchi on October 9, 2026.

- Expected: `SUPPORTED`
- Predicted: `REFUTED`
- Category: retrieval miss
- Confidence: 0.597
- Support score: 0.000
- Refute score: 0.662
- Total weight: 1.109
- Latency: 5609 ms
- Explanation: REFUTED: 3 relevant sources refute the claim (top: business-standard.com, channelnewsasia.com); support 0.000 vs refute 0.662; 3 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | aljazeera.com | relevance 0.779 | credibility 0.70 | P(entail) 0.005 | P(contra) 0.087 | [West Indies beat India to level T20 series as Hope continues record run](https://www.aljazeera.com/sports/2026/10/14/west-indies-beat-india-to-level-series-as-hope-continues-record-run)
  - `NEUTRAL` | sportingnews.com | relevance 0.733 | credibility 0.40 | P(entail) 0.003 | P(contra) 0.205 | [Where to watch India vs. West Indies live stream, TV channel, start time and lineups for 5th T20I in Bengaluru](https://www.sportingnews.com/in/cricket/news/watch-india-west-indies-live-stream-tv-channel-time-lineups-5th-t20i-bengaluru/3b9d25a5fa2020ec478427d8)
  - `REFUTES` | sports.yahoo.com | relevance 0.734 | credibility 0.40 | P(entail) 0.005 | P(contra) 0.575 | [Shai Hope’s 102 gives West Indies series-leveling win over India in fourth cricket T20](https://sports.yahoo.com/articles/shai-hope-102-gives-west-193242328.html)
  - `REFUTES` | channelnewsasia.com | relevance 0.853 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.629 | [India beat West Indies by eight wickets in first T20](https://www.channelnewsasia.com/sport/india-beat-west-indies-eight-wickets-in-first-t20-6437271)
  - `REFUTES` | business-standard.com | relevance 0.677 | credibility 0.70 | P(entail) 0.005 | P(contra) 0.588 | [India vs West Indies 3rd T20I: Playing 11, live toss time (IST), streaming](https://www.business-standard.com/cricket/news/india-vs-west-indies-3rd-t20i-playing-11-live-toss-time-ist-streaming-126101000329_1.html)
  - `NEUTRAL` | cricketnews.com | relevance 0.739 | credibility 0.40 | P(entail) 0.006 | P(contra) 0.222 | [IND vs WI T20 2026 schedule: Start dates, venues and timings for every T20I match of West Indies' tour of India](https://www.cricketnews.com/en/cricket/news/ind-vs-wi-t20-2026-schedule-start-dates-venues-timings-t20i-match/e1c5e4883bb9e3acc9dd285e)

### 3. The Reserve Bank of India raised its repo rate by 25 basis points to 5.50 percent on October 7, 2026.

- Expected: `SUPPORTED`
- Predicted: `UNVERIFIED`
- Category: NLI too strict
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.000
- Total weight: 0.000
- Latency: 3704 ms
- Explanation: UNVERIFIED: no relevant article supports or refutes the claim; neutral evidence is treated as no information; 4 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | economictimes.indiatimes.com | relevance 0.930 | credibility 0.70 | P(entail) 0.029 | P(contra) 0.086 | [Higher EMIs ahead: PNB, Bank of India, Bank of Baroda and Indian Bank raise lending rates; check details](https://economictimes.indiatimes.com/wealth/borrow/higher-emis-ahead-pnb-bank-of-india-bank-of-baroda-and-indian-bank-raise-lending-rates-check-details/articleshow/134782736.cms)
  - `NEUTRAL` | livemint.com | relevance 0.827 | credibility 0.70 | P(entail) 0.005 | P(contra) 0.009 | [RBI has hiked the repo rate at last but can its monetary policy committee make up for lost time?](https://www.livemint.com/opinion/online-views/rbi-hikes-its-policy-rate-but-can-its-monetary-policy-committee-make-up-for-lost-time-11791285221067.html)
  - `NEUTRAL` | thehindubusinessline.com | relevance 0.773 | credibility 0.40 | P(entail) 0.066 | P(contra) 0.389 | [RBI seen beginning rate-hike cycle, repo rate may reach 6% by FY27 as inflation risks rise: Reports](https://www.thehindubusinessline.com/money-and-banking/rbi-seen-beginning-rate-hike-cycle-repo-rate-may-reach-6-by-fy27-as-inflation-risks-rise-reports/article71549894.ece)
  - `NEUTRAL` | ibtimes.com.au | relevance 0.580 | credibility 0.40 | P(entail) 0.005 | P(contra) 0.004 | [The Nifty Falls 0.8 Percent to 22,587 After the RBI Raises Rates and Crude Oil Holds Above $101 a Barrel](https://www.ibtimes.com.au/india-nifty-falls-after-rbi-rate-hike-1876390)

### 4. TSMC reported third-quarter 2026 net profit of NT$763.06 billion, up 68.7 percent from a year earlier.

- Expected: `SUPPORTED`
- Predicted: `UNVERIFIED`
- Category: NLI too strict
- Confidence: 0.189
- Support score: 0.351
- Refute score: 0.518
- Total weight: 0.886
- Latency: 4136 ms
- Explanation: UNVERIFIED: no side clears the decision margin (support 0.351 vs refute 0.518; margin 1.50); top supporting: digitimes.com; top refuting: cryptobriefing.com, techpinions.com; 3 relevant neutral article(s) treated as no information.
- Evidence:
  - `SUPPORTS` | digitimes.com | relevance 0.884 | credibility 0.40 | P(entail) 0.991 | P(contra) 0.002 | [TSMC's AI-fuelled profit streak hits sixth straight record in 3Q26, but 2nm expansion weighs on margins](https://www.digitimes.com/news/a20261015VL218/tsmc-profit-2nm-revenue-2026.html)
  - `NEUTRAL` | biztoc.com | relevance 0.782 | credibility 0.40 | P(entail) 0.006 | P(contra) 0.003 | [After Elon Musk’s Invitation, TSMC CEO Reportedly Says Open To Working On Terafab But Must Run It](https://biztoc.com/x/1e6eca345ef5ed81)
  - `REFUTES` | techpinions.com | relevance 0.651 | credibility 0.40 | P(entail) 0.002 | P(contra) 0.971 | [Chip Stocks Slid for Six Straight Days. TSMC’s CEO Says AI Demand Still Exceeds Supply.](https://techpinions.com/tsmc-q3-2026-ai-demand-chip-selloff/)
  - `NEUTRAL` | economictimes.indiatimes.com | relevance 0.706 | credibility 0.70 | P(entail) 0.002 | P(contra) 0.004 | [Global Market: TSMC set for sixth straight record quarterly profit on AI chip boom](https://economictimes.indiatimes.com/markets/us-stocks/wall-street-guide/global-market-tsmc-set-for-sixth-straight-record-quarterly-profit-on-ai-chip-boom/articleshow/134957993.cms)
  - `REFUTES` | cryptobriefing.com | relevance 0.680 | credibility 0.40 | P(entail) 0.003 | P(contra) 0.975 | [TSMC posts $46.72 billion Q3 2026 revenue, up 41.2% as AI demand keeps climbing](https://cryptobriefing.com/tsmc-q3-2026-earnings-revenue-up-41/)
  - `NEUTRAL` | observer.com | relevance 0.513 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.002 | [Chipmaking Titan C.C. Wei Eyes Texas as TSMC’s A.I. Demand Surges](https://observer.com/2026/10/cc-wei-tsmc-texas-terafab-chip-expansion/)

### 5. Bayern Munich and Arsenal drew 2-2 in a Champions League match on October 21, 2026.

- Expected: `SUPPORTED`
- Predicted: `UNVERIFIED`
- Category: retrieval miss
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.000
- Total weight: 0.000
- Latency: 4882 ms
- Explanation: UNVERIFIED: no relevant article supports or refutes the claim; neutral evidence is treated as no information; 6 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | footballmuse.com | relevance 0.624 | credibility 0.40 | P(entail) 0.010 | P(contra) 0.023 | [Gerrard hails Harry Kane as greatness and backs Bayern star to win the Ballon d’Or](https://www.footballmuse.com/news/gerrard-hails-kane-as-greatness-and-backs-bayern-star-to-win-the-ballon-dor)
  - `NEUTRAL` | techradar.com | relevance 0.629 | credibility 0.40 | P(entail) 0.016 | P(contra) 0.498 | [How to watch Bayern Munich vs Arsenal: FREE live Streams, TV channels & kick-off time for Champions League 2026/27](https://www.techradar.com/entertainment/football/bayern-munich-vs-arsenal-free-champions-league-2026-27)
  - `NEUTRAL` | usatoday.com | relevance 0.619 | credibility 0.40 | P(entail) 0.007 | P(contra) 0.176 | [Bayern Munich-Arsenal LIVE UEFA Champions League updates, starting XIs & more](https://www.usatoday.com/story/sports/soccer/ucl/2026/10/21/bayern-munich-arsenal-live-uefa-champions-league-updates-starting-xis-more--live/92365766007/)
  - `NEUTRAL` | worldsoccertalk.com | relevance 0.550 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.001 | [How to watch Bayern Munich vs Arsenal match in the USA: Live stream and TV for 2026-27 UEFA Champions League](https://worldsoccertalk.com/watch-on-us-tv/how-to-watch-bayern-munich-vs-arsenal-match-in-the-usa-live-stream-and-tv-for-2026-27-uefa-champions-league/)
  - `NEUTRAL` | the4thofficial.net | relevance 0.529 | credibility 0.40 | P(entail) 0.003 | P(contra) 0.005 | [White, Lewis-Skelly And Eze To Start | 4-2-3-1 Arsenal Predicted Lineup Vs Everton](https://the4thofficial.net/2026/10/white-lewis-skelly-and-eze-to-start-4-2-3-1-arsenal-predicted-lineup-vs-everton/)
  - `NEUTRAL` | playmakerstats.com | relevance 0.526 | credibility 0.40 | P(entail) 0.007 | P(contra) 0.026 | [Kane closes in on Cristiano Ronaldo's Champions League rcord](https://www.playmakerstats.com/news/kane-closes-in-on-cristiano-ronaldo-s-champions-league-rcord/1235859)

### 6. US consumer prices rose 3.7 percent in the year to September 2026, the Labor Department reported on October 14.

- Expected: `SUPPORTED`
- Predicted: `UNVERIFIED`
- Category: NLI too strict
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.000
- Total weight: 0.000
- Latency: 6080 ms
- Explanation: UNVERIFIED: no relevant article supports or refutes the claim; neutral evidence is treated as no information; 4 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | aljazeera.com | relevance 0.790 | credibility 0.70 | P(entail) 0.001 | P(contra) 0.002 | [US inflation rises before midterm elections, fuelling rate hike fears](https://www.aljazeera.com/economy/2026/10/14/us-inflation-rises-before-midterm-elections-fuelling-rate-hike-fears)
  - `NEUTRAL` | thehindubusinessline.com | relevance 0.667 | credibility 0.40 | P(entail) 0.003 | P(contra) 0.006 | [US inflation hits 3.7%, raising odds of Fed rate hike in October](https://www.thehindubusinessline.com/news/world/us-inflation-hits-37-raising-odds-of-fed-rate-hike-in-october/article71584531.ece)
  - `NEUTRAL` | naturalnews.com | relevance 0.596 | credibility 0.40 | P(entail) 0.006 | P(contra) 0.013 | [Trump Faces Second Inflation Report as Midterms Approach](https://www.naturalnews.com/2026-10-18-trump-faces-second-inflation-report-midterms-approach.html)
  - `NEUTRAL` | dailymail.com | relevance 0.628 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.002 | [Trump's midterm nightmare deepens as inflation surges and gas prices skyrocket across America](https://www.dailymail.com/yourmoney/article-16208643/trump-midterms-inflation-september-cpi-gas-prices-america.html)
  - `NEUTRAL` | theeconomiccollapseblog.com | relevance 0.398 | credibility 0.40 | P(entail) 0.005 | P(contra) 0.002 | [44% Of Americans Have Trouble Paying For Basic Necessities And The Lifetime Cost Of The American Dream Has Risen To $5.3 Million](http://theeconomiccollapseblog.com/44-of-americans-have-trouble-paying-for-basic-necessities-and-the-lifetime-cost-of-the-american-dream-has-risen-to-5-3-million/)
  - `NEUTRAL` | finance.yahoo.com | relevance 0.408 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.014 | [U.S. job growth falls short of expectations as unemployment rises to 4.2%](https://finance.yahoo.com/economy/live/september-jobs-report-live-updates-labor-market-adds-29000-124648123.html)

### 7. Robert Hall and Michael Woodford won the 2026 Nobel economics prize for their work on the economics of climate change.

- Expected: `REFUTED`
- Predicted: `UNVERIFIED`
- Category: compound claim
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.000
- Total weight: 0.000
- Latency: 4857 ms
- Explanation: UNVERIFIED: no relevant article supports or refutes the claim; neutral evidence is treated as no information; 6 relevant neutral article(s) treated as no information.
- Evidence:
  - `NEUTRAL` | newstalkflorida.com | relevance 0.835 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.003 | [Unlocking Macroeconomics: Why Robert E. Hall And Michael Woodford Won The 2026 Nobel Prize In Economics](https://www.newstalkflorida.com/featured/noble-peace-prize-winners-and-why-they-won/)
  - `NEUTRAL` | economictimes.indiatimes.com | relevance 0.885 | credibility 0.70 | P(entail) 0.001 | P(contra) 0.000 | [Nobel Prize in Economics 2026 awarded to Robert E. Hall and Michael Woodford](https://economictimes.indiatimes.com/news/international/world-news/nobel-prize-in-economics-2026-awarded-to-robert-e-hall-and-michaelwoodford-sveriges-riksbank-prize-in-economic-sciences/articleshow/134888651.cms)
  - `NEUTRAL` | yahoo.com | relevance 0.866 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.000 | [Americans Robert E. Hall and Michael Woodford win 2026 Nobel prize in economics](https://www.yahoo.com/news/world/articles/nobel-memorial-prize-economics-awarded-054014824.html)
  - `NEUTRAL` | biztoc.com | relevance 0.857 | credibility 0.40 | P(entail) 0.011 | P(contra) 0.380 | [Hall, Woodford win Nobel Prize in Economics](https://biztoc.com/x/ce860509c6aa19d5)
  - `NEUTRAL` | aljazeera.com | relevance 0.852 | credibility 0.70 | P(entail) 0.003 | P(contra) 0.002 | [Nobel economics prize awarded to Robert Hall and Michael ⁠Woodford](https://www.aljazeera.com/news/2026/10/12/nobel-economics-prize-awarded-to-robert-hall-and-michael-%e2%81%a0woodford)
  - `NEUTRAL` | nypost.com | relevance 0.837 | credibility 0.40 | P(entail) 0.011 | P(contra) 0.021 | [Robert E. Hall and Michael Woodford win Nobel memorial prize in economics](https://nypost.com/2026/10/12/world-news/robert-e-hall-and-michael-woodford-win-nobel-memorial-prize-in-economics/)

### 8. Tesla reported third-quarter 2026 revenue of $38 billion, up 35 percent from a year earlier.

- Expected: `REFUTED`
- Predicted: `UNVERIFIED`
- Category: retrieval miss
- Confidence: 0.000
- Support score: 0.000
- Refute score: 0.459
- Total weight: 0.473
- Latency: 5291 ms
- Explanation: UNVERIFIED: weighted relevant evidence is too low (0.473 < 0.500); top refuting: thenextweb.com, macrumors.com; 1 relevant neutral article(s) treated as no information.
- Evidence:
  - `REFUTES` | thenextweb.com | relevance 0.640 | credibility 0.40 | P(entail) 0.000 | P(contra) 0.982 | [Anthropic tells investors Q3 revenue reached $16.8bn, FT reports](https://thenextweb.com/news/anthropic-q3-revenue-16-8bn-ipo)
  - `NEUTRAL` | techpowerup.com | relevance 0.543 | credibility 0.40 | P(entail) 0.003 | P(contra) 0.489 | [(PR) LG Display Reports Third Quarter 2026 Results](https://www.techpowerup.com/354033/lg-display-reports-third-quarter-2026-results)
  - `NEUTRAL` | bitcoinfoundation.org | relevance 0.489 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.316 | [What Is Robinhood Chain? How Its $51M Debut Beat Ethereum](https://bitcoinfoundation.org/news/companies/what-is-robinhood-chain-how-its-51m-debut-beat-ethereum/)
  - `REFUTES` | macrumors.com | relevance 0.542 | credibility 0.40 | P(entail) 0.000 | P(contra) 0.957 | [iPhone Takes Bigger Slice of a Shrinking Smartphone Market](https://www.macrumors.com/2026/10/13/iphone-takes-bigger-slice-of-shrinking-market/)
  - `NEUTRAL` | m.economictimes.com | relevance 0.453 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.003 | [Meta Q3 Earnings: AI spending, revenue beat & new stock class](https://m.economictimes.com/markets/us-stocks/wall-street-guide/meta-q3-earnings-ai-spending-revenue-beat-amp-new-stock-class/meta-beats-q3-estimates-as-ai-spending-surges/slideshow/135112550.cms)
  - `NEUTRAL` | prnewswire.com | relevance 0.433 | credibility 0.40 | P(entail) 0.001 | P(contra) 0.006 | [M&T Bank Corporation (NYSE:MTB) announces third quarter 2026 results](https://www.prnewswire.com/news-releases/mt-bank-corporation-nysemtb-announces-third-quarter-2026-results-302910052.html)

## Credibility ablation

All evidence credibility scores were set to 1.0 and the decision was re-run with the same NLI probabilities. Retrieval, ranking, and NLI were not re-run.

| Metric | Real tiers | All weights 1.0 |
| --- | ---: | ---: |
| Exact accuracy | 0.600 (12/20) | 0.650 (13/20) |
| Wrong-verdict rate | 0.071 (1/14) | 0.071 (1/14) |
| Coverage | 0.500 (7/14) | 0.571 (8/14) |

| Claim | Expected | Real tiers | All weights 1.0 | Changed |
| --- | --- | --- | --- | --- |
| Max Verstappen won the 2026 Singapore Grand Prix, finishing ahead of Charles Leclerc. | SUPPORTED | UNVERIFIED | UNVERIFIED | no |
| West Indies beat India by six wickets in the second T20I in Ranchi on October 9, 2026. | SUPPORTED | REFUTED | REFUTED | no |
| The Reserve Bank of India raised its repo rate by 25 basis points to 5.50 percent on... | SUPPORTED | UNVERIFIED | UNVERIFIED | no |
| TSMC reported third-quarter 2026 net profit of NT$763.06 billion, up 68.7 percent fro... | SUPPORTED | UNVERIFIED | UNVERIFIED | no |
| Bayern Munich and Arsenal drew 2-2 in a Champions League match on October 21, 2026. | SUPPORTED | UNVERIFIED | UNVERIFIED | no |
| US consumer prices rose 3.7 percent in the year to September 2026, the Labor Departme... | SUPPORTED | UNVERIFIED | UNVERIFIED | no |
| Actor Tom Skerritt, known for Top Gun and Alien, died at age 93. | SUPPORTED | SUPPORTED | SUPPORTED | no |
| Robert Hall and Michael Woodford won the 2026 Nobel economics prize for their work on... | REFUTED | UNVERIFIED | UNVERIFIED | no |
| India won the five-match T20I series against West Indies 3-1. | REFUTED | REFUTED | REFUTED | no |
| Rafael Jodar beat Alejandro Davidovich Fokina in the final of the 2026 Shanghai Masters. | REFUTED | REFUTED | REFUTED | no |
| Tesla reported third-quarter 2026 revenue of $38 billion, up 35 percent from a year e... | REFUTED | UNVERIFIED | REFUTED | yes |
| OpenAI launched the Muse Wonder family of AI models on October 12, 2026. | REFUTED | REFUTED | REFUTED | no |
| On October 16, 2026, President Trump signed an executive order putting new Graham Act... | REFUTED | REFUTED | REFUTED | no |
| Manchester United beat Leeds United 2-0 at Elland Road on October 18, 2026. | REFUTED | REFUTED | REFUTED | no |
| Anne Carson would have won the Nobel Prize in Literature ten years earlier if the com... | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| Max Verstappen will win the 2027 Formula 1 drivers' championship. | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| India will win the 2027 Cricket World Cup. | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| A shopkeeper in Pune sold exactly 40 umbrellas on Tuesday, October 13, 2026. | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| A cafe in Kochi served 212 cups of filter coffee on the morning of October 15, 2026. | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |
| The mayor of a small town in Wisconsin had pancakes for breakfast on October 12, 2026. | UNVERIFIED | UNVERIFIED | UNVERIFIED | no |

