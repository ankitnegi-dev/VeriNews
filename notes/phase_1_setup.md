# Phase 1 - Setup and baseline

## What we built

A Python 3.12 virtual environment with pinned dependencies, a Git repo on branch
`phase-1-setup`, and one central `config.py` for every path, model name and threshold.
`scripts/baseline_isot.py` trains a TF-IDF + logistic-regression classifier on ISOT twice:
once on raw text and once with the leading wire-service dateline removed. `scripts/smoke_nli.py`
loads the fixed NLI model and checks one entailment, one neutral and one contradiction pair.

## ISOT baseline

Raw files: 23,481 Fake and 21,417 True articles. After dropping 630 Fake and 1 True article
with an empty body, 44,267 rows remain. Stratified 80/20 split, `random_state=42`; 8,854 test rows.

| Variant | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Confusion (actual Fake/True x predicted Fake/True) |
|---|---:|---:|---:|---:|---|
| As-is | 0.9864 | 0.9863 | 0.9866 | 0.9864 | [[4494, 77], [43, 4240]] |
| Dateline stripped | 0.9794 | 0.9793 | 0.9796 | 0.9794 | [[4453, 118], [64, 4219]] |

**Why the difference matters.** 19,451 of 21,416 usable True articles (90.8%) start with a
dateline such as `WASHINGTON (Reuters) - `; **zero** Fake articles do. The as-is model can
use that shortcut, so its 98.6% is not evidence that it detects truth. Removing the dateline
costs 0.70 accuracy points, which shows the shortcut was real but not the only signal: ISOT
Fake and True articles also differ in topic and writing style. Even 97.9% is a style score,
not a fact-checking score. Exact values are in `results/baseline_metrics.json`.

Caveat: 5,628 duplicate article bodies remain, so some near-duplicates can fall on both sides
of the split and make the score slightly optimistic.

## NLI smoke test

Model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Raw `id2label`:
`{0: entailment, 1: neutral, 2: contradiction}`. CPU, 8 threads.

| Premise (evidence) | Hypothesis (claim) | P(entail) | P(neutral) | P(contra) | Latency |
|---|---|---:|---:|---:|---:|
| Eiffel Tower is in Paris, France. | Eiffel Tower is in Paris. | 0.998 | 0.001 | 0.000 | 2.45 s |
| Eiffel Tower is in Paris, France. | Eiffel Tower was completed in 1889. | 0.004 | 0.996 | 0.001 | 2.06 s |
| Eiffel Tower is in Paris, France. | Eiffel Tower is in Berlin, Germany. | 0.000 | 0.000 | 1.000 | 2.08 s |

3/3 correct; mean 2.20 s/pair, max 2.45 s. First load (including download) took 79.9 s;
cached load took 27.9 s. Under the 3 s budget, so the fallback model is **not** used.

## FEVER inspection (inspection only; nothing built on it)

| File | Rows | Columns | Notes |
|---|---:|---|---|
| `data/raw/fever/train.jsonl` | 145,449 | `id`, `verifiable`, `label`, `claim`, `evidence` | Original JSONL |
| `data/raw/fever/fever.csv` | 145,449 | same five | CSV conversion; `evidence` is stringified |
| `data/raw/fever/1.py` | - | - | Conversion helper, not data |

Labels: SUPPORTS 80,035 (55.0%), REFUTES 29,775 (20.5%), NOT ENOUGH INFO 35,639 (24.5%).
Verifiable: 109,810; not verifiable: 35,639.

**Evidence contains references, not text.** Each reference is
`[page_id, sentence_id, page_title, sentence_number]`, for example
`[92206, 104971, "Nikolaj_Coster-Waldau", 7]`. There is no evidence-sentence text.
`NOT ENOUGH INFO` rows contain `[page_id, null, null, null]`. Therefore the optional
Phase 5 NLI-on-FEVER check is **not possible from these files alone**; it would need the
Wikipedia sentence text (or the FEVER wiki dump). We did not build anything on FEVER.

## Concepts, simply

- **TF-IDF** turns text into numbers: a word matters more when it is frequent in one article
  but rare across all articles. Analogy: a word that appears in every newspaper is not a clue.
- **Logistic regression** learns a weighted boundary between the two classes. It is a simple,
  fast classifier, not a language model.
- **Source-tag leakage** is shortcut learning: the model learns "Reuters dateline = True"
  instead of learning what makes a claim true. Analogy: judging a book by its cover stamp.
- **NLI (natural language inference)** asks whether a premise supports, contradicts, or is
  neutral about a hypothesis. Here premise = evidence snippet, hypothesis = claim.
- We do **not** fine-tune because the deadline is short, a zero-shot NLI model already works,
  and fine-tuning would need labelled claim/evidence pairs and more compute.

## How the code works

1. `validate_isot_files()` stops immediately if a CSV is missing or 0 bytes.
2. `load_isot()` reads `title` and `text`, labels Fake=0 / True=1, and drops empty bodies.
3. `strip_source_tag()` removes one leading `CITY (Reuters|AP|AFP) - ` tag with a regex.
4. The same train/test indices are used for both variants; TF-IDF is fitted on train only,
   then logistic regression predicts the test rows.
5. `evaluate_variant()` writes accuracy, macro/weighted precision-recall-F1, per-class scores,
   a confusion matrix and the sklearn report to `results/baseline_metrics.json`.
6. `smoke_nli.py` maps label names from `model.config.id2label` (never assumes index order),
   runs one pair at a time, applies softmax and prints probabilities plus latency.

## Why these choices

- TF-IDF + logistic regression is the simplest credible baseline and runs in seconds.
- We use the `text` field only; titles would add another shortcut and blur the dateline test.
- Empty bodies are dropped because a zero-vector article teaches the classifier nothing.
- We keep the base DeBERTa model because it meets the 3 s budget; the small fallback would
  trade accuracy for speed we do not currently need.
- Trade-off: the baseline measures dataset style, not factual truth. It is a reference point,
  not the project's verification engine.

## Likely viva questions

1. **Why build a baseline?** To get a cheap reference score and expose dataset shortcuts
   before building the real pipeline.
2. **Why did accuracy fall after stripping datelines?** The model had been using the Reuters
   tag as a strong hint; removing it forces more reliance on wording and topic.
3. **Does 98% mean the model can detect fake news?** No. ISOT is a style/topic dataset, and
   the split still contains duplicates; the score does not transfer to live claims.
4. **What do the three NLI outputs mean?** Entailment = evidence supports the claim,
   contradiction = evidence conflicts with it, neutral = evidence does not decide it.
5. **Why not fine-tune the NLI model?** No labelled in-domain data, limited CPU time, and a
   zero-shot model already passes the sanity check.

## Check yourself

1. Why can a classifier score 98.6% on ISOT without learning whether a claim is true?
2. If `p_contra = 0.82`, what stance should the NLI stage assign, and why?
3. Why can we not run the optional FEVER NLI evaluation directly from `train.jsonl`?
