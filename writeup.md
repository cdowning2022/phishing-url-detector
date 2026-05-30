# Methodology and Lessons Learned

This document goes deeper than the README into the decisions made during the
build of this project, the bugs encountered, and what I'd do differently.

## The project in one sentence

A Random Forest classifier that takes a raw URL string, extracts ~20 structural
features (length, character ratios, subdomain count, HTTPS presence, etc.), and
predicts phishing or legitimate with a confidence score.

## Dataset

The [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
from the UCI Machine Learning Repository.

- 235,795 URLs total
- 54 pre-extracted features per URL
- Roughly 57% legitimate, 43% phishing
- Labels: `1 = legitimate`, `0 = phishing`

Legitimate URLs were sourced from Tranco (top-sites list); phishing URLs from
PhishTank and OpenPhish. This sourcing choice matters — see "limitations" below.

## Modeling decisions

### Why Random Forest over Logistic Regression

Both were trained and compared on the same train/test split (`random_state=42`,
20% test, stratified). Results were essentially tied on accuracy, but Random
Forest was chosen for:

1. **Better explainability via feature importances.** Trees naturally rank
   features by how much they contribute to splits, giving us the chart in the
   README. Logistic Regression gives coefficients, but they're harder to read
   when features are on different scales.
2. **No feature scaling required.** Logistic Regression needs `StandardScaler`;
   Random Forest doesn't. Fewer moving parts in production.
3. **Better-calibrated probabilities.** `predict_proba` outputs from Random
   Forest are more useful for the CLI's confidence display.

### Why not deep learning

The dataset is tabular (54 numeric features per row). Tabular data is the one
domain where classical ML (gradient-boosted trees, random forests) typically
matches or beats neural nets, especially at this scale. There's no spatial or
sequential structure in the features that would benefit from convolution or
recurrence.

### Hyperparameters

`n_estimators=100`, `max_depth=None`, `random_state=42`, `n_jobs=-1`. No
hyperparameter tuning was done because the model already saturates near the
top of what the dataset allows; tuning would mostly fit noise.

## The bug that taught me the most

After the first end-to-end test of the CLI, every URL — `github.com`,
`google.com`, an obviously-malicious paypal lookalike, anything — got the same
verdict: phishing, 96% confidence.

The cause was **train-serve skew**. The training pipeline was using the
dataset's 54 pre-computed feature columns. The CLI was extracting features
from raw URL strings, but could only compute ~20 of them — the rest (page
content, WHOIS data, certificate validity) require lookups the CLI doesn't do.
So at prediction time the model received 20 real values plus 34 zeros, and the
"mostly-zero" pattern got confidently classified as phishing.

Two fixes were needed:

1. **Align the feature set.** The model should only ask for features that the
   CLI can actually produce. The training script now derives the feature list
   by calling `extract_features("https://example.com").keys()` — automatically
   keeping training and prediction in sync.
2. **Align the feature *values*.** Even after fixing the feature set, the
   dataset's `NoOfOtherSpecialCharsInURL` (and a few others) were computed
   slightly differently than my extractor. So training on the dataset's
   numbers and predicting with mine still produced inconsistent results. The
   fix was to re-extract features from raw URLs at training time using the
   same `extract_features()` function the CLI uses. Now training and
   prediction are guaranteed to see the exact same feature values for a
   given URL.

This was the most valuable debugging experience of the build. Train-serve
skew is a known failure mode in ML systems, and now I've personally hit it.

## Leakage check

During exploratory analysis, `URLSimilarityIndex` showed near-perfect
correlation (|r| ≈ 0.99) with the label. This is a classic sign of
**target leakage** — a feature whose value was computed using information
from the label itself.

Comparing models trained with and without it:

| Model | Accuracy |
|---|---|
| With URLSimilarityIndex | 99.99% |
| Without (honest baseline) | 99.92% |

The gap was only 0.07 percentage points, meaning the rest of the feature set
was already enough to (over-)separate the classes. URLSimilarityIndex was
excluded from the final model on principle anyway — even if it weren't
leakage, it's not a feature available at prediction time.

## Limitations

### Dataset separability

Test accuracy in the high-90s is suspicious for any real classification problem.
The likely cause: phishing URLs were collected from automated phishing feeds
(short-lived, randomized, structurally distinctive), while legitimate URLs
came from a top-sites list (mature, well-formed). The two distributions are
very different in ways that aren't necessarily about phishing — they're about
how each kind of URL is generated.

A model trained on this data will probably perform worse on fresh phishing URLs
in the wild, because real attackers actively try to mimic legitimate URL
structure. The benchmark accuracy is real but optimistic.

### String-only features

The CLI doesn't fetch the URL or do any network lookups. This is by design —
visiting phishing URLs is risky and out of scope — but it means features like
domain age, certificate trust, page content, and DNS records are unavailable.
A v2 could add an optional `--live-fetch` flag that pulls these features with
appropriate safety measures.

### No defensive layering

A production phishing detector would combine an ML model with:
- Allowlists for known-good domains
- Blocklists from threat intelligence feeds
- Reputation services (Google Safe Browsing, etc.)
- User reporting and feedback loops

This project is the ML component only. Treating its output as a sole signal
in production would be unwise.

## What I'd do differently next time

- **Start with the production interface.** I built the training pipeline first
  and bolted the CLI on after. Building the CLI's feature extractor first
  would have caught the train-serve skew on day one.
- **Use a noisier dataset.** PhiUSIIL is too clean to expose modeling weaknesses.
  The Mendeley web-page phishing dataset is messier and would have forced
  harder design choices.
- **Add cross-validation.** Single train/test split is fine for a quick
  baseline; k-fold would give a more honest accuracy estimate.
- **Calibrate probabilities.** The confidence scores from `predict_proba` are
  raw probabilities and not necessarily well-calibrated. A calibration step
  (Platt scaling or isotonic regression) would make them more trustworthy.

## References

- PhiUSIIL Dataset: Prasad, A., & Chandra, S. (2024). PhiUSIIL Phishing URL Dataset.
  UCI Machine Learning Repository. https://doi.org/10.24432/C5N625
- scikit-learn documentation: https://scikit-learn.org/
