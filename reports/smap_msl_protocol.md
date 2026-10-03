# Preregistered protocol: OES32 vs simple baselines on NASA SMAP/MSL (real public telemetry)

**Status: LOCKED on 2026-10-03, before any method was scored on the test split.**
The machine-readable version is [`smap_msl_protocol.json`](smap_msl_protocol.json); the evaluation
(`oes-resilience smap-msl evaluate`) reads its parameters from that file and records its SHA-256 in
the results. The lock is the git commit that adds these two files, pushed with CI green before the
evaluation was run. Any later change to the protocol, the code path or the data must be disclosed
next to the results.

## Data

- **Dataset.** NASA Soil Moisture Active Passive (SMAP) satellite and Mars Science Laboratory (MSL,
  Curiosity rover) telemetry with expert-labelled anomalies, released with K. Hundman, V. Constantinou,
  C. Laporte, I. Colwell and T. Soderstrom, "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
  Dynamic Thresholding", *KDD '18*, pp. 387–395, [doi:10.1145/3219819.3219845](https://doi.org/10.1145/3219819.3219845)
  ([arXiv:1802.04431](https://arxiv.org/abs/1802.04431)). Code and labels: <https://github.com/khundman/telemanom>.
- **Licence and terms.** No explicit open-data licence is attached to the data files. The telemanom
  repository asks users to cite the paper; its README states Apache-2.0 while its `LICENSE.txt` is a
  Caltech/JPL BSD-style copyright notice (both concern the code). The Kaggle mirror that the telemanom
  README now links to (`patrickfleith/nasa-anomaly-detection-dataset-smap-msl`) labels the files
  "Data files © Original Authors". We therefore **do not redistribute any raw data**: the fetch command
  downloads it from the public source into a cache outside the repository, and only derived, aggregate
  results, per-channel metrics and file hashes are committed. Users must obtain the data themselves
  and cite Hundman et al. (2018).
- **Source.** The original S3 bucket (`https://s3-us-west-2.amazonaws.com/telemanom/data.zip`) returned
  HTTP 403 on 2026-10-03; the fetch command uses the Kaggle mirror's public download endpoint first and
  keeps S3 as a fallback. The archive is re-packed by the mirror, so verification is per extracted file:
  [`smap_msl_data.sha256`](smap_msl_data.sha256) lists the SHA-256 of all 82 train and 82 test `.npy`
  files and `labeled_anomalies.csv` (165 files; the CSV is byte-identical to the one in the telemanom
  GitHub repository, SHA-256 `057ce2d6…7539`). Archive SHA-256 at lock time: `6084d3ee…3733`.
- **Known properties of the data, stated before evaluation.**
  - Values are pre-scaled to (−1, 1) using the min/max of the **test** set (telemanom README). This is
    inherent to the published data and affects every method equally; it is a leak we cannot remove.
  - `labeled_anomalies.csv` has 82 rows: 55 SMAP rows (54 unique channels; `P-2` appears twice with
    overlapping windows) and 27 MSL channels; 69 SMAP and 36 MSL labelled windows. `T-10` has data files
    but no labels and is not evaluated.
  - 9 SMAP channels (D-2, G-2, D-7, P-4, D-8, D-9, D-12, B-1, D-13) and 5 MSL channels (M-6, S-2, T-5,
    C-2, D-14) have constant telemetry in train, so the train std is 0 and any deviation in test alarms
    for every method. They stay in the primary analysis; a sensitivity analysis excludes them.

## What was looked at before the lock

Only: the label CSV (needed to define events), test-array **shapes** (to check them against
`num_values`; all 82 match), and the **train** split (lengths, constant channels, and the train-only
`train-diagnostics` command that reports each method's calibrated threshold and achieved train alarm
rate). No method was scored on any test sample and no test value was inspected.

## Methods (same input for all)

Per channel, only column 0 (the telemetry value) is used; the one-hot command columns are ignored by
every method. Input `z = (x − mean_train) / std_train` (ddof 1, floor 1e-9). Train and test are scored
as separate segments.

| method | score |
|---|---|
| `oes32` (candidate) | `0.45·max|z| + 0.35·RMS(z) + 0.20·mean|z|` over the trailing 32-sample window (the reference 32-channel block becomes 32 consecutive samples; partial mask-aware windows for the first 31 samples) |
| `maxabs` | `max|z|` over the same window |
| `zscore` | `|z_t|` (simple pointwise z-score) |
| `ewma` | built-in EWMA recursion on `z`, λ = 0.2, score `|e_t| / sqrt(λ/(2−λ))` |
| `cusum` | built-in two-sided CUSUM on `z`, k = 0.5 |

All parameters are the package defaults; nothing is tuned.

## Thresholds (train only)

`calibrate_threshold` (the rule used by `compare`, `stress` and `replay`) on the channel's train scores
with unit = one sample: the smallest threshold reached by at most `floor(target × n_train)` train samples.
Primary target **0.01** (the project's shared ~1% FP budget); secondary targets 0.001 and 0 (just above
the train maximum). Alarm when `score >= threshold`. No test label is used for any choice. Because
`maxabs` scores are piecewise constant over windows, ties can make its achieved train alarm rate well
below the target (seen in the train diagnostics); this follows from the shared rule and is not adjusted.

## Metrics (fixed in advance)

- **Primary:** pooled event-level F1 per dataset at target 0.01. An event (merged labelled window,
  inclusive) is **detected if at least one alarm sample falls inside it**. A **false alarm** is a maximal
  run of consecutive alarm samples that touches no labelled window. Precision = TP / (TP + FP runs),
  recall = TP / events; counts are summed over the dataset's channels.
- **Secondary:** event precision and recall; point-adjusted precision/recall/F1 (reported only as
  secondary: point adjustment is known to inflate scores, Kim et al., AAAI 2022, [doi:10.1609/aaai.v36i7.20680](https://doi.org/10.1609/aaai.v36i7.20680)); detection latency
  in samples (first alarm in the window minus its start; detected events only); false alarms per 1000
  test samples; test alarm rate on unlabelled samples; mean per-channel F1; recall by anomaly class;
  per-channel results.
- SMAP and MSL are always reported separately.

## Uncertainty and comparisons

- 95% percentile bootstrap CIs over channels (10,000 resamples, seed 42).
- For each baseline: paired channel bootstrap of the pooled F1 difference (oes32 − baseline) and a
  two-sided Wilcoxon signed-rank test over channels on per-channel event F1; Holm correction over the 8
  primary comparisons (4 baselines × 2 datasets).

## Success criteria (pre-stated)

- Per comparison: **oes32 better** if the paired-bootstrap 95% CI of ΔF1 lies entirely above 0 *and*
  Holm-adjusted Wilcoxon p < 0.05; **oes32 worse** if the CI lies entirely below 0 *and* Holm p < 0.05;
  otherwise **no significant difference**.
- Headline: the evaluation supports OES32 on real data only if, at target 0.01 on all channels, OES32 is
  better than every baseline on at least one dataset and worse than none on either. Anything else is
  reported as the criterion not being met. Latency, false alarms, point-adjusted F1, secondary targets
  and the sensitivity subset are descriptive only.

## Reproduce

```bash
oes-resilience smap-msl fetch --data-dir ~/.cache/oes-resilience/smap_msl     # download + SHA-256 verification
oes-resilience smap-msl evaluate --data-dir ~/.cache/oes-resilience/smap_msl \
    --protocol reports/smap_msl_protocol.json --out-dir reports/smap_msl_results --verify
```
