# SMAP/MSL baseline comparisons and confidence intervals

## Scope

This document summarizes the locked primary comparison at `target_fp = 0.01`, using all channels. SMAP and MSL are reported separately, as required by `reports/smap_msl_protocol.json`.

- Candidate: `oes32`
- Baselines: `maxabs`, `zscore`, `ewma`, and `cusum`
- Primary metric: pooled event-level F1
- Uncertainty: 10,000-resample paired channel bootstrap, 95% percentile intervals, seed 42
- Significance: paired channel bootstrap plus two-sided Wilcoxon signed-rank test with Holm correction across the eight primary comparisons
- Source artifacts: `smap_msl_summary.csv` and `smap_msl_comparisons.csv`

## Headline verdict

**The preregistered headline success criterion was not met.**

OES-32 significantly outperforms EWMA on SMAP, but it does not significantly outperform maxabs or CUSUM on either dataset. Maxabs and CUSUM also have higher point estimates than OES-32 on both SMAP and MSL. Therefore, OES-32 is not supported as better than every baseline on at least one dataset while worse than none on either dataset.

## Primary event-F1 results

| Dataset | Method | Event F1 | 95% CI |
|---|---|---:|---:|
| SMAP | oes32 | 0.378601 | [0.273556, 0.522167] |
| SMAP | maxabs | 0.475410 | [0.328767, 0.645841] |
| SMAP | zscore | 0.060887 | [0.032237, 0.155983] |
| SMAP | ewma | 0.086957 | [0.060642, 0.131869] |
| SMAP | cusum | 0.169252 | [0.122105, 0.256413] |
| MSL | oes32 | 0.434783 | [0.333291, 0.575758] |
| MSL | maxabs | 0.500000 | [0.350868, 0.657143] |
| MSL | zscore | 0.248756 | [0.177911, 0.389381] |
| MSL | ewma | 0.223176 | [0.126982, 0.423358] |
| MSL | cusum | 0.514286 | [0.387097, 0.681319] |

## Paired comparisons at the primary target

Positive `delta_event_f1` favors OES-32. A comparison is significant only when the paired-bootstrap 95% CI excludes zero and the Holm-adjusted Wilcoxon p-value is below 0.05.

| Dataset | Baseline | OES-32 minus baseline | 95% paired-bootstrap CI | Holm-adjusted Wilcoxon p | Verdict |
|---|---|---:|---:|---:|---|
| SMAP | maxabs | -0.096809 | [-0.242713, 0.038642] | 0.329945 | No significant difference |
| SMAP | zscore | 0.317714 | [0.200296, 0.447927] | 0.928614 | No significant difference |
| SMAP | ewma | 0.291644 | [0.192098, 0.425666] | 0.021368 | OES-32 better |
| SMAP | cusum | 0.209349 | [0.113143, 0.325648] | 1.000000 | No significant difference |
| MSL | maxabs | -0.065217 | [-0.163052, 0.042137] | 1.000000 | No significant difference |
| MSL | zscore | 0.186027 | [0.039972, 0.318276] | 0.938672 | No significant difference |
| MSL | ewma | 0.211607 | [0.060800, 0.368370] | 0.928614 | No significant difference |
| MSL | cusum | -0.079503 | [-0.224497, 0.063691] | 1.000000 | No significant difference |

## Interpretation

- OES-32 has a substantial point-estimate advantage over zscore and EWMA, but only the SMAP-versus-EWMA comparison satisfies both preregistered significance conditions.
- OES-32 does not establish an advantage over maxabs or CUSUM.
- The negative result is not evidence that OES-32 is universally worse; it is evidence that the locked headline claim of broad baseline superiority is not supported.
- Latency, false alarms per 1,000 test samples, point-adjusted F1, secondary false-positive targets, and the constant-train sensitivity subset are descriptive only and do not alter this verdict.

## Reproducibility

The full paired-comparison table, including unadjusted p-values, test statistics, non-zero channel counts, and per-comparison verdicts, is in `smap_msl_comparisons.csv`. The full metric table, including confidence intervals for absolute event F1, is in `smap_msl_summary.csv`.
