# SMAP/MSL locked evaluation: final report

## Final verdict

**The preregistered headline success criterion was not met.**

At the primary false-positive target of 0.01 on all channels, OES-32 significantly outperformed EWMA on SMAP, but it did not significantly outperform maxabs or CUSUM on either SMAP or MSL. Maxabs and CUSUM also had higher point-estimate event F1 than OES-32 on both datasets. The evaluation therefore does not support a claim that OES-32 is superior to every preregistered baseline.

## Locked evaluation

- Protocol: `reports/smap_msl_protocol.json`, locked before test-split scoring on 2026-10-03.
- Datasets: NASA SMAP and MSL telemetry anomaly datasets, analyzed separately.
- Candidate: `oes32`.
- Baselines: `maxabs`, `zscore`, `ewma`, and `cusum`.
- Primary metric: pooled event-level F1.
- Primary target: false-positive rate 0.01.
- Uncertainty: 10,000-resample paired channel bootstrap, 95% percentile confidence intervals, seed 42.
- Significance: paired channel bootstrap plus two-sided Wilcoxon signed-rank test with Holm correction across eight primary comparisons.

## Primary results

| Dataset | Method | Event F1 | 95% CI |
|---|---|---:|---:|
| SMAP | OES-32 | 0.378601 | [0.273556, 0.522167] |
| SMAP | maxabs | 0.475410 | [0.328767, 0.645841] |
| SMAP | zscore | 0.060887 | [0.032237, 0.155983] |
| SMAP | ewma | 0.086957 | [0.060642, 0.131869] |
| SMAP | cusum | 0.169252 | [0.122105, 0.256413] |
| MSL | OES-32 | 0.434783 | [0.333291, 0.575758] |
| MSL | maxabs | 0.500000 | [0.350868, 0.657143] |
| MSL | zscore | 0.248756 | [0.177911, 0.389381] |
| MSL | ewma | 0.223176 | [0.126982, 0.423358] |
| MSL | cusum | 0.514286 | [0.387097, 0.681319] |

## Key paired comparisons

Positive differences favor OES-32.

| Dataset | Baseline | Difference | 95% paired-bootstrap CI | Holm-adjusted Wilcoxon p | Verdict |
|---|---|---:|---:|---:|---|
| SMAP | maxabs | -0.096809 | [-0.242713, 0.038642] | 0.329945 | No significant difference |
| SMAP | zscore | 0.317714 | [0.200296, 0.447927] | 0.928614 | No significant difference |
| SMAP | ewma | 0.291644 | [0.192098, 0.425666] | 0.021368 | OES-32 better |
| SMAP | cusum | 0.209349 | [0.113143, 0.325648] | 1.000000 | No significant difference |
| MSL | maxabs | -0.065217 | [-0.163052, 0.042137] | 1.000000 | No significant difference |
| MSL | zscore | 0.186027 | [0.039972, 0.318276] | 0.938672 | No significant difference |
| MSL | ewma | 0.211607 | [0.060800, 0.368370] | 0.928614 | No significant difference |
| MSL | cusum | -0.079503 | [-0.224497, 0.063691] | 1.000000 | No significant difference |

## Ablation status

No preregistered ablations exist. Residual-normalization, latch-threshold, channel-grouping, and calibration variants were not implemented or run. They are recorded as potential exploratory work in `reports/smap_msl_ablations/ABLATION_PLAN.md` and cannot alter this final verdict.

## Limitations

- The evaluation covers the NASA SMAP/MSL telemetry dataset only; it does not establish performance on other domains, streaming systems, field deployments, quantum hardware, or medical settings.
- Event-level F1 is sensitive to the dataset’s limited number of labeled events, especially for MSL.
- Confidence intervals are channel-bootstrap intervals and reflect sampling variability across channels, not every possible source of implementation or dataset variation.
- Secondary metrics, secondary false-positive targets, latency, false alarms per 1,000 samples, point-adjusted F1, and the constant-train sensitivity subset are descriptive only.

## Artifact provenance

- Result metrics: `smap_msl_summary.csv`
- Paired comparisons: `smap_msl_comparisons.csv`
- Full machine-readable results: `smap_msl_results.json`
- Per-channel results: `smap_msl_per_channel.csv`
- Event-F1 figure: `smap_msl_event_f1.svg`
- Result checksums: `SHA256SUMS`
- Dataset checksums: `../smap_msl_data.sha256`
- Run metadata: `smap_msl_run_metadata.json`
- Verification record: `VERIFICATION.md`
- Baseline comparison record: `BASELINE_COMPARISONS.md`
- Ablation status: `../smap_msl_ablations/ABLATION_PLAN.md`

## Replication archive

See `REPLICATION.md` for the exact archive manifest, verification procedure, environment, and Zenodo release instructions.
