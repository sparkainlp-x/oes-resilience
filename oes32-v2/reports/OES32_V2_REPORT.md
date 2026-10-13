# OES-32-v2 on NASA SMAP/MSL: locked-protocol result

**Result: OES-32-v2 did not meet its predeclared improvement criterion.**

The selected configuration failed:
- one validation hard constraint: runtime was 2.27× v1, against a limit of 2×;
- the confirmatory single-run test criterion: v2 − v1 event F1 was −0.166 on SMAP and −0.001 on MSL, against a required +0.05 on each.

This is research software evaluated offline on one public benchmark. It is not an operational detection system and makes no flight, spacecraft fault-detection, quantum or state-of-the-art claim. Mode: offline benchmark mode.

## 1. Research question and hypotheses
Does a transparent residual/window detector, selected only on validation data, improve event-level F1 over frozen OES-32-v1 under a locked protocol?
- **H1:** all five hard constraints hold on validation.
- **H2:** in the single test run, v2 event F1 ≥ v1 + 0.05 on SMAP and on MSL.

The protocol is in [`protocols/oes32_v2_protocol.md`](../protocols/oes32_v2_protocol.md); the user's verbatim text is in [`protocols/PROTOCOL_USER.md`](../protocols/PROTOCOL_USER.md). It was locked in commit `8b474b6` before any run.

## 2. Data, split and gap
- **Data:** the SMAP/MSL telemanom archive (Hundman et al. 2018), the same bytes as v1. All 165 files were verified against `reports/smap_msl_data.sha256`; the manifest is [`data/dataset_manifest.json`](../data/dataset_manifest.json).
- **Channels:** 54 SMAP and 27 MSL labelled channels. Only column 0 is used.
- **Split:** chronological, per channel, on the train file only:
  - fit segment: `[0, n − ⌊0.3n⌋ − 128)`;
  - gap: 128 samples, equal to the maximum window;
  - validation segment: the last ⌊0.3n⌋ samples.
- **D-12 (SMAP):** too short for a validation segment; for the test it uses the whole train file.
- **Validation events:** the train files are unlabelled, so validation events are deterministic synthetic injections (122 events: 36 spikes, 23 level shifts, 34 drifts, 29 variance bursts). **Validation scores therefore measure detection of injected events, not real anomalies.**

## 3. Preprocessing and calibration
- **v2 scaling:** robust median/MAD, either per channel or global per spacecraft, with ε and clipping as searched, fit on the fit segment.
- **Thresholds:** computed from scores on the clean, un-injected validation segment.
- **Baselines B, D and E:** v1's 1 % calibration rule on the same clean segment.
- **Baseline C:** the 99.5th percentile of fit-segment |x|.
- **Frozen v1:** exactly as published (train mean/std standardization, 1 % train calibration).
- **Data quality:** no non-finite values. 22 SMAP and 11 MSL channels have zero train MAD, which interacts strongly with ε under robust scaling.

## 4. Validation search and selection
- **Search:** 4223 trials (stage 1: 2100, stage 2a: 90, stage 2b: 2000, stage 2c: 33), 0 failures, 67 s. The budget was 6000 trials and 4 h.
- **Logs:** [`experiments/all_trials.csv`](../experiments/all_trials.csv) and [`experiments/validation_leaderboard.csv`](../experiments/validation_leaderboard.csv).
- **Selected configuration:** trial 4193, model G.
  - Global median/MAD scaling, ε = 1e-9, no clipping.
  - Window 128, max of |x|, no smoothing, combined by `max` with a normalized rolling-MAD score.
  - Per-channel 0.995 quantile threshold.
  - Hysteresis 0.6; min duration 1; merge gap 0; cooldown 0.
- **Why G over F:** the best standalone F (trial 2591) scored validation F1 0.949; G scored 0.971, which clears the +0.01 rule.

| Validation (synthetic events, pooled) | P | R | F1 | macro F1 | FP events |
|---|---|---|---|---|---|
| A v1 frozen | 0.651 | 0.811 | 0.723 | 0.739 | 53 |
| B rolling robust z | 0.347 | 0.754 | 0.475 | 0.571 | 173 |
| C train percentile | 0.458 | 0.795 | 0.581 | 0.686 | 115 |
| D PCA residual | 0.322 | 0.746 | 0.449 | 0.541 | 192 |
| E Isolation Forest (5 seeds) | — | — | 0.341 ± 0.015 | — | 165–191 |
| F v2 standalone (best) | 0.991 | 0.910 | 0.949 | 0.896 | 1 |
| **G v2 selected** | **0.967** | **0.975** | **0.971** | **0.953** | **4** |

**Hard constraints (selected configuration):**

| Constraint | Result |
|---|---|
| Precision 0.967 ≥ 0.55 | met |
| Recall 0.975 ≥ 0.55 | met |
| v2 − v1 = +0.249 ≥ 0.05 | met |
| Runtime 0.176 s / 0.077 s = **2.27× > 2×** | **not met** |
| Test influence | none |

H1 therefore failed before the test.

**Caveat on validation:** the selected configuration has a median predicted-event length of 682 samples on validation, with point precision 0.19. Long alerts that overlap an injected event count as one TP and no FP under the locked any-overlap rule. This favours wide alerts and inflates validation event F1. It is a property of the locked metric on this validation design, recorded here rather than corrected after the fact.

## 5. Single test run
Run once at 00:36–00:45 EDT, 2026-10-13, after test lock `5477ffb`, exit 0, no rerun. Frozen v1 reproduced the published v0.5 pooled event F1 exactly (SMAP 0.378601, MSL 0.434783).

| Test (single run) | SMAP P / R / F1 [95 % CI] | SMAP macro, FP | MSL P / R / F1 [95 % CI] | MSL macro, FP |
|---|---|---|---|---|
| A v1 frozen | 0.263 / 0.676 / **0.379** [0.272, 0.525] | 0.545, 129 | 0.316 / 0.694 / **0.435** [0.333, 0.578] | 0.495, 54 |
| B rolling robust z | 0.035 / 0.647 / 0.066 | 0.459, 1216 | 0.091 / 0.722 / 0.161 | 0.374, 260 |
| C train percentile | 0.047 / 0.632 / 0.087 | 0.553, 875 | 0.164 / 0.667 / 0.264 | 0.384, 122 |
| D PCA residual | 0.061 / 0.824 / 0.114 | 0.539, 859 | 0.098 / 0.750 / 0.174 | 0.453, 248 |
| E Isolation Forest (mean ± sd, 5 seeds) | F1 0.062 ± 0.006 | — | F1 0.151 ± 0.007 | — |
| **G v2 selected** | 0.125 / 0.721 / **0.213** [0.109, 0.681] | 0.696, 344 | 0.290 / 0.861 / **0.434** [0.243, 0.857] | 0.778, 76 |

- **v2 − v1 event F1:** SMAP −0.166 (95 % CI [−0.340, 0.296]); MSL −0.001 (95 % CI [−0.255, 0.435]). H2 failed.
- **Secondary metrics (test):**
  - Median detection delay is 0 samples for v2 (v1: 29 SMAP, 17 MSL).
  - Duplicates per matched event: v2 2.88 SMAP / 0.03 MSL; v1 0.57 / 0.76.
  - Macro channel F1 is higher for v2 (0.696 vs 0.545 SMAP; 0.778 vs 0.495 MSL).
  - Unadjusted point F1 is lower for v2 (0.234 vs 0.401 SMAP; 0.212 vs 0.218 MSL).
  - Point-adjusted F1 is secondary only.
- **Comparison with published oes-resilience v0.5 (test, 1 % target):**
  - maxabs: 0.475 SMAP / 0.500 MSL.
  - CUSUM: 0.169 / 0.514.
  - All rows are in [`results/baseline_comparison.csv`](../results/baseline_comparison.csv).

## 6. Runtime and memory
- **Validation (locked measure, median of 3 passes):** v1 0.077 s, v2 0.176 s (2.27×). tracemalloc peak: v1 2.4 MB, v2 1.6 MB.
- **Test (single pass, 81 channels):** v1 0.45 s, v2 1.09 s. Isolation Forest took about 100–113 s per seed.
- **Figure:** [`figures/runtime_memory/`](../figures/runtime_memory/). Timings depend on the machine; this was one 8-core Linux box.

## 7. Failure atlas (test; automatic heuristic categories, protocol section 9)
- **Totals:** v1 has 183 FP and 33 FN events; v2 has 420 FP and 24 FN.

| Rank | v1 | v2 |
|---|---|---|
| 1 | FP / drift (68) | FP / data-quality issue (153) |
| 2 | FP / data-quality issue (45) | FP / drift (116) |
| 3 | FP / periodic mismatch (25) | FP / periodic mismatch (77) |
| 4 | FP / isolated spike (25) | FP / regime shift (43) |
| 5 | FN / threshold miss (12) | FP / noisy segment (16) |

- **Reading the atlas:** "data-quality issue" means the channel has zero train MAD or the alert sits on a constant segment. With global scaling and ε = 1e-9, small level changes in near-constant channels give very large scaled values; the overlay for A-8 shows this.
- **Files:** overlays are in [`figures/score_label_alert_overlays/`](../figures/score_label_alert_overlays/); the full list is [`results/failure_atlas.csv`](../results/failure_atlas.csv).
- **Status of the categories:** they come from fixed rules, not expert review.

## 8. Limitations
- **Synthetic validation.** It was easy (v2 validation F1 0.97) and did not transfer to the real test anomalies. This is the main reason the selection failed on the test.
- **Event-F1 rule.** The any-overlap event F1 rewards long alerts and does not penalize alert length.
- **SMAP/MSL as a benchmark.** It has known problems. Ruszczak et al. (2025) list unrealistic anomaly density, many trivial anomalies, mislabelled ground truth, distributional shifts and weak correlation between commands and channels, and introduce the OPS-SAT benchmark as an alternative. Wu & Keogh (2023) argue that many exemplars in the NASA and other popular benchmarks are flawed: triviality, unrealistic density, mislabelling and run-to-failure bias. Kim et al. (2022) show that point adjustment inflates scores; it is therefore secondary here. The series are short, only column 0 is used, and test sizes are small (68 + 36 events), which explains the wide CIs.
- **Scope.** One dataset, one protocol, one run. Independent replication and additional datasets are required.

## 9. Reproduction
See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## 10. Artifacts
- **Protocol and configs:**
  - `protocols/oes32_v2_protocol.md`, `protocols/PROTOCOL_USER.md`
  - `configs/search_space.yaml`, `configs/evaluation_rules.yaml`, `configs/final_config.yaml`
  - `data/dataset_manifest.json`
- **Experiments:** `experiments/all_trials.csv`, `experiments/validation_leaderboard.csv`, `experiments/search_meta.json`, `experiments/RUN_LOG.md`, `experiments/TEST_LOCK.md`
- **Results:**
  - `results/v1_reference.json`, `results/baseline_comparison.csv`, `results/validation_summary.csv`, `results/validation_selection.json`
  - `results/final_test_summary.json`, `results/per_channel_metrics.csv`, `results/event_matches.csv`
  - `results/failure_atlas.csv`, `results/failure_atlas_top5.json`, `results/test_scores_v1_v2.npz`
- **Figures:** `figures/score_label_alert_overlays/`, `figures/per_channel_f1_distribution/`, `figures/precision_recall/`, `figures/runtime_memory/`
- **Reports and metadata:** `reports/OES32_V2_REPORT.md`, `reports/EVIDENCE_PASSPORT.json`, `reports/REPRODUCIBILITY.md`, `CITATION.cff`, `zenodo/.zenodo.json`

## References (verified via Crossref, 2026-10-13)
- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. (2018). Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. KDD '18, 387–395. doi:10.1145/3219819.3219845
- Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. (2025). The OPS-SAT benchmark for detecting anomalies in satellite telemetry. *Scientific Data* 12, 710. doi:10.1038/s41597-025-05035-3
- Ruszczak, B., Kotowski, K., Andrzejewski, J., et al. (2023). Machine Learning Detects Anomalies in OPS-SAT Telemetry. ICCS 2023, LNCS 14073, 295–306. doi:10.1007/978-3-031-35995-8_21
- Wu, R., Keogh, E. J. (2023). Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress. *IEEE TKDE* 35(3), 2421–2429. doi:10.1109/TKDE.2021.3112126
- Kim, S., Choi, K., Choi, H.-S., Lee, B., Yoon, S. (2022). Towards a Rigorous Evaluation of Time-Series Anomaly Detection. AAAI 36(7), 7194–7201. doi:10.1609/aaai.v36i7.20680
