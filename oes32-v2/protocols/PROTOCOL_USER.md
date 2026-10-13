# Agent Instruction: Maximize Legitimate OES-32-v2 Performance on NASA SMAP/MSL

**Version:** OES-32-v2 Protocol Lock · October 2026  
**Status:** Preregistered research protocol for transparent residual-style detector improvement  
**Scope:** Offline benchmark evaluation only. Not operational, not flight software, not quantum hardware.

---

## OBJECTIVE

You are an autonomous research-engineering agent working on **OES-32-v2**.

**Primary objective:** Maximize legitimate, reproducible, unadjusted event-level F1 of OES-32-v2 on the NASA SMAP/MSL telemetry anomaly benchmark (Hundman et al., KDD 2018 / telemanom), subject to hard scientific integrity constraints.

The goal is **not** to maximize a number by any means. The goal is to determine, under a locked protocol, whether a carefully engineered improvement to the transparent OES-32 residual/window detector produces a meaningful, honest gain over:

- Frozen OES-32-v1 (as evaluated in sparkainlp-x/oes-resilience v0.5)
- Simple, strong classical baselines (maxabs, CUSUM, EWMA, robust MAD/z-score, PCA residual, Isolation Forest)

If no predeclared improvement is observed, report “no improvement demonstrated” without hedging.

---

## CONTEXT FROM PRIOR WORK

In oes-resilience v0.5 (preregistered protocol), the one-line OES32 detector:

```
score = 0.45·max|x| + 0.35·RMS(x) + 0.20·mean|x|   (32-sample window)
```

failed to meet its success criterion against maxabs, CUSUM, z-score, and EWMA on both SMAP and MSL under event-level F1 with train-calibrated false-alarm-rate thresholds. maxabs was particularly strong. Thresholds showed substantial train-to-test drift.

v2 must improve on this baseline under stricter leakage controls and a validation-driven selection process while remaining fully transparent and interpretable.

---

## NON-NEGOTIABLE BOUNDARIES

- Preserve all OES-32-v1 code, protocol, configurations, figures, metrics, and claims completely unchanged.
- All new work lives exclusively in a new versioned directory: `oes32-v2/`.
- **No test-set influence of any kind** on configuration selection, threshold calibration, feature construction, or model fitting.
- No future information in any mode labeled “online”.
- No random splitting of overlapping windows.
- No post-hoc deletion or down-weighting of low-performing channels after seeing results.
- No change to scoring rules, anomaly definitions, or primary success criteria after the final test evaluation.
- Do not report any result as operational spacecraft performance, flight validation, hardware sensing, quantum advantage, or real-world fault prediction.
- Explicitly acknowledge known community limitations of the SMAP/MSL benchmark (unrealistic anomaly density, possible label noise, distributional shift, short series, weak command correlation) in the final report. Cite relevant critiques (e.g., OPS-SAT paper and related analyses).

---

## PRIMARY OPTIMIZATION TARGET

Maximize **validation event-level F1**:

\[
\text{event\_F1} = \frac{2 \cdot \text{event\_precision} \cdot \text{event\_recall}}{\text{event\_precision} + \text{event\_recall}}
\]

**Hard constraints (all must be satisfied):**

1. Validation event precision ≥ 0.55
2. Validation event recall ≥ 0.55
3. Absolute improvement ≥ +0.05 over frozen OES-32-v1 event F1 on the same validation split
4. Wall-clock runtime ≤ 2× frozen OES-32-v1 on identical hardware and input
5. Zero test-set influence on any selection decision

If the primary constraint cannot be met, the run ends with a clear negative result.

**Secondary targets** (only after primary is satisfied):

- Lower false-positive event count
- Lower duplicate alerts per true event
- Lower median detection delay
- Higher macro-average per-channel F1
- Practical memory footprint
- Full score-component and threshold interpretability for every alert
- Consistency across SMAP and MSL (not success on only one)

---

## FROZEN EVALUATION PROTOCOL

Before any experimental run, create and commit:

- `protocols/oes32_v2_protocol.md` (this document, expanded with exact implementation details)
- `configs/search_space.yaml`
- `configs/evaluation_rules.yaml`
- `data/dataset_manifest.json`

The protocol must lock:

- Exact data source, version, SHA-256 checksums, channel list, shapes, time ordering, and label mapping (telemanom / Hundman et al. 2018)
- Chronological train / validation / test ranges with a temporal gap ≥ maximum window length
- Exact event-overlap rule, event-merging rule, and point-level vs event-level definitions
- Fixed seed list for any stochastic component
- Compute budget
- Primary and secondary targets
- Rule that final test evaluation occurs exactly once on the frozen configuration

---

## DATA HANDLING

1. Download raw data from the canonical telemanom source. Record SHA-256 of every file.
2. Verify timestamp/index alignment between telemetry and labels.
3. Create a chronological validation set by holding out the final portion of the anomaly-free train data (with a gap). Never touch the official test set until lock.
4. Fit all scalers, PCA, imputers, and threshold calibrations **only on training data**.
5. Tune all hyperparameters **only on validation data**.
6. Default scaling: robust per-channel median/MAD:

   \[
   x_{\text{scaled}} = \frac{x - \text{median}_{\text{train}}}{\max(1.4826 \times \text{MAD}_{\text{train}}, \epsilon)}
   \]

7. Log channels with near-zero MAD, constant segments, missing values, or invalid values.
8. Explicit, documented missing-value policy. Never silently drop records.
9. Clearly label every run as “online” (causal) or “offline benchmark mode”.

---

## REQUIRED MODEL FAMILY

Implement and evaluate under identical splits and event-level scoring:

A. Frozen OES-32-v1 reference (oes-resilience formula and protocol)
B. Rolling robust MAD / z-score baseline
C. Fixed training-percentile threshold baseline
D. PCA reconstruction-residual baseline (fit on train only)
E. Isolation Forest baseline (train features only)
F. OES-32-v2 standalone
G. OES-32-v2 + one simple complementary score (only if validation shows clear added value)

All methods deterministic where possible. Stochastic methods report mean ± std across the full fixed seed list.

---

## OES-32-V2 SEARCH SPACE (PREDECLARED ONLY)

Search only within the following. Do not expand after validation search begins.

1. **Robust normalization**
   - Global vs per-channel median/MAD
   - Epsilon: {1e-12, 1e-9, 1e-6}
   - Optional clipping: none, [-5, 5], [-10, 10]

2. **Score windows & aggregation**
   - Window length: {8, 16, 32, 64, 128}
   - Aggregation: mean, median, max, top-k mean
   - Top-k: {1, 2, 4, 8}
   - Optional smoothing: none, 3, 5, 9

3. **Threshold calibration (validation only)**
   - Per-channel quantile: {0.95, 0.975, 0.99, 0.995, 0.999}
   - Global quantile (same candidates)
   - Robust multiplier: {2.5, 3.0, 3.5, 4.0, 4.5}

4. **Alert post-processing**
   - Hysteresis low/high ratio: {0.60, 0.70, 0.80, 0.90}
   - Minimum alert duration: {1, 2, 3, 5, 8}
   - Maximum merge gap: {0, 1, 2, 4, 8}
   - Cooldown: {0, 1, 2, 4}

5. **Complementary features (select at most one)**
   - OES-32 only
   - OES-32 + rolling MAD
   - OES-32 + PCA residual
   - Combination: normalized mean, max, or weighted sum
   - Weights: {0.25, 0.50, 0.75}

---

## SEARCH STRATEGY

- Stage 1: Small factorial screening to identify high-value regions.
- Stage 2: Bounded validation-only search inside promising regions.
- Deterministic trial ordering.
- Log **every** trial (including failures) to `experiments/all_trials.csv`.
- Validation leaderboard ranked by: event F1 → event precision → macro channel F1 → fewer FP events → lower runtime.
- Select exactly one final configuration. Prefer simpler, faster, more interpretable when statistically similar.
- Freeze in `configs/final_config.yaml` **before** any test evaluation.

---

## EVENT-LEVEL SCORING (LOCKED)

- Ground-truth event = contiguous labeled anomaly interval.
- Predicted event = contiguous alert after hysteresis, min-duration, and merging.
- Event TP = ≥1 predicted event overlaps a ground-truth event under the frozen overlap rule.
- Event FN = unmatched ground-truth event.
- Event FP = predicted event matching no ground-truth event.

Report (clearly labeled):

- Event precision, recall, F1
- Point precision, recall, F1 (unadjusted)
- TP / FP / FN event counts
- Duplicate predictions per matched event
- Median and percentile detection delay
- Predicted-event duration distribution
- Per-channel metrics + macro and micro averages
- Separate SMAP and MSL results
- 95% bootstrap CIs (resample at event or channel level, not individual timesteps)
- Point-adjusted F1 only as a clearly labeled secondary metric

---

## FAILURE ATLAS

For frozen v1 and selected v2:

- Extract all FP and FN events.
- Categorize: isolated spike, drift, regime shift, periodic mismatch, noisy segment, threshold miss, score failure, data-quality issue, unknown.
- Generate overlay plots (telemetry + score + threshold + alerts + labels).
- Report the five most frequent failure patterns with examples.

---

## FINAL TEST DISCIPLINE

Before the single final test:

1. Commit code and record SHA.
2. Freeze `final_config.yaml`.
3. Save data-manifest hashes.
4. Record seeds and evaluation-code version.
5. Write a short “test lock” note confirming zero test-label influence.

Then run the frozen configuration **exactly once**. No post-hoc tuning. Rerun only to correct an execution failure under identical conditions.

---

## REQUIRED ARTIFACTS

- `protocols/oes32_v2_protocol.md`
- `data/dataset_manifest.json`
- `configs/search_space.yaml`
- `configs/evaluation_rules.yaml`
- `configs/final_config.yaml`
- `experiments/all_trials.csv`
- `experiments/validation_leaderboard.csv`
- `results/v1_reference.json`
- `results/baseline_comparison.csv`
- `results/validation_summary.csv`
- `results/final_test_summary.json`
- `results/per_channel_metrics.csv`
- `results/event_matches.csv`
- `results/failure_atlas.csv`
- `figures/score_label_alert_overlays/`
- `figures/per_channel_f1_distribution/`
- `figures/precision_recall/`
- `figures/runtime_memory/`
- `reports/OES32_V2_REPORT.md`
- `reports/EVIDENCE_PASSPORT.json`
- `reports/REPRODUCIBILITY.md`
- `CITATION.cff` and Zenodo-ready deposit package

---

## FINAL REPORT REQUIREMENTS

The report must include:

- Frozen research question and hypotheses
- Exact dataset, split, and gap definition
- All preprocessing and calibration choices
- Full baseline comparison (including prior oes-resilience v0.5 numbers)
- Validation selection result
- One-time final test result with CIs
- Per-channel and SMAP/MSL breakdowns
- Runtime and resource use
- Failure atlas
- Explicit limitations (including known SMAP/MSL benchmark critiques)
- Reproduction commands
- Paths to all artifacts
- Clear statement on whether the predeclared criterion was met

---

## FINAL CLAIM RULES

**Allowed:**

- “OES-32-v2 improved event-level F1 under this frozen SMAP/MSL benchmark protocol.”
- “OES-32-v2 did not meet its predeclared improvement criterion.”
- “The method shows preliminary benchmark-scoped evidence.”
- “Independent replication and additional datasets are required.”
- “This is research software, not an operational detection system.”

**Forbidden:**

- Any claim of spacecraft fault detection, mission readiness, flight validation, quantum advantage, medical/safety validation, or state-of-the-art status without protocol-matched evidence.

---

## END-OF-RUN RESPONSE FORMAT

Respond with:

1. Selected configuration (full YAML or structured summary)
2. Validation event precision, recall, F1, macro F1, FP count, runtime
3. Final-test event precision, recall, F1, macro F1, FP count, runtime
4. Comparison table: v1, v2, and all baselines
5. Whether the predeclared criterion was met (yes/no + magnitude)
6. Top five failure modes
7. One strongest defensible conclusion
8. One central limitation
9. Exact paths to all saved artifacts
10. Recommendation for a new preregistered v3 experiment only if v2 fails

---

**End of locked protocol.**
