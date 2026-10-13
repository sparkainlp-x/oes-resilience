# OES-32-v2 protocol (LOCKED)

**Status:** locked and committed before any experiment run, validation search or test evaluation.
**Mode:** offline benchmark mode. Scores are causal (trailing windows); alert post-processing is retrospective.
**Scope:** offline benchmark evaluation of research software on one public dataset. Not operational, not flight software, not hardware sensing, not quantum.
**Source:** the user's protocol, verbatim in [`PROTOCOL_USER.md`](PROTOCOL_USER.md). This file adds the exact implementation details that document requires. Where the two differ, the differences are listed in section 12, and they were fixed before any run.
**Location:** everything lives in `oes32-v2/`. Frozen v1 code, protocol, results and tags in this repository are not modified; v2 imports v1 code read-only.

## 1. Research question and hypotheses
- **Q:** Under a locked protocol, does a transparent residual/window detector (OES-32-v2), selected only on validation data, improve event-level F1 over frozen OES-32-v1 on NASA SMAP/MSL?
- **H1 (validation):** the selected v2 configuration meets all five hard constraints (event precision ≥ 0.55, event recall ≥ 0.55, event F1 ≥ v1 + 0.05 on the same validation split, runtime ≤ 2× v1, zero test influence).
- **H2 (confirmatory, single test run):** v2 test event F1 ≥ frozen-v1 test event F1 + 0.05 on SMAP and on MSL separately.
- **Criterion met** only if H1 and H2 both hold. Otherwise the result reads: "OES-32-v2 did not meet its predeclared improvement criterion."

## 2. Data
- **Dataset:** NASA SMAP/MSL telemetry anomaly data (Hundman et al., KDD 2018, doi:10.1145/3219819.3219845; telemanom). The archive is the same one used by v1.
- **Manifest:** `data/dataset_manifest.json` lists the SHA-256 of all 165 files. It reuses the verified v1 manifest `reports/smap_msl_data.sha256` (SHA-256 `bf39e514…8422d`) and records shapes, channel list, label mapping, splits and test-event windows.
- **Not redistributed.** The data are downloaded with `oes-resilience smap-msl fetch` and verified.
- **Channels:** every labelled channel (SMAP 54, MSL 27). Unlabelled T-10 is excluded, as in v1. No channel is removed after seeing results.
- **Signal:** column 0 only (float64). Row order is time order, and train precedes test.
- **Labels:** labels are loaded with v1 `load_labels`. Rows sharing (spacecraft, chan_id) are pooled. Overlapping or adjacent inclusive windows are merged into one event.
- **Missing and invalid values:** v1 `load_series` rejects non-finite column-0 values; none exist in the verified files. If one appeared, the run would stop with an error rather than drop records. Channels with zero train MAD, constant segments, or near-zero MAD (< 1e-9) are logged in `results/data_quality.csv`.

## 3. Chronological split (per channel, train file only)
- Let `n` be the train length and `val_len = floor(0.30 n)`.
- **Fit segment:** `[0, n − val_len − 128)`.
- **Gap:** 128 samples, which equals the maximum window length.
- **Validation segment:** `[n − val_len, n)`.
- **Minimum length:** fit and validation must each have ≥ 128 samples. Only SMAP D-12 (n = 312) fails this and has no validation segment. D-12 is still evaluated in the test. For D-12, normalization statistics and thresholds come from the whole train file, using the same rules applied to train scores.
- **Test:** the official test file, untouched until the test lock.
- **Fitting:** all scalers, PCA and Isolation Forest are fit on the fit segment only.
- **Thresholds:** thresholds are computed from detector scores on the clean, un-injected validation segment. These are train-file samples and carry no labels.
- **Tuning:** hyperparameters are tuned only on validation metrics.

## 4. Validation events (synthetic injection)
The train files carry no anomaly labels. Event-level validation metrics therefore need labelled validation events. Each channel's validation segment gets deterministic synthetic anomalies, injected into a copy of the segment:
- **Count:** `k = max(1, floor(len/400))` events, one per equal slot.
- **Seed:** `20261013 + channel index`. The channel index follows the manifest order.
- **Type:** drawn uniformly from {spike (1–3 samples, ±4–8 s), level shift (50–150, ±2–4 s), linear drift (100–200, ramp to ±3–6 s), variance burst (50–150, Gaussian noise with sd 2–4 s)}.
- **Scale `s`:** 1.4826·MAD of the fit segment. If that is ≤ 1e-9, the fit std is used; if that is also ≤ 1e-9, `s = 1`.

Code: `oes32v2/data.py::inject`. Validation event F1 is a measure of detecting these synthetic events. It is **not** a measure of detecting real anomalies, and the report states this. The clean validation segment is used only for threshold calibration.

## 5. Methods (identical splits and scoring)
- **A. Frozen OES-32-v1:** the v1 code as published.
  - **Validation:** v1 standardization (mean and std on the fit segment) and the v1 calibration rule at the 1 % target on fit-segment scores, with alarm rule `score ≥ T`.
  - **Test:** the exact v1 protocol, with standardization and calibration on the full train file. This must reproduce the published v0.5 pooled event F1 (SMAP 0.378601, MSL 0.434783); a mismatch is reported as an execution failure.
- **B. Rolling robust z:**
  - **Score:** `|x_t − median(x_{t−63..t})| / max(1.4826·MAD(x_{t−63..t}), 1)`, where `x` is per-channel median/MAD-scaled with ε = 1e-9.
  - **Threshold:** v1 calibration rule at 1 % on clean-validation scores.
- **C. Fixed train-percentile:**
  - **Score:** `|x_scaled|`.
  - **Threshold:** T = 99.5th percentile of the fit-segment scores; alarm when score > T.
- **D. PCA reconstruction residual:**
  - **Model:** lag-16 embedding of `x_scaled` clipped to ±10. PCA is fit on fit-segment windows, keeping the smallest k explaining ≥ 90 % of variance (k ≤ 8).
  - **Score:** residual L2 norm.
  - **Threshold:** as in B.
- **E. Isolation Forest:**
  - **Model:** the same lag-16 windows. scikit-learn `IsolationForest(n_estimators=100, max_samples=256, random_state=seed)` is fit on ≤ 5000 evenly strided fit-segment windows.
  - **Score:** `−score_samples`.
  - **Threshold:** as in B.
  - **Seeds:** {0, 1, 2, 3, 4}, reported as mean ± std.
- **F. OES-32-v2 standalone:** the search space in section 6 with `complementary.feature: none`.
- **G. OES-32-v2 + one complementary score:** kept only if its validation event F1 exceeds the best F by ≥ 0.01. Otherwise F is selected.
- **Alert handling:** B–E use plain alarm runs (no post-processing). B, D and E use the alarm rule `score > T`.
- **Window padding:** every trailing window of a segment's first samples is padded with that segment's first value.

## 6. OES-32-v2 detector (search space `configs/search_space.yaml`)
1. **Scale:** `x = (raw − median_fit) / max(1.4826·MAD_fit, ε)`, either per channel or global. "Global" means one median and MAD per spacecraft, pooled over all fit segments. The scaled value is optionally clipped to [−c, c].
2. **Window score:** an aggregation of `|x|` over the trailing window of length w: mean, median, max, or the mean of the top-k values. Optionally, a trailing moving average of length 3, 5 or 9 follows.
3. **Complementary feature (optional):**
   - **Features:** rolling MAD is the B score; PCA residual is the D score, fit on the fit segment.
   - **Normalization:** each term is divided by the 99th percentile of its clean-validation scores, floored at 1e-12.
   - **Combination:** normalized_mean = (a + b)/2; max = max(a, b); weighted_sum = w·a + (1 − w)·b, where a is the OES term.
4. **Threshold T:** computed from the clean-validation scores. The options are the per-channel quantile q, the global quantile q pooled per spacecraft, or `median + m·1.4826·MAD`.
5. **Alert post-processing**, in this order:
   1. Hysteresis: an alert starts when score > T and continues while score > r·T.
   2. Runs separated by ≤ g non-alert samples are merged.
   3. Runs shorter than d samples are dropped.
   4. Cooldown: a run starting ≤ c samples after the end of the previous kept run is dropped.

## 7. Search strategy and compute budget
- **Budget (fixed before running):** at most 6000 validation trials and at most 4 wall-clock hours on the review box (8 cores, 15 GB RAM).
- **Staged screening:** the full factorial grid is far larger than the budget, so the search is staged as follows.
  - **Stage 1 (2100 trials):** window length × aggregation (including top-k) × smoothing × 15 threshold rules. Fixed settings: per-channel scaling, ε = 1e-9, no clipping, r = 0.9, d = 1, g = 0, c = 0, no complementary feature.
  - **Stage 2a (90 trials):** the top 5 from stage 1 × 18 normalization settings.
  - **Stage 2b (2000 trials):** the top 5 from stage 2a × 400 post-processing settings.
  - **Stage 2c (33 trials):** the top 3 from stage 2b × 11 complementary settings.
- **Trial order:** deterministic, following the enumeration order in `oes32v2/search.py`.
- **Logging:** every trial, including failures, is logged in `experiments/all_trials.csv`.
- **Ranking:** event F1 rounded to 3 dp (descending), then complexity (ascending), event precision (desc), macro channel F1 (desc), FP events (asc), runtime (asc), trial id (asc). Complexity is the number of settings that differ from the stage-1 fixed defaults, plus 1 for median or top-k aggregation, plus 2 for a complementary feature. The 3-dp rounding and the complexity term implement "prefer simpler when statistically similar".
- **Final configuration:** exactly one configuration is frozen in `configs/final_config.yaml` before the test.

## 8. Scoring (locked; `configs/evaluation_rules.yaml`)
- **Events:** a ground-truth event is a merged labelled interval. A predicted event is a run after post-processing. A predicted event matches a ground-truth event if they share ≥ 1 sample. TP and FN are counted per ground-truth event; FP is counted per unmatched predicted event.
- **Event metrics:** event precision, recall and F1 on counts summed over channels (micro). Macro F1 is the mean per-channel event F1 over channels with events.
- **Other metrics:**
  - unadjusted point precision, recall and F1;
  - TP, FP and FN counts;
  - duplicates per matched event;
  - detection delay (median, p25, p75, p90);
  - predicted-event duration distribution;
  - per-channel tables;
  - SMAP and MSL reported separately;
  - point-adjusted F1, labelled as a secondary metric.
- **Confidence intervals:** 95 % percentile bootstrap, resampling channels (10 000 resamples, seed 42), for event F1 and for v2 − v1.

## 9. Failure atlas (v1 and selected v2, test set, after the single test run)
Every FP and FN event is categorized automatically, using the first rule that applies. These are heuristics, not expert labels.
- **Data-quality issue:** zero train MAD, or the event lies in a constant segment.
- **Isolated spike:** event length ≤ 5.
- **Threshold miss:** FN only; the maximum score in the event is ≥ 0.8 T.
- **Drift:** |slope| × length ≥ 2 robust sd, from a linear fit over the event.
- **Regime shift:** |mean(event) − mean(preceding 128 samples)| ≥ 2 robust sd.
- **Noisy segment:** std(event) ≥ 2 × the robust sd of the preceding 128 samples.
- **Periodic mismatch:** the lag-1..64 autocorrelation of the preceding 256 samples has a peak ≥ 0.5.
- **Score failure:** FN only; the maximum score is < 0.8 T.
- **Unknown:** none of the above.

Overlay plots show telemetry, score, threshold, alerts and labels for the five most frequent patterns.

## 10. Final test discipline
Before the test:
1. Commit all code and record the SHA.
2. Freeze `configs/final_config.yaml`.
3. Save the data-manifest hash.
4. Record seeds and the evaluation-code version.
5. Write `experiments/TEST_LOCK.md`, which states that no test labels influenced any choice.

`oes32v2/run_test.py` refuses to run if the lock note is missing, if the lock's config hash differs, or if `results/final_test_summary.json` already exists. The test runs exactly once. A rerun is allowed only after an execution failure, under identical conditions, and is logged.

## 11. Claim rules
- **Allowed wording:** "improved event-level F1 under this frozen SMAP/MSL benchmark protocol", or "did not meet its predeclared improvement criterion".
- **Forbidden:** operational, flight, spacecraft fault detection, quantum, medical/safety or state-of-the-art claims.
- **Limitations:** the report cites known SMAP/MSL benchmark critiques.

## 12. Implementation decisions relative to PROTOCOL_USER.md (fixed before any run)
1. **Validation events are synthetic** (section 4), because the train files are unlabelled. The user protocol asks for a validation split from anomaly-free train data and validation event F1, and these cannot both hold without injected events.
2. **"Threshold calibration (validation only)"** is read as calibration on the clean validation segment, which is train-file data without labels. Fitting stays on the fit segment.
3. **Baselines and validation thresholds:** baseline hyperparameters are fixed (section 5) and not tuned. Their thresholds use the v1 1 % calibration rule on the same clean validation segment.
4. **Confirmatory test criterion (H2)** is added so that the single test run has a predeclared pass/fail rule.
5. **Run order:** the protocol says the run ends on a negative validation result. The task still requires the single frozen test run, so it is run and reported either way. A validation failure already means the criterion is not met.
6. **D-12** has no validation segment (section 3).
