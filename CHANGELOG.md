# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-10-03

First evaluation on **real public telemetry**. Everything else in the project remains synthetic, and the existing synthetic caveats are unchanged.

### Added
- **Preregistered protocol for a blind evaluation on real public telemetry** (NASA SMAP/MSL, Hundman et al. 2018): `reports/smap_msl_protocol.json` and `reports/smap_msl_protocol.md`, locked before any method was scored on the test split. It fixes the train-only calibration rule and targets, the five methods and their parameters, the event-level metrics, the bootstrap/Wilcoxon/Holm analysis, the seed and the success criteria.
- `oes_resilience.smap_msl` and CLI `oes-resilience smap-msl {fetch,verify,train-diagnostics,evaluate}`: download into a cache outside the repository with per-file SHA-256 verification against `reports/smap_msl_data.sha256` (raw data is never bundled), univariate adapter, train-only calibration and the preregistered analysis. `scripts/fetch_smap_msl.py` wraps the fetch step.
- `tests/test_smap_msl.py` on tiny synthetic fixtures (CI never downloads the dataset), including a blindness test: moving every test label leaves every threshold unchanged.
- **Results of the blind evaluation** in `reports/smap_msl_results/`: results JSON, summary, comparison and per-channel CSVs, an SVG plot and `SHA256SUMS`, from one `--verify` run of the lock-commit code. The README has a new section labelled REAL DATA.
  - **The pre-stated success criterion is not met.** At the primary train-calibrated 1% target, OES32's pooled event F1 is 0.379 on SMAP and 0.435 on MSL.
  - It is significantly better only than EWMA on SMAP (ΔF1 +0.29, paired 95% CI [+0.19, +0.43], Holm p 0.021) and significantly worse than no baseline.
  - `maxabs` has a higher pooled F1 on both datasets (0.475 and 0.500) and CUSUM has the highest on MSL (0.514); none of these differences is significant.
  - OES32's pooled advantage over `zscore`, EWMA and CUSUM comes mostly from fewer, merged false-alarm runs, an effect of its 32-sample window that `maxabs` shares. Per-channel Wilcoxon tests do not confirm it.
  - Train-calibrated thresholds did not transfer to test: the alarm rate on unlabelled test samples was 10–41% against the 1% train target.
- References: the dataset paper (Hundman et al. 2018, doi:10.1145/3219819.3219845) is added to `CITATION.cff` and `.zenodo.json`, together with the keywords `nasa-smap-msl` and `preregistration`.

### Changed
- README: the intro, "What it is not", the replay caveat, the roadmap, the limitations, the tests and the citation now reflect the v0.5 real-data evaluation. The synthetic caveats are kept.
- `.gitignore` ignores `*.npy` and the download archive, so raw data cannot be committed by accident.
- The version is now 0.5.0 in `_version.py`, `CITATION.cff` and `.zenodo.json`, and the version test expects 0.5.0.

### Disclosures
- **No change after the lock.** Neither the protocol (`reports/smap_msl_protocol.json`, SHA-256 `927cf78f…8ea3`, locked in commit `fdf9905`), the evaluation code nor the data changed after the lock. The results were produced by the lock-commit code, whose package version string was still 0.4.0; the `version` field in the results JSON records that.
- **Data source.** The original telemanom S3 bucket returned HTTP 403, so the data was fetched from the Kaggle mirror that the telemanom README links to. Every extracted file matches the committed SHA-256 manifest, and the label CSV is byte-identical to the one in the telemanom repository.

## [0.4.0] - 2026-09-30

### Added
- **Replay evaluation** (`oes_resilience.replay`, CLI `oes-resilience replay`). It scores a JSONL telemetry replay (512 channels, timezone-aware timestamps, regime and event labels, an optional incumbent alarm and flags) against a hashed preregistration. Folded in from the private `oes512-replay-eval` pilot harness, which is now archived. See `docs/replay.md`.
  - `oes32` scores come from `core.score_signals`, identical to the pilot's formula. All other detectors come from the plugin API (`maxabs`, `zscore`, `ewma`, `cusum`, `oes32+ewma`, and optional or plugin detectors).
  - For each detector and regime: event recall and missed event IDs; FP frame rate, FP frames and alarm episodes per 10 minutes; framewise block-localisation precision, recall and IoU; and detection latency.
  - Preregistration schema 2 has a `threshold_policy`:
    - `calibrated` (the default in the example) sets every detector's threshold on an event-free calibration replay, which is locked by SHA-256. It uses the same `calibrate_threshold` rule and ~1% FP budget as `compare` and `stress`, with frame units.
    - `fixed` keeps written-down thresholds, marks the report `comparison_is_calibrated: false` and adds a fairness note.
  - Schema 1 (the pilot's format) is still accepted and maps to fixed `oes32` and `maxabs` thresholds.
  - The pilot's input-hardening fixes are carried over. Deep nesting, oversized integers, non-finite values, duplicate or unknown keys, naive or non-increasing timestamps, non-contiguous events, partial incumbent columns and blank lines are rejected with exit code 2. Lines are split on `\n` only, and CRLF is accepted.
  - Deterministic report, scorecard CSV and manifest; `--verify` exits with code 4 on any byte difference. `--write-example-inputs` regenerates the deterministic example replay (byte-identical to the pilot's data) and the calibration replay.
- **`maxabs` built-in detector** (`max|x|` per block, NaN-aware), the pilot's baseline.
- `examples/replay_prereg_calibrated.json`, `examples/replay_prereg_fixed050.json`, `examples/replay_scorecard.csv` and `examples/replay_fixed050_scorecard.csv`. CI regenerates the replays and checks both scorecards byte for byte; they are also byte-identical across NumPy 1.26–2.5.
- README: a replay section with SYNTHETIC results and caveats. Measured negative result: on a regime-segmented replay, the calibrated CUSUM misses every event and EWMA adds FP frames after events.

### Changed
- Roadmap: the SMAP/MSL dataset adapters move to v0.5.
- `docs/detector-api.md`: the plugin example is renamed `peakabs`, so it no longer clashes with the built-in `maxabs`.

## [0.3.0] - 2026-09-30

### Added
- **Stress suite** (`oes_resilience.stress`, CLI `oes-resilience stress`). It has 15 seeded scenarios with exact ground truth on the frame and stream tracks:
  - clean references;
  - heavy-tailed Student-t background (configurable df);
  - channel dropout (NaN; configurable fraction);
  - saturated (stuck) channels and clipped bursts;
  - narrow and cross-block bursts;
  - global shift, with and without a burst;
  - correlated bursts across adjacent blocks;
  - slow linear drift and an abrupt baseline step (stream only).

  Thresholds are calibrated exactly as in `compare` and are not re-tuned per scenario. See `docs/stress.md`.
- **Robustness metrics** in the stress scorecard, relative to each detector's clean baseline:
  - FP inflation and recall retention;
  - latency delta;
  - 95% Wilson intervals;
  - a conservative `change` flag (worse or better only when the intervals do not overlap).

  The JSON and CSV are deterministic and hash-verified (`stress --verify`).
- **Explicit missing-data handling.**
  - `Detector.supports_missing`: NaN is rejected unless a detector declares support, and `inf` is always rejected.
  - Mask-aware scoring for `oes32`, `zscore`, `ewma`, `cusum` and the hybrid, via `masked_block_stats`, with no zero-filling. A fully missing block scores 0.
  - The optional `iforest` is reported as unsupported for dropout scenarios.
- **`oes32+ewma` hybrid detector:** it alarms if OES32 or EWMA fires. The parts are normalised on clean streams and one threshold is calibrated jointly to the same FP budget. Measured result: it does not beat EWMA alone (see the README).
- `compare`: the hybrid is added to the default detectors, plus `--fit-streams` (default 200 per clean regime) for temporal detectors that need fitting.
- `examples/stress_scorecard.csv` and `examples/stress_burst03_scorecard.csv`; CI checks both byte for byte.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and `SECURITY.md`.
- README: a Limitations section and a Python-versions badge.
- New keywords in `CITATION.cff` and `.zenodo.json`: observability, operational-resilience, signal-processing and scientific-software.

### Changed
- Detector names may contain `+`.
- `examples/compare_scorecard.csv` gains the five `oes32+ewma` rows. All other rows are byte-identical to v0.2.0.
- `write_compare_outputs` accepts custom CSV fields and a Markdown renderer. The calibration steps are exposed as `frame_calibration` and `stream_calibration`, shared by `compare` and `stress`.

### Unchanged
- The v0.1 and v0.2 regimes, seeds, scoring and example outputs (`baseline_summary.csv`, `sweep_summary.csv` and the v0.2 rows of `compare_scorecard.csv`). Complete-data inputs take the v0.2 code paths.

## [0.2.0] - 2026-09-30

### Added
- Package layout (`oes_resilience/`: `core`, `detectors`, `streams`, `scorecard`). Run the CLI with `python -m oes_resilience` or `oes-resilience`. The v0.1 API is available from `oes_resilience` and `oes_resilience.core`.
- Detector plugin API: a `Detector` base class with `fit()` and `detect()` returning a common `DetectionResult` (scores per block, detected blocks, status, explanation). Also a registry (`register_detector`, `create_detector`, ...) and entry-point plugins in the `oes_resilience.detectors` group. See `docs/detector-api.md`.
- Detectors:
  - `oes32`, the reference; it reproduces v0.1 scores exactly.
  - `zscore`, a robust z-score of block RMS against a median/MAD baseline from clean frames.
  - `ewma` and `cusum`, temporal detectors on block means standardised against each stream's warm-up.
  - `iforest`, an optional Isolation Forest (extra: `oes-resilience[iforest]`). The core stays NumPy-only.
- Multi-step synthetic streams (T × 512) with a recorded onset step: `stream_stable`, `stream_noisy`, `stream_burst`, `stream_shift` (a +0.02 persistent shift on one block) and `stream_shock`. The single-frame v0.1 regimes are unchanged.
- `compare` subcommand: fits all detectors on the same clean data and calibrates every threshold the same way on held-out clean data (target FP per clean regime, default 1%).
  - It writes a deterministic scorecard (JSON and CSV, with a hash manifest). The Markdown report and timing JSON include machine-dependent CPU time per frame.
  - Metrics: FP rate, early-alarm rate, recall, exact localization, mean IoU, and detection latency in steps.
  - `--verify` reruns the comparison and exits with code 4 if the hash differs.
- `examples/compare_scorecard.csv`; CI checks that it is reproduced byte for byte.

### Unchanged
- The v0.1 regimes, scoring, seeds and outputs. `examples/baseline_summary.csv` and `examples/sweep_summary.csv` are still reproduced byte for byte.

## [0.1.0] - 2026-09-30

First public release, as OES-Resilience. It is a rewrite of the single-file prototype
"OES-512 Telemetry Triage" 0.1.0 (kept in `original/`; see REVIEW.md for the review).

### Added
- OES32 reference detector: per-32-channel block score `0.45·max|x| + 0.35·RMS + 0.20·mean|x|` with threshold 0.50, fully vectorised.
- Four synthetic regimes (stable, noisy, localized_burst, global_shock), unchanged from the prototype.
- Keyed per-trial seeding (`SeedSequence(seed, spawn_key=(regime_key, trial))`), independent of trial count and regime order.
- Metrics with explicit per-regime expectations: false-positive rate (stable/noisy only), detection rate with Wilson 95% interval, exact match, IoU, localization error, block coverage, and status counts.
- Documented-expectation checks, with `run --strict` exiting with code 3 when a check fails.
- `sweep` subcommand that scores once and re-thresholds; its rows are identical to separate runs.
- Deterministic results JSON, summary CSV and SHA-256 manifest; non-deterministic run metadata is written to a separate file. Writes are atomic.
- CLI subcommands `run`, `sweep` and `test`, with input validation and exit codes 0/1/2/3. `pyproject.toml` provides the `oes-resilience` console script.
- pytest/unittest suite (72 tests, 99% coverage) and GitHub Actions CI on Python 3.10–3.13.

### Fixed (relative to the prototype)
- `global_shock` was reported as a 100% false-positive rate.
- A NaN threshold was accepted, which silently gave zero detections and wrote invalid JSON.
- Seeds overlapped between regimes and depended on the trial count.
- `Event.blocks` hardcoded a block size of 32.
- Averaging an empty list gave NaN plus a RuntimeWarning.
- Output hashes changed on every run.
- Non-atomic writes, a fixed manifest file name, and CSV columns taken from the first row.
