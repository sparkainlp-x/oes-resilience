# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
