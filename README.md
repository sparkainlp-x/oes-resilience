# OES-Resilience

**Benchmarking telemetry intelligence under real-world stress**: an open, reproducible benchmark for multichannel telemetry anomaly detection, with **OES32** as its transparent reference detector.

[![tests](https://github.com/sparkainlp-x/oes-resilience/actions/workflows/tests.yml/badge.svg)](https://github.com/sparkainlp-x/oes-resilience/actions/workflows/tests.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23071166.svg)](https://doi.org/10.5281/zenodo.23071166)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#what-it-is-not)

Version 0.1.0 is the first public release. It ships the benchmark harness, four **synthetic** signal regimes and the OES32 reference detector. It depends only on NumPy. Stress tests on real-world data are on the [roadmap](#roadmap); they are not in this release.

> **This is the `v0.2-dev` branch (0.2.0, unreleased).** It adds a detector plugin API, robust z-score, EWMA and CUSUM baselines (plus an optional Isolation Forest), multi-step synthetic streams and a calibrated [detector comparison](#detector-comparison-v02-in-development). The v0.1 results below are unchanged.

## What it is

- **A reproducible benchmark harness.** Each trial generates a synthetic 512-channel signal and splits it into sixteen 32-channel blocks. The harness asks whether a detector flags the right blocks and reports false-positive, detection, exact-match, overlap (IoU) and localization metrics.
- **A transparent reference detector (OES32).** Every 32-channel block gets the score `0.45·max|x| + 0.35·RMS(x) + 0.20·mean|x|`. A block is flagged when its score is `>= threshold` (default 0.50). There are no learned parameters, and the whole rule fits on one line.
- **Deterministic by construction.** Trial *i* of each regime is seeded with `numpy.random.SeedSequence(seed, spawn_key=(regime_key, i))`. The results JSON, summary CSV and manifest are byte-for-byte reproducible, and run metadata (time, platform) goes in a separate file.
- **A small Python package** (`oes_resilience/`) with a CLI (`run`, `sweep`, `compare`, `test`), a documented [detector plugin API](docs/detector-api.md) and a tested library API. NumPy is the only required dependency.

## What it is not

- **Not quantum.** No qubits, quantum hardware or quantum error correction are involved.
- **Not GPS or navigation.** It does not estimate position, timing or trajectories.
- **Not a physical sensor or device.** It reads no hardware, and all signals are generated synthetically by NumPy.
- **Not a medical, safety or certified system,** and not an operational monitoring product.
- **Not evidence about real telemetry.** The numbers below describe how this software behaves on its own synthetic regimes. They are not performance claims about any real system or dataset.

## Synthetic regimes (v0.1.0)

| regime | how each signal is generated | expected OES32 behaviour |
|---|---|---|
| `stable` | N(0, 0.05) on all 512 channels | no blocks flagged |
| `noisy` | N(0.25, 0.10) on all channels | no blocks flagged |
| `localized_burst` | `stable`, plus N(0.80, 0.15) added to one random 32-channel block | exactly the injected block flagged |
| `global_shock` | N(2.0, 0.5) on all channels | most or all blocks flagged |

## Measured results (seed 42, 1000 trials per regime, threshold 0.50)

Command: `oes-resilience run --trials 1000 --seed 42 --threshold 0.50 --strict` (exit 0). The summary is committed as [`examples/baseline_summary.csv`](examples/baseline_summary.csv), and CI checks that it is reproduced byte for byte.

| regime | false-positive rate | detection rate | exact match | mean detected blocks | per-trial max-score range |
|---|---|---|---|---|---|
| stable | 0.000 (0/1000; 95% upper bound 0.38%) | 0.000 | 1.000 | 0.00 | 0.080 – 0.143 |
| noisy | 0.000 (0/1000; 95% upper bound 0.38%) | 0.000 | 1.000 | 0.00 | 0.361 – 0.471 |
| localized_burst | n/a | 1.000 | 1.000 | 1.00 | 0.822 – 1.153 |
| global_shock | n/a | 1.000 | 1.000 (all 16 blocks) | 16.00 | 2.524 – 3.211 |

All five expectation checks pass at 1000 trials. These checks are software-behaviour thresholds defined in the code: false positives ≤ 1% for stable and noisy, burst detection ≥ 99% and exact match ≥ 95%, and a majority of blocks in ≥ 99% of shock trials.

**The noisy margin is thin.** The largest noisy block score in those 1000 trials was 0.471. In a 100,000-trial run at threshold 0.50, 13 noisy trials crossed it, a false-positive rate of **0.013%** (95% upper bound 0.022%; the largest noisy score was 0.514). The rate rises quickly below 0.50: 0.26% at 0.47, 0.67% at 0.46 and 1.5% at 0.45. So "no detections" is what normally happens, not a guarantee.

### Threshold sweep 0.30–1.00 (step 0.05, 1000 trials)

Command: `oes-resilience sweep`. Full table: [`examples/sweep_summary.csv`](examples/sweep_summary.csv).

| threshold | noisy FP | burst detection | burst exact match | all checks pass |
|---|---|---|---|---|
| 0.30 | 1.000 | 1.000 | 1.000 | no |
| 0.35 | 1.000 | 1.000 | 1.000 | no |
| 0.40 | 0.487 | 1.000 | 1.000 | no |
| 0.45 | 0.014 | 1.000 | 1.000 | no |
| 0.50 – 0.80 | 0.000 | 1.000 | 1.000 | yes |
| 0.85 | 0.000 | 0.990 | 0.990 | yes (at the boundary) |
| 0.90 | 0.000 | 0.888 | 0.888 | no |
| 0.95 | 0.000 | 0.509 | 0.509 | no |
| 1.00 | 0.000 | 0.138 | 0.138 | no |

The stable false-positive rate is 0 and the shock regime flags all 16 blocks at every threshold in the sweep. The default of 0.50 sits at the low edge of the passing range (0.50–0.85).

## Detector comparison (v0.2, in development)

`oes-resilience compare` runs every detector through the same fit, calibration and evaluation steps and writes a scorecard (JSON, CSV and Markdown). The [detector API guide](docs/detector-api.md) has the full method; in short:

- **Fit.** Detectors that learn a baseline (`zscore`, `iforest`) are fitted on 1000 stable plus 1000 noisy frames.
- **Calibrate.** Every threshold is calibrated the same way on a separate, held-out clean set: at most 1% of stable and at most 1% of noisy calibration units may reach it.
- **Evaluate.** The frame track uses the exact v0.1 trials. The stream track uses 1000 streams per regime, each 64 steps long with 16 warm-up steps and the event onset at step 24–48.
- **Reproducibility.** The scorecard JSON/CSV hash matched across reruns (`--verify`) and was identical on Python 3.10/NumPy 1.26 and Python 3.13/NumPy 2.2. The committed [`examples/compare_scorecard.csv`](examples/compare_scorecard.csv) is checked byte for byte in CI.

**Frame track** (v0.1 regimes, 1000 trials each). All rows except `oes32@0.50` use calibrated thresholds:

| detector | threshold | stable FP | noisy FP | burst recall / exact / IoU | shock recall / exact |
|---|---|---|---|---|---|
| oes32 (calibrated) | 0.4543 | 0.000 | 0.009 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 |
| oes32@0.50 (v0.1 reference) | 0.50 | 0.000 | 0.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 |
| zscore | 1.2258 | 0.000 | 0.006 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 |

The frame regimes are easy: every frame detector separates them perfectly once calibrated. They differ only in how many noisy frames cross the threshold, and that is set by the calibration target.

**Stream track** (1000 streams per regime; FP counts any alarm on a clean stream after warm-up; latency is measured in steps after onset):

| detector | stable FP | noisy FP | burst recall (latency) | shift recall / exact | shift latency mean / median / p90 | shock recall (latency) |
|---|---|---|---|---|---|---|
| oes32 | 0.000 | 0.012 | 1.000 (0) | 0.000 / 0.000 | – | 1.000 (0) |
| zscore | 0.000 | 0.007 | 1.000 (0) | 0.000 / 0.000 | – | 1.000 (0) |
| ewma | 0.007 | 0.008 | 1.000 (0) | 0.987 / 0.985 | 6.99 / 6 / 13 | 1.000 (0) |
| cusum | 0.007 | 0.003 | 1.000 (0) | 0.954 / 0.954 | 14.21 / 14 / 19 | 1.000 (0) |

What the comparison shows:

- **OES32 loses on the small persistent shift.** In `stream_shift`, one block gets a constant +0.02 (0.4 standard deviations per channel). OES32 and the frame z-score miss it completely. EWMA finds it in 98.7% of streams (median 6 steps) and CUSUM in 95.4% (median 14 steps).
- **Bursts and shocks are easy for everyone.** All four detectors catch them at the onset step, with exact localization.
- **CUSUM is slower than EWMA here.** Its warm-up baseline comes from only 16 steps, so the calibrated threshold ends up high (25.9).
- **The shift regime was designed for this.** It is a deliberately chosen case that single-frame detectors are not built for, so it is not evidence of general superiority either way.
- **FP rates on held-out data scatter around the 1% target** (e.g. OES32 1.2% and CUSUM 0.3% on noisy streams). Each rate comes from 1000 streams; the Wilson intervals are in the CSV.

**CPU time per frame** is measured with `time.process_time` on one machine and depends on hardware and load, so it is not part of the hashed scorecard. On the development machine: OES32 about 4 µs per frame, z-score about 1 µs, EWMA and CUSUM about 0.9 µs per step.

**Optional Isolation Forest** (`pip install ".[iforest]"`, scikit-learn 1.9.1; not in the default run). On the frame track it matched the others: noisy FP 0.009, burst and shock recall and exact match 1.000. On streams it gave noisy FP 0.018, burst recall 1.000 (mean latency 0.09 steps) and shift recall 0.000. It cost about 60 µs per frame. Adding it did not change any other detector's rows.

## Install and run

Requires Python 3.10 or newer and NumPy.

```bash
python -m pip install .                  # installs the `oes-resilience` command
oes-resilience run                       # 1000 trials/regime, seed 42, threshold 0.50
oes-resilience run --threshold 0.6 --trials 5000 --out-dir outputs --strict
oes-resilience sweep                     # thresholds 0.30..1.00, step 0.05
oes-resilience sweep --thresholds 0.45,0.5,0.55
oes-resilience compare --verify          # detector scorecard (JSON, CSV, Markdown), hash re-check
oes-resilience compare --detectors oes32,ewma --tracks stream --streams 500
oes-resilience --help
```

Without installing, run `python -m oes_resilience …` from the repository root instead. Options shared by `run` and `sweep` are `--trials`, `--seed`, `--channels`, `--block-size`, `--global-fraction` (the share of blocks needed for `GLOBAL_SHOCK` status; default 1.0, meaning all), `--out-dir`, `--prefix` and `--quiet`. `run` also takes `--no-trials`, which leaves out the per-trial records.

**Outputs** (written atomically to `--out-dir`, default `outputs/`):

| file | deterministic? | contents |
|---|---|---|
| `<prefix>_results.json` | yes | config, per-regime summaries, expectation checks, per-trial records |
| `<prefix>_summary.csv` | yes | one row per regime (per threshold for sweeps), fixed columns |
| `<prefix>_manifest.json` | yes | SHA-256 of the two files above |
| `<prefix>_run_metadata.json` | no | UTC time, Python, platform, NumPy, elapsed time, argv |

`compare` writes `<prefix>_scorecard.json`, `<prefix>_scorecard.csv` and `<prefix>_manifest.json`, which are deterministic. It also writes `<prefix>_scorecard.md`, `<prefix>_timing.json` and `<prefix>_run_metadata.json`, which include machine-dependent timings. Its options include `--detectors`, `--tracks`, `--target-fp`, `--streams`, `--steps`, `--warmup`, `--plugins` (load entry-point detectors) and `--verify`.

**Exit codes:** `0` ok · `1` test failure or I/O error · `2` invalid arguments or configuration · `3` `run --strict` and an expectation check failed · `4` `compare --verify` found a hash mismatch.

**Library use:** `assess_signal(signal, Config())` scores and classifies any 512-value array. `run_benchmark`, `threshold_sweep` and `run_compare` return JSON-ready dictionaries. `create_detector("ewma").detect(stream)` returns a `DetectionResult`, and `register_detector` adds your own detector (see [docs/detector-api.md](docs/detector-api.md)).

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest --cov            # 114 tests (+76 subtests); coverage gate 90%
python -m oes_resilience test     # stdlib unittest runner, no pytest needed (source checkout; else pass --tests-dir)
```

CI runs `ruff check`, pytest with the coverage gate, a byte-for-byte check of the example CSVs (including the compare scorecard), and a pip-install smoke test on Python 3.10–3.13. A separate job installs scikit-learn and tests the optional Isolation Forest. The tests cover the scoring formula worked by hand, block mapping for arbitrary block sizes, seed stability, validation errors, metric edge cases, the sweep matching separate runs, reproducible exports and the CLI exit codes. [REVIEW.md](REVIEW.md) is the review of the original prototype (kept in [`original/`](original/)) that this release is based on.

## Roadmap

- **v0.2 (in development on `v0.2-dev`):** detector plugin API ✓; robust z-score, EWMA and CUSUM baselines ✓ (optional Isolation Forest ✓); calibrated comparative scorecard ✓. Not yet released.
- **v0.3:** stress tests for drift, channel dropout and heavy-tailed noise.
- **v0.4:** adapters for the public NASA SMAP/MSL telemetry anomaly datasets.

Items for v0.3 and v0.4 are not implemented yet, and no results are claimed for them.

## Related work

- **[oes32-residual](https://github.com/sparkainlp-x/oes32-residual)** ([DOI 10.5281/zenodo.22985521](https://doi.org/10.5281/zenodo.22985521)) is the normative OES-32 residual reference: a deterministic max-absolute residual over two 32-component vectors with a strict tolerance rule. OES-Resilience applies the same 32-channel block granularity in a benchmark setting.
- **[signal-commons](https://github.com/sparkainlp-x/signal-commons)** is an offline proof of concept that reduces 512-channel residual frames (16 × 32 groups) to coarse categorical incident postcards for sharing without raw data.

## Citation

See [CITATION.cff](CITATION.cff). The released v0.1.0 is archived on Zenodo: concept DOI [10.5281/zenodo.23071166](https://doi.org/10.5281/zenodo.23071166), covering all versions. This branch is unreleased work.

## License

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).

Organizations that want to use it in proprietary products or services without AGPL obligations can contact the author about a commercial license via https://sparkainlpx.xyz. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).
