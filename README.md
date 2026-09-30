# OES-Resilience

**Benchmarking telemetry intelligence under real-world stress**: an open, reproducible benchmark for multichannel telemetry anomaly detection, with **OES32** as its transparent reference detector.

[![tests](https://github.com/sparkainlp-x/oes-resilience/actions/workflows/tests.yml/badge.svg)](https://github.com/sparkainlp-x/oes-resilience/actions/workflows/tests.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#what-it-is-not)

Version 0.1.0 is the first public release. It ships the benchmark harness, four **synthetic** signal regimes and the OES32 reference detector. It depends only on NumPy. Stress tests on real-world data are on the [roadmap](#roadmap); they are not in this release.

## What it is

- **A reproducible benchmark harness.** Each trial generates a synthetic 512-channel signal and splits it into sixteen 32-channel blocks. The harness asks whether a detector flags the right blocks and reports false-positive, detection, exact-match, overlap (IoU) and localization metrics.
- **A transparent reference detector (OES32).** Every 32-channel block gets the score `0.45·max|x| + 0.35·RMS(x) + 0.20·mean|x|`. A block is flagged when its score is `>= threshold` (default 0.50). There are no learned parameters, and the whole rule fits on one line.
- **Deterministic by construction.** Trial *i* of each regime is seeded with `numpy.random.SeedSequence(seed, spawn_key=(regime_key, i))`. The results JSON, summary CSV and manifest are byte-for-byte reproducible, and run metadata (time, platform) goes in a separate file.
- **A single Python file** (`oes_resilience.py`) with a CLI (`run`, `sweep`, `test`) and a tested library API.

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

## Install and run

Requires Python 3.10 or newer and NumPy.

```bash
python -m pip install .                  # installs the `oes-resilience` command
oes-resilience run                       # 1000 trials/regime, seed 42, threshold 0.50
oes-resilience run --threshold 0.6 --trials 5000 --out-dir outputs --strict
oes-resilience sweep                     # thresholds 0.30..1.00, step 0.05
oes-resilience sweep --thresholds 0.45,0.5,0.55
oes-resilience --help
```

Without installing, run `python oes_resilience.py …` instead. Options shared by `run` and `sweep` are `--trials`, `--seed`, `--channels`, `--block-size`, `--global-fraction` (the share of blocks needed for `GLOBAL_SHOCK` status; default 1.0, meaning all), `--out-dir`, `--prefix` and `--quiet`. `run` also takes `--no-trials`, which leaves out the per-trial records.

**Outputs** (written atomically to `--out-dir`, default `outputs/`):

| file | deterministic? | contents |
|---|---|---|
| `<prefix>_results.json` | yes | config, per-regime summaries, expectation checks, per-trial records |
| `<prefix>_summary.csv` | yes | one row per regime (per threshold for sweeps), fixed columns |
| `<prefix>_manifest.json` | yes | SHA-256 of the two files above |
| `<prefix>_run_metadata.json` | no | UTC time, Python, platform, NumPy, elapsed time, argv |

**Exit codes:** `0` ok · `1` test failure or I/O error · `2` invalid arguments or configuration · `3` `run --strict` and an expectation check failed.

**Library use:** `assess_signal(signal, Config())` scores and classifies any 512-value array. `run_benchmark` and `threshold_sweep` return JSON-ready dictionaries.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest --cov        # 72 tests (+56 subtests); coverage gate 90% (currently 99%)
python oes_resilience.py test # stdlib unittest runner, no pytest needed (source checkout; else pass --tests-dir)
```

CI runs `ruff check`, pytest with the coverage gate, a byte-for-byte check of the example CSVs, and a pip-install smoke test on Python 3.10–3.13. The tests cover the scoring formula worked by hand, block mapping for arbitrary block sizes, seed stability, validation errors, metric edge cases, the sweep matching separate runs, reproducible exports and the CLI exit codes. [REVIEW.md](REVIEW.md) is the review of the original prototype (kept in [`original/`](original/)) that this release is based on.

## Roadmap

- **v0.2:** a detector plugin API; z-score and EWMA/CUSUM baseline detectors; a comparative scorecard.
- **v0.3:** stress tests for drift, channel dropout and heavy-tailed noise.
- **v0.4:** adapters for the public NASA SMAP/MSL telemetry anomaly datasets.

Planned items are not implemented yet, and no results are claimed for them.

## Related work

- **[oes32-residual](https://github.com/sparkainlp-x/oes32-residual)** ([DOI 10.5281/zenodo.22985521](https://doi.org/10.5281/zenodo.22985521)) is the normative OES-32 residual reference: a deterministic max-absolute residual over two 32-component vectors with a strict tolerance rule. OES-Resilience applies the same 32-channel block granularity in a benchmark setting.
- **[signal-commons](https://github.com/sparkainlp-x/signal-commons)** is an offline proof of concept that reduces 512-channel residual frames (16 × 32 groups) to coarse categorical incident postcards for sharing without raw data.

## Citation

See [CITATION.cff](CITATION.cff). A DOI will be added once the repository is archived on Zenodo.

## License

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).

Organizations that want to use it in proprietary products or services without AGPL obligations can contact the author about a commercial license via https://sparkainlpx.xyz. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).
