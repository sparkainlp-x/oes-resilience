# Code review: OES-512 Telemetry Triage prototype (0.1.0) → OES-Resilience 0.1.0

Scope: this reviews the single-file prototype `oes512_prototype.py` ("OES-512 Telemetry Triage",
version 0.1.0, kept verbatim in `original/oes512_prototype_v0.1.0.py`). The rewrite is published as
**OES-Resilience 0.1.0** (`oes_resilience.py`); its block scorer is the "OES32" reference detector.
Below, "the prototype" means the original file and "OES-Resilience" means the rewrite. Everything here is about how the
software behaves on **synthetic** signals. Nothing in this review is a claim about physical
sensors, quantum hardware or real-world performance.

The documented baseline is kept exactly as it was: 512 channels, sixteen 32-channel blocks, the
four regimes and their distributions, the score `0.45·max|x| + 0.35·RMS + 0.20·mean|x|`, and the
threshold of 0.50. Given the same generator state, `generate_signal` makes the same draws in the
same order as the prototype. I re-scored all 4,000 trials from a prototype run with the new vectorised
scorer, and every score and every detected-block list matched exactly (0 mismatches).

## Issues found (severity: High / Medium / Low)

| # | Sev | Issue in the prototype | Fix in OES-Resilience |
|---|-----|-----------------|---------------|
| 1 | High | **`global_shock` was counted as a false positive.** Any regime that wasn't the burst got `false_positive = bool(detected)`, so the shock regime reported `false_positive_rate = 1.0`, the opposite of what it should mean. | Each regime now has an explicit expectation (`none` / `event_blocks` / `all_blocks`). `false_positive_rate` is only reported for stable and noisy. The shock regime reports `mean_block_coverage`, `majority_blocks_rate`, `all_blocks_rate`, exact match (all blocks) and IoU. |
| 2 | High | **A NaN threshold was accepted.** `NaN < 0` is False, so the check passed and every comparison came out False. Every regime, shock included, silently reported 0 detections, and the JSON had 22 `NaN` tokens, which isn't valid JSON. | `validate_threshold` requires a finite real number ≥ 0 that isn't a bool. It's used in `Config`, `detect`, the sweep and the CLI. JSON is written with `allow_nan=False`. |
| 3 | High | **The seed scheme depended on trial count and regime order.** `seed + regime_index·trials + i` meant that with `--trials 500`, the noisy regime used seeds 542–1041, which are the *stable* seeds of a 1000-trial run. Changing `--seed` by 1 shifted every stream by one trial. Adding or reordering regimes changed the seeds. | `SeedSequence(seed, spawn_key=(REGIME_KEYS[regime], trial_index))` (tested to equal `SeedSequence(seed).spawn(..)[k].spawn(..)[i]`). Regime keys are fixed constants rather than enum order. Trial *i* is the same whatever the trial count, chunking or threshold (tested). |
| 4 | High | **Results were not reproducible byte-for-byte.** `created_utc`, `python`, `platform` and `elapsed_seconds` were inside the results file, so the manifest hash changed on every run. | Deterministic `<prefix>_results.json` and `<prefix>_summary.csv`, and a deterministic `<prefix>_manifest.json` holding their SHA-256 hashes. Time, platform and argv go to a separate `<prefix>_run_metadata.json`. Verified: two runs give identical hashes. |
| 5 | Medium | `Event.blocks` hardcoded 32, and the true-block logic was repeated in three places (`Event.blocks`, `interval_iou`, `evaluate_trial`). With `block_size=16`, `Event(32,96).blocks` returned `(1, 2)` instead of `(2, 3, 4, 5)`. | There is now one `block_range(start, stop, block_size)`, and `Event.blocks(block_size)` uses it. It's tested against a brute-force channel→block mapping for random sizes. |
| 6 | Medium | `summarize_trials` took `np.mean([])` when no burst trial had a detection, which gave NaN plus a RuntimeWarning. The NaN then leaked into the JSON. | `_mean_or_none` returns `None`. `localization_error_trials` reports how many trials the mean is based on. There's a test that runs with warnings turned into errors. |
| 7 | Medium | The threshold sweep regenerated every signal for each threshold (15 thresholds × 4,000 trials). | Signals are scored once per regime, and each threshold only re-applies `scores >= t`. A test checks that the rows are identical to separate `run_benchmark` calls. Measured: the prototype took 12.98 s, OES-Resilience takes 0.16 s. |
| 8 | Medium | Scoring used a Python loop per block and per trial. | `score_signals` reshapes to `(n, blocks, block_size)` and uses axis reductions. Metrics (IoU, exact match, localization error via an O(n·blocks) distance transform, status) are all array operations. Trials are generated in chunks of 4096 to bound memory. Generation still uses one generator per trial on purpose, so every (regime, trial) can be reproduced on its own. A 1000-trial run went from 0.93 s to 0.16 s. |
| 9 | Medium | Weaker validation. Non-finite weights were only rejected because `isclose` happened to fail. Float or bool `channels`/`block_size` were accepted (`Config(channels=512.0, block_size=32.0).blocks == 16.0`, which later breaks `reshape`). Negative seeds were accepted and then failed inside NumPy. | Integer fields must be real integers (NumPy ints are fine, bools are not). Weights must be finite, ≥ 0 and sum to 1 (abs tol 1e-9). Seed must be ≥ 0. `global_fraction` must be in (0, 1]. |
| 10 | Medium | The CLI had no subcommands, no sweep, no `--trials` validation, and no exit codes. Errors produced a traceback with exit 1, and `--trials 0` printed a traceback. | Subcommands `run`, `sweep` and `test` (with `--test` kept as an alias; no subcommand means `run`). argparse validates `--trials` and `--threshold`. Exit codes: 0 ok, 1 test failure or I/O error, 2 invalid arguments or config (a one-line message, no traceback), 3 `run --strict` when an expectation check fails. |
| 11 | Low | Writes weren't atomic, `manifest.json` always had the same name (two runs in one directory overwrote each other's manifest), and the CSV header came from the first row. | `atomic_write` uses a temp file in the same directory, then fsync, then `os.replace`. It cleans up if something fails and keeps umask-based permissions. All output names come from `--prefix`. The CSV header is the fixed `SUMMARY_FIELDS`, unknown keys are rejected, and line endings are `\n`. |
| 12 | Low | Status rule: `GLOBAL_SHOCK` required all 16 blocks, while the documentation says the shock activates "most or all" blocks. With one block, a single detection was ambiguous. | The default is unchanged (all blocks, `global_fraction=1.0`) so baseline outputs keep their meaning. `--global-fraction` makes the rule configurable, and the priority order is documented (0 → STABLE, 1 → LOCALIZED, ≥ min → GLOBAL, otherwise DISTRIBUTED). Summaries always include `majority_blocks_rate` and `all_blocks_rate`. |
| 13 | Low | Tests: the built-in `--test` made a handful of weak asserts (e.g. shock `>= 1` block). | `tests/test_oes_resilience.py` has 72 tests plus 56 subtests and runs under both unittest and pytest. `oes-resilience test` / `python oes_resilience.py test` is a thin unittest-discovery wrapper. |
| 14 | Low | Minor: unused `Iterable` import. Stored scores were rounded to 10 dp but `maximum_score` wasn't. `threshold` was repeated in every trial row. Style was inconsistent (one-line `if`s, no type hints or docstrings). | Full type hints and docstrings. Stored floats are rounded consistently to 12 dp, while thresholding uses full precision. The threshold is stored once per summary. Every trial row gets `seed_key` and `expected_blocks`. Wilson 95% intervals are added for detection rates. |

## Measured baseline (seed 42, 1000 trials per regime, threshold 0.50)

`oes-resilience run --trials 1000 --seed 42 --threshold 0.50 --strict` → exit 0. The resulting summary is committed as `examples/baseline_summary.csv`.

| regime | FP rate | detection rate (any block) | exact match | mean detected blocks | max-score range (min–max) |
|---|---|---|---|---|---|
| stable | 0.000 (0/1000; 95% upper 0.38%) | 0.000 | 1.000 | 0.000 | 0.080 – 0.143 |
| noisy | 0.000 (0/1000; 95% upper 0.38%) | 0.000 | 1.000 | 0.000 | 0.361 – 0.471 |
| localized_burst | n/a | 1.000 (event detected) | 1.000 | 1.000 | 0.822 – 1.153 |
| global_shock | n/a | 1.000 | 1.000 (all 16) | 16.000 | 2.524 – 3.211 |

For the burst, mean IoU is 1.000 and localization error is 0.000. For the shock, all 1000 trials
have status `GLOBAL_SHOCK`.

**Does it match the documented expectations?** Yes, at 1000 trials. Stable and noisy had no
detections, the burst was located exactly in every trial, and the shock activated all 16 blocks
in every trial. All five expectation checks pass. The prototype run (with its different seed scheme)
gave the same headline numbers.

**Noisy margin, reported honestly.** The worry that noisy would often cross 0.50 does not hold up:
the largest noisy block score in 1000 trials was 0.471. The margin is small, though. At threshold
0.45 the noisy FP rate is 1.4% (1000 trials). A 100,000-trial tail check at 0.50
(`oes-resilience sweep --trials 100000 --thresholds 0.45,0.46,0.47,0.48,0.49,0.50`) found 13 false positives, i.e. **0.013%** (95% upper bound
0.022%), with a largest noisy score of 0.514. So "normally no detections" holds, but it is not
"never", and the FP rate rises quickly below 0.50 (0.47 → 0.26%, 0.46 → 0.67%, 0.45 → 1.5%).

## Threshold sweep 0.30–1.00 (step 0.05, 1000 trials, seed 42; `examples/sweep_summary.csv`)

| threshold | stable FP | noisy FP | burst detect | burst exact | burst mean blocks | shock mean blocks | shock all-16 rate | all checks pass |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | no |
| 0.35 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | no |
| 0.40 | 0.000 | 0.487 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | no |
| 0.45 | 0.000 | 0.014 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | no |
| 0.50 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | yes |
| 0.55 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | yes |
| 0.60 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | yes |
| 0.65 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | yes |
| 0.70 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | yes |
| 0.75 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | yes |
| 0.80 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 16.00 | 1.000 | yes |
| 0.85 | 0.000 | 0.000 | 0.990 | 0.990 | 0.990 | 16.00 | 1.000 | yes |
| 0.90 | 0.000 | 0.000 | 0.888 | 0.888 | 0.888 | 16.00 | 1.000 | no |
| 0.95 | 0.000 | 0.000 | 0.509 | 0.509 | 0.509 | 16.00 | 1.000 | no |
| 1.00 | 0.000 | 0.000 | 0.138 | 0.138 | 0.138 | 16.00 | 1.000 | no |

Highlights:
- **0.50–0.85 meets all the checks.** 0.85 only just passes (burst detection is exactly 0.990).
- **Below 0.50 the noisy regime fails.** It reaches 100% FP at ≤ 0.35 (15.6 of 16 blocks flagged on average at 0.30).
- **Above 0.85 the burst gets missed.** Burst detection falls to 88.8%, 50.9% and 13.8% at 0.90, 0.95 and 1.00.
- **Stable and shock don't change** anywhere in 0.30–1.00.
- **The default 0.50 is at the low edge of the passing range.** A value around 0.65 would sit in the middle. I have **not** changed the baseline default; that is the author's call.

The expectation checks are my way of turning the documented behaviour into numbers, as software
checks: FP ≤ 1% for stable and noisy, burst detection ≥ 99% and exact match ≥ 95%, and a majority
of blocks in ≥ 99% of shock trials. They are named constants in the file and easy to adjust.

## Tests

- `python -m pytest --cov` (config in `pyproject.toml`): **72 tests (+56 subtests), all pass. Coverage is 99%** (line and branch; the only line not covered is the `if __name__ == "__main__"` guard, which a subprocess test runs anyway). Checked locally on Python 3.13.5 / NumPy 2.2.4 and Python 3.10.21 / NumPy 1.26.4; the committed example CSVs are byte-identical on both; CI runs Python 3.10–3.13.
- Covered: determinism; the scoring formula checked by hand (`[3,-4,0,0]` → 3.025) and against the literal prototype loop for five block sizes; block mapping for arbitrary sizes (brute force); SeedSequence/spawn equivalence and independence from trial count and chunking; prototype draw-order compatibility; every validation error; metric edge cases (no detections → `None`, no warnings; multi-block truth; shock not counted as FP); the sweep matching separate runs; byte-for-byte reproducible results, summary and manifest; atomic-write rollback; CLI exit codes 0/1/2/3, both in-process and via subprocess.
