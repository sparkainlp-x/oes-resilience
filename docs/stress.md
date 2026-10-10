# Stress suite (v0.3.0)

`oes-resilience stress` measures how detectors that are already calibrated hold up under seeded synthetic perturbations. Everything is synthetic; the results describe the detectors' behaviour on these generators, not on any real system.

## Protocol

1. **Fit and calibrate exactly as `compare` does.** Same seeds, same clean stable and noisy data, same target FP (default 1% per clean regime). The thresholds are **not** re-tuned per scenario. The tests check that `stress` and `compare` produce identical thresholds for identical settings.
2. **Generate each scenario from its own seed stream.** The seed is `SeedSequence(seed, spawn_key=(300, track, scenario, index))`, which is disjoint from the v0.1 trials, the compare streams and the fit and calibration data. Trial *i* does not depend on how many trials run.
3. **Two tracks.**
   - *Frame*: one 512-channel frame per trial, with the event present (onset 0). Only frame detectors apply.
   - *Stream*: 64 steps, warm-up 16, onset drawn uniformly from 24–48, as in `compare`. Alarms during warm-up are ignored.
4. **Score each row** (one row per track, detector and scenario):
   - Background scenarios: FP = share of units with any alarm (after warm-up).
   - Event scenarios:
     - recall = share of units with an alarm on a true block at or after onset;
     - also early-alarm rate, exact localization at the first post-onset alarm, IoU, and median latency in steps.
5. **Robustness** against the same detector's clean reference on the same track:
   - `fp_inflation` = FP − FP(`clean`);
   - `recall_retention` = recall / recall(`clean_burst`) (empty if the reference recall is 0);
   - `latency_delta` = median latency − reference median latency;
   - `change` = `worse` or `better` only if the 95% Wilson intervals of the stressed and reference rates do not overlap, otherwise `n.s.`. This is conservative: some real differences will show as `n.s.`.

   A retention ratio is hard to read when the reference recall is small (it can exceed 1 by a large factor), so use it together with the intervals.

## Scenarios

Parameters are `StressParams` fields and CLI flags; the defaults are shown in parentheses.

| scenario | kind | tracks | what it does | truth |
|---|---|---|---|---|
| `clean` | background | both | N(0, 0.05) on every channel | none |
| `clean_burst` | event | both | clean + N(`burst_mean` (0.8, the v0.1 value), 0.15) on one random block from onset | that block |
| `heavy_tail` | background | both | Student-t background with `t_df` (3) degrees of freedom, scaled to std 0.05 when df > 2 (scale 0.05 otherwise) | none |
| `heavy_tail_burst` | event | both | heavy-tailed background + burst | burst block |
| `dropout` | background | both | `dropout_fraction` (0.10) of channels are NaN for the whole stream | none |
| `dropout_burst` | event | both | missing channels + burst (the missing channels may fall inside the burst block) | burst block |
| `saturated` | background | both | `saturated_fraction` (0.02) of channels stuck at `+clip_level` (0.6) | none |
| `clipped_burst` | event | both | burst, then every value clipped to `±clip_level` | burst block |
| `narrow_burst` | event | both | burst on `narrow_width` (8) contiguous channels inside one block | that block |
| `cross_block_burst` | event | both | block-wide (32-channel) burst starting mid-block, so it covers half of two adjacent blocks | both blocks |
| `global_shift` | background | both | `+global_offset` (0.25) on every channel | none |
| `global_shift_burst` | event | both | global offset + burst | burst block only |
| `correlated_burst` | event | both | one common amplitude per step, N(`correlated_mean` (0.6), 0.1), added to `correlated_blocks` (3) adjacent blocks | those blocks |
| `drift` | event | stream | `drift_slope · (t − onset + 1)` (0.001 per step) on one block | that block |
| `baseline_step` | event | stream | `+step_size` (0.10) on every channel from onset | all blocks |
| `regime_steps` (v0.6) | background | stream | four clean segments with offsets 0.00, +0.20, −0.15, and +0.10, alternating from segment to segment | none; all steps are treated as baseline |

The references for background scenarios and burst-type scenarios (`clean`, `clean_burst`) are always evaluated. `drift` and `baseline_step` have no clean counterpart and are reported in absolute terms. `regime_steps` is a deterministic background-only probe for how existing detectors behave through alternating operating-level changes; its alarm rates are synthetic and are not evidence that a model has identified a benign regime change.

### Adaptive-threshold convergence (v0.6)

When `cusum-cp` is selected and `regime_steps` is included, the scorecard also streams the detector's maximum block score through `POTThreshold`. POT is calibrated once on held-out clean synthetic streams using post-warmup windows that match the regime-segment length, then runs across each regime-step stream without oracle resets. The known step boundaries are used only to define measurement windows. A transition is counted as converged when the threshold is within 10% of the matched clean reference threshold for three consecutive unflagged scores before the next transition. The report includes the number and rate of converged transitions and median/p90 convergence steps. This measures the combined synthetic `cusum-cp` + POT policy; it is not an online guarantee or a real-telemetry result.

## Missing data

Dropout is NaN, never zero. See [detector-api.md](detector-api.md#missing-data-v03) for the mask-aware scoring rules. A detector without `supports_missing` gets `supported = false` rows with empty metrics.

## Commands

```bash
oes-resilience stress --verify                                  # default suite (examples/stress_scorecard.csv)
oes-resilience stress --burst-mean 0.3 --correlated-mean 0.3 --prefix stress_burst03   # weaker events
oes-resilience stress --scenarios drift --drift-slope 0.0005 --tracks stream
oes-resilience stress --detectors oes32,iforest --tracks frame     # needs the iforest extra
```

Outputs (in `--out-dir`, default prefix `oes_resilience_stress`):
- deterministic, hashed in the manifest: `<prefix>_scorecard.json` and `<prefix>_scorecard.csv`;
- not deterministic (timings, metadata): `<prefix>_scorecard.md`, `<prefix>_timing.json` and `<prefix>_run_metadata.json`.

`--verify` recomputes the scorecard and exits with code 4 if the hash differs.

## Library use

```python
from oes_resilience import StressConfig, StressParams, run_stress
from oes_resilience.scorecard import CompareConfig

sc = StressConfig(compare=CompareConfig(streams=200, detectors=("ewma", "cusum")),
                  params=StressParams(t_df=5), scenarios=("heavy_tail", "drift"))
scorecard, timing = run_stress(sc)
```
