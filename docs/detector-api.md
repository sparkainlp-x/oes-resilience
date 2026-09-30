# Detector plugin API (v0.2.0; v0.3 and v0.4 additions marked)

OES-Resilience compares **block-level detectors**. A detector turns telemetry into one non-negative score per block, where a block is `block_size` consecutive channels (the default is 32, so a 512-channel frame has 16 blocks). A block is detected when `score >= threshold`. Everything is synthetic; nothing here connects to real sensors.

## Telemetry shapes

| input | meaning | frame detectors | temporal detectors |
|---|---|---|---|
| `(channels,)` | one frame | ✓ → scores `(blocks,)` | ✗ |
| `(n, channels)` | `n` independent frames, or one stream of `n` steps | ✓ → `(n, blocks)` | ✓ (one stream) → `(n, blocks)` |
| `(n, steps, channels)` | `n` streams | ✓, each step scored on its own → `(n, steps, blocks)` | ✓ → `(n, steps, blocks)` |

Inputs must not contain `inf`. NaN (missing channels) is accepted only by detectors that declare `supports_missing = True` (v0.3; see [Missing data](#missing-data-v03)); fit data must always be complete. `fit` takes `(n, channels)` clean frames for frame detectors, and one stream or `(n, steps, channels)` clean streams for temporal detectors.

## The `Detector` base class

```python
from oes_resilience import Config, Detector, register_detector

@register_detector
class PeakAbs(Detector):
    name = "peakabs"                      # registry key: lowercase letters, digits, '_', '-' or '+' (v0.3)
    rule = "score = max|x| per block"     # one-line, human-readable rule
    default_threshold = 0.5
    temporal = False                      # True if scores depend on earlier steps
    requires_fit = False                  # True if fit() must run before score()
    supports_missing = False              # v0.3: True only if the scoring handles NaN explicitly

    def _score_frames(self, frames):      # (n, channels) -> (n, blocks)
        return abs(self._blocks(frames)).max(axis=-1)

    def params(self):                     # optional: reported in the scorecard and explanations
        return {}

det = PeakAbs(Config())                   # or create_detector("peakabs", Config(), threshold=0.4)
result = det.fit(clean_frames).detect(frame)
result.scores, result.detected_blocks, result.status, result.explanation
```

The hooks you implement:

- `_score_frames(frames)`: required for frame detectors.
- `_score_streams(streams)`: required when `temporal = True`. It receives `(n, steps, channels)` and returns `(n, steps, blocks)`.
- `_fit(telemetry)`: optional. By default there is nothing to learn.
- `params()` and `explain(scores, detected, threshold)`: optional extensions of the explanation dictionary.

The base class validates shapes, rejects `inf` (and NaN unless `supports_missing`) and refuses to score an unfitted detector. It also rejects negative or non-finite scores, and it builds the `DetectionResult`.

`DetectionResult` fields:

| field | contents |
|---|---|
| `detector` | the detector's name |
| `threshold` | the threshold used |
| `scores` | per-block scores (float array) |
| `detected` | per-block booleans |
| `status` | per frame or step: `STABLE`, `LOCALIZED_ALERT`, `DISTRIBUTED_ALERT` or `GLOBAL_SHOCK`, using the same rule as v0.1 |
| `explanation` | rule, params, threshold, detected count, and the top block/score in the last frame |

`to_dict()` makes the result JSON-ready.

Temporal detectors can subclass `TemporalDetector`. It supplies `standardize(streams)`, which computes block means and z-scores them against each stream's own first `warmup` steps: a per-block mean and a standard deviation pooled across blocks. Scores during warm-up are 0.

## Missing data (v0.3)

Missing channels are represented as NaN. The rules:

- A detector with `supports_missing = False` (the default, and the optional `iforest`) raises `ValueError` on NaN input. The stress suite reports such detectors as "unsupported" for the dropout scenarios instead of filling the gaps.
- **No silent zero-filling.** The built-in mask-aware detectors compute every block statistic over the observed channels only, using `masked_block_stats(blocks)`, which returns `count`, `max_abs`, `rms`, `mean_abs` and `mean`:
  - `oes32`: `0.45·max|x| + 0.35·RMS + 0.20·mean|x|` over the observed channels of each block.
  - `zscore`: the chosen statistic over the observed channels.
  - `ewma`, `cusum`, `oes32+ewma`: block means over the observed channels. Each residual is multiplied by `sqrt(observed / block_size)`, so a block with fewer channels (noisier mean) is not over-weighted. The pooled warm-up sigma uses finite values only, and a block-step with no observed channel contributes `z = 0`.
- **A fully missing block scores 0** and therefore cannot alarm. That shows up as lost recall, not as a hidden fill.
- **Fast path.** When the input contains no NaN, every detector takes its v0.2 code path, so complete-data outputs are unchanged. The tests check that the masked temporal path equals the fast path on complete data.

## Registry and third-party plugins

`register_detector`, `unregister_detector`, `available_detectors()`, `get_detector_class(name)` and `create_detector(name, config, **kwargs)` manage the in-process registry. A different class cannot take a name that is already registered unless you pass `replace=True`.

An installed package can expose detectors through an entry point:

```toml
[project.entry-points."oes_resilience.detectors"]
peakabs = "my_package.detectors:PeakAbs"
```

`oes-resilience compare --plugins --detectors oes32,peakabs` calls `load_plugins()`, which imports those classes and registers them. A plugin that fails to load is skipped with a `RuntimeWarning`. Plugins run arbitrary code at import time, so install only the ones you trust.

## Built-in detectors

| name | type | rule | fitted on |
|---|---|---|---|
| `oes32` | frame | `0.45·max|x| + 0.35·RMS + 0.20·mean|x|` (the v0.1 reference; scores identical to `core.score_signals`) | nothing |
| `maxabs` (v0.4) | frame | `max|x|` per block (the pilot replay harness's baseline; the example above as a built-in); NaN-aware | nothing |
| `zscore` | frame | `|s − median_b| / (1.4826·MAD_b)` with `s` = block RMS (options: `statistic="mean"` or `"mean_abs"`) | clean frames |
| `ewma` | temporal | EWMA of the standardised block mean, `λ = 0.2`; score `|e_t| / sqrt(λ/(2−λ))` | nothing (warm-up baseline per stream) |
| `cusum` | temporal | two-sided tabular CUSUM of the standardised block mean, `k = 0.5` | nothing (warm-up baseline per stream) |
| `oes32+ewma` (v0.3) | temporal | `max(oes32 / c_oes32, ewma / c_ewma)`; the alarm fires if either part exceeds its share, and warm-up scores are 0 | clean streams: `c` = 99th percentile (`balance_quantile`) of each part's per-stream maximum after warm-up |
| `iforest` | frame, optional | `−IsolationForest.score_samples` on [mean, std, max|x|, RMS] per block | clean frames; needs `pip install "oes-resilience[iforest]"` |

## How `compare` treats every detector the same way

1. **Fit.** Detectors that need fitting use the same clean data from a seed stream that doesn't overlap evaluation: 1000 stable plus 1000 noisy frames for frame detectors, or `fit_streams` (default 200, v0.3; `--fit-streams`) stable plus noisy streams for temporal detectors.
2. **Calibrate.** Each detector's threshold comes from a third, held-out clean set. For each clean regime, the threshold is the smallest value that at most `floor(target_fp · n)` calibration units reach, where a unit's score is its maximum block score. A unit is one frame on the frame track, or a whole stream after warm-up on the stream track. The final threshold is the maximum over the clean regimes, so stable and noisy are each held to the target (default 1%). The thresholds and the calibration-set FP rates are written into the scorecard.
3. **Evaluate.**
   - The frame track uses the exact v0.1 trials.
   - The stream track uses separate seeds; alarms during warm-up are ignored for every detector.
   - OES32 also gets a row at its fixed v0.1 threshold 0.50 (`oes32@0.50`), which reproduces v0.1.

Stream metrics:

| metric | definition |
|---|---|
| FP | clean streams with any alarm after warm-up |
| early alarm | event streams with an alarm between warm-up and onset |
| recall | an alarm on an expected block at or after onset |
| latency | first such step minus onset |
| exact / IoU | the blocks alarmed at the first alarm step at or after onset, compared with the expected blocks |

CPU time is measured with `time.process_time` around `score()` calls on the evaluation data. It depends on the machine, so it is kept out of the hashed scorecard.

## The `oes32+ewma` hybrid (v0.3)

The hybrid is a `TemporalDetector` with `requires_fit = True`:

- **Fit.** `fit` scores the clean streams with OES32 (warm-up zeroed) and with EWMA, and stores each part's `balance_quantile` (0.99) quantile of the per-stream maximum as its scale.
- **Score.** The hybrid score is the larger of the two normalised part scores. One threshold on it is therefore an OR of the two parts.
- **Calibration.** `compare` and `stress` calibrate that one threshold like any other detector, so the union alarm spends a single FP budget. With the default settings the calibrated threshold is 1.0205 (calibration FP: stable 0.009, noisy 0.010).
- **Explanation.** `explain` lists the two parts.

Whether the hybrid helps is reported in the README, with measured numbers: it does not beat EWMA alone on this benchmark.
