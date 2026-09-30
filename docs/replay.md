# Replay evaluation (v0.4.0)

`oes-resilience replay` scores a **recorded or synthetic telemetry replay** (JSONL, one 512-channel frame per line) against a **preregistration**: a JSON file, written and hashed *before* the replay is scored, that fixes the detectors, the scoring rules and how each threshold is set. The same detectors from the [plugin API](detector-api.md) are used (`oes32`, `maxabs`, `zscore`, `ewma`, `cusum`, `oes32+ewma`, and optional or plugin detectors). `oes32` block scores come from `core.score_signals`, the v0.1 reference formula, with no reimplementation.

This is an offline evaluation. It is not an alarm, a monitoring system or a process-control function.

The module comes from the private `oes512-replay-eval` pilot harness, including its input-hardening fixes. With the `examples/replay_prereg_fixed050.json` preregistration it reproduces that harness's metrics exactly.

## Replay format

Each line is one JSON object with exactly these keys:

| key | type | rule |
|---|---|---|
| `timestamp` | string | ISO-8601 **with a timezone**; normalised to UTC; strictly increasing |
| `channels` | array of 512 numbers | finite; booleans, `NaN`, `Infinity` and oversized integers are rejected |
| `regime` | string | non-empty, no leading or trailing whitespace; metrics are grouped by regime |
| `event_label` | string | `"none"` for normal frames |
| `event_id` | string or null | null exactly when `event_label` is `"none"`; each event must be contiguous and stay in one regime |
| `affected_blocks` | array of block indices 0–15 | non-empty for event frames, empty otherwise; no duplicates |
| `incumbent_alarm`, `incumbent_flags` | optional | an existing system's frame alarm and per-block flags; if given, they must be on every line, and they are scored as the `incumbent` column |

Lines are split on `\n` only (U+2028 inside strings is data). CRLF is accepted. Blank lines, duplicate keys, unknown keys, deeply nested JSON and invalid UTF-8 are rejected with exit code 2. Validation fails closed and never guesses.

## Preregistration

**Schema 2 (recommended).** Lists the detectors and a `threshold_policy`:

```json
{
  "schema_version": 2,
  "protocol_id": "...",
  "locked_at": "2026-09-30T18:00:00-04:00",
  "detectors": ["oes32", "maxabs", "zscore", "ewma", "cusum", "oes32+ewma"],
  "warmup": 16,
  "threshold_policy": {"mode": "calibrated", "target_fp": 0.01, "unit": "frame",
                       "calibration_sha256": "<sha256 of the calibration replay>"},
  "score_spec": {"channel_count": 512, "block_size": 32,
                 "weights": {"peak_abs": 0.45, "rms": 0.35, "mean_abs": 0.2}, "comparison": ">="},
  "event_hit_rule": "any_alert_frame_during_event",
  "localization_rule": "framewise_micro_precision_recall_iou",
  "latency_rule": "first_alert_frame_minus_first_event_frame_ms"
}
```

`threshold_policy` is either:

- `{"mode": "calibrated", ...}` (**the default and recommended mode**), or
- `{"mode": "fixed", "thresholds": {"<detector>": <number>, ...}}`, which must give one threshold for every detector.

**Schema 1** (the pilot harness's format) is still accepted. `candidate_threshold` becomes the fixed `oes32` threshold and `baseline_max_abs_threshold` becomes the fixed `maxabs` threshold.

The score spec and the three rules are locked. A preregistration that changes them is rejected rather than silently evaluated under different rules.

## How preregistered thresholds relate to the shared ~1% FP calibration

The benchmark's rule everywhere (`compare`, `stress`) is that **every detector gets the same false-positive budget on held-out clean data**, which is 1% by default. A fixed, equal *numeric* threshold does not do that, because the detectors' scores are on different scales. For example, 0.50 is a reasonable value for `oes32`, but for `maxabs` it sits inside the Noisy regime's normal range, and for `ewma` it is meaningless.

The replay therefore offers two policies:

1. **Calibrated (default).** The preregistration names the calibration replay by SHA-256 and the target FP rate. The replay tool:
   - refuses to run if the supplied `--calibration` file has a different hash, or if it contains any event frame;
   - scores each calibration frame by its maximum block score, skipping the first `warmup` frames for temporal detectors;
   - for each regime, finds the smallest threshold that at most `floor(target_fp · n)` frames reach, and uses the maximum over regimes. This is exactly `scorecard.calibrate_threshold`, the rule used by `compare` and `stress`, applied with frame units.

   Detectors that need fitting (`zscore`, `oes32+ewma`, `iforest`) are fitted on the same calibration replay. The thresholds are therefore fixed by the preregistration and the calibration data *before* the evaluation replay is scored. They are "preregistered" in the sense that the procedure and its input are locked, not the numbers.
2. **Fixed.** The preregistration writes the numbers down. This reproduces the pilot's "both detectors at 0.50" setup. The report sets `comparison_is_calibrated: false` and carries a `fairness_note`, and the CLI prints it: *equal numeric thresholds on detectors with different score scales do not make a fair comparison.* Use fixed mode to check a preregistered operating point, not to rank detectors.

Two caveats apply to calibrated mode:

- **The ≤1% FP holds on the calibration replay by construction, not on the evaluation replay.** In the example, `oes32` has 1 FP frame out of 60 in Noisy (1.7%) and `zscore` has 4 out of 60 (6.7%). Short replays give coarse and noisy FP estimates.
- **Temporal detectors see the whole replay as one stream.** They take their baseline from its first `warmup` frames. A regime that is labelled normal but shifts the mean (the example's Noisy regime, mean 0.25) looks like a sustained change to `ewma` and `cusum`. That inflates their calibrated thresholds and produces FP frames after events. In the example, CUSUM's calibrated threshold (about 8.6·10³) misses every event. This is a real limitation of applying stream detectors to a regime-segmented replay, and it is reported rather than tuned away.

## Outputs

`--out-dir` receives `{prefix}_report.json`, `{prefix}_scorecard.csv`, `{prefix}_manifest.json` (SHA-256 of each output) and `{prefix}_run_metadata.json`. The report records the SHA-256 of the evaluation replay, the calibration replay and the preregistration, together with the policy and the per-regime calibration details. `--verify` recomputes everything and exits with code 4 if any byte differs.

For each detector and regime, the CSV reports:

- event recall and missed event IDs;
- FP frame count and rate on no-event frames, plus FP frames and alarm episodes per 10 minutes;
- framewise micro block-localisation precision, recall and IoU;
- mean and median detection latency.

## Examples (SYNTHETIC)

```bash
oes-resilience replay --write-example-inputs /tmp/rp      # deterministic; prints the SHA-256 of both files
oes-resilience replay --input /tmp/rp/synthetic_replay.jsonl --prereg examples/replay_prereg_calibrated.json \
    --calibration /tmp/rp/synthetic_calibration.jsonl --out-dir out --prefix replay --verify
oes-resilience replay --input /tmp/rp/synthetic_replay.jsonl --prereg examples/replay_prereg_fixed050.json \
    --out-dir out --prefix replay_fixed050 --verify
```

- **Evaluation replay:** 240 frames, 60 in each regime (Stable, Noisy, Localized Burst with two single-block bursts, Global Shock with one shock). It uses seed 20260930 and is byte-identical to the pilot harness's data.
- **Calibration replay:** 600 event-free frames (Stable and Noisy), seed 20261001.

CI regenerates both files (about 1 MB each, not committed) and checks `examples/replay_scorecard.csv` and `examples/replay_fixed050_scorecard.csv` byte for byte on Python 3.10–3.13. They are also byte-identical across NumPy 1.26–2.5.

## Real replays: provenance rule

No real telemetry has been replayed. Every number in this repository is **SYNTHETIC**. Results on real data are **UNRUN**.

Before publishing any result from a real replay, record:

- the source system and who exported it;
- the export time window;
- any transformation from raw data to the 512-channel frames;
- who labelled the events, and how;
- the SHA-256 of the replay, the calibration replay and the preregistration;
- evidence that the preregistration was locked before the replay was scored.

Without that provenance, a replay result is an anecdote and not evidence. Even with it, a replay result describes that replay only.
