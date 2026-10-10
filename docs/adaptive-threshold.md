# Streaming POT threshold (v0.6.0)

`POTThreshold` estimates an upper-tail threshold for a **single scalar anomaly score**. It is intentionally separate from the benchmark's fixed-threshold `Detector` API: the frame/stream scorecards, hashed replay preregistrations, and v0.5 NASA SMAP/MSL protocol retain their frozen threshold rules.

## Example

```python
import numpy as np
from oes_resilience import OES32Detector, POTThreshold

# Calibration data must be known-clean and independent of evaluation.
clean_frames = np.random.default_rng(7).normal(0.0, 0.05, size=(2000, 512))
clean_scores = OES32Detector().score(clean_frames).max(axis=1)
threshold = POTThreshold(risk_level=0.01, init_quantile=0.98, window_size=1000)
threshold.fit_calibration(clean_scores)

for frame in incoming_frames:
    score = OES32Detector().score(frame).max()
    is_anomaly, used_threshold = threshold.update(float(score))
    # Persist or act on (is_anomaly, used_threshold) in the caller's application.
```

`update` returns the decision and **the threshold used for that decision**. It uses a strict `score > threshold` comparison. Flagged values are not added to the rolling calibration history; accepted values update the sliding-window fit. The implementation estimates GPD shape and scale by moments and uses an empirical `1-risk_level` quantile when there are too few tail exceedances. It needs only NumPy.

## Regime changes and contamination

A scalar score alone cannot reliably distinguish a sustained anomaly from a legitimate change in operating regime. POT therefore masks flagged observations rather than silently learning from them. If an independently validated change-point detector establishes a clean new regime, call `reset(clean_scores_from_new_regime)` using a separate, known-clean calibration sample. Do not reset from the event interval itself. Dynamic POT does not replace the locked threshold rules in replay or the preregistered SMAP/MSL evaluation.

## Interpretation

The target risk level is a tail-model target, not a guarantee of operational false-positive rate. GPD moment fitting can be unstable when data are short, dependent, or non-stationary; check calibration size and evaluate on held-out clean streams. Scores and outputs in this repository remain research-benchmark results, not deployment evidence.
