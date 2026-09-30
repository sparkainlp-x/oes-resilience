# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""OES-Resilience: an open, reproducible benchmark for multichannel telemetry anomaly detection.

Synthetic data only: not a physical sensor, quantum processor, navigation (GPS) system
or medical device, and no claims about any of those.

Modules: :mod:`.core` (v0.1 single-frame benchmark, OES32 scoring, CLI),
:mod:`.detectors` (plugin API and detectors), :mod:`.streams` (multi-step streams),
:mod:`.scorecard` (calibrated detector comparison).
"""

from ._version import __version__
from .core import (
    PROJECT,
    Config,
    Event,
    Regime,
    ScoreWeights,
    Status,
    assess_signal,
    main,
    run_benchmark,
    run_trial,
    score_signals,
    threshold_sweep,
)
from .detectors import (
    CUSUMDetector,
    DetectionResult,
    Detector,
    EWMADetector,
    IsolationForestDetector,
    OES32Detector,
    RobustZScoreDetector,
    TemporalDetector,
    available_detectors,
    create_detector,
    get_detector_class,
    load_plugins,
    register_detector,
    unregister_detector,
)
from .scorecard import CompareConfig, calibrate_threshold, run_compare
from .streams import StreamConfig, StreamRegime, generate_streams

__all__ = [
    "PROJECT", "Config", "Event", "Regime", "ScoreWeights", "Status", "__version__", "assess_signal", "main",
    "run_benchmark", "run_trial", "score_signals", "threshold_sweep",
    "CUSUMDetector", "DetectionResult", "Detector", "EWMADetector", "IsolationForestDetector", "OES32Detector",
    "RobustZScoreDetector", "TemporalDetector", "available_detectors", "create_detector", "get_detector_class",
    "load_plugins", "register_detector", "unregister_detector",
    "CompareConfig", "calibrate_threshold", "run_compare", "StreamConfig", "StreamRegime", "generate_streams",
]
