# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""OES-Resilience: an open, reproducible benchmark for multichannel telemetry anomaly detection.

Synthetic benchmark: not a physical sensor, quantum processor, navigation (GPS) system
or medical device, and no claims about any of those. The only real data is the optional,
preregistered NASA SMAP/MSL evaluation (v0.5), downloaded by the user and never bundled.

Modules: :mod:`.core` (v0.1 single-frame benchmark, OES32 scoring, CLI),
:mod:`.detectors` (plugin API and detectors), :mod:`.streams` (multi-step streams),
:mod:`.scorecard` (calibrated detector comparison), :mod:`.stress` (robustness stress suite),
:mod:`.replay` (preregistered evaluation of timestamped replay files),
:mod:`.smap_msl` (v0.5: adapter and preregistered evaluation on the real public NASA SMAP/MSL dataset;
the data is downloaded and hash-verified, never bundled), :mod:`.qbench` (concept, unreleased: a classical
evidence and statistics layer around local simulator runs of small benchmark circuits; optional extra
``quantum``; no hardware results).
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
    MaxAbsDetector,
    OES32Detector,
    OES32EWMAHybridDetector,
    RobustZScoreDetector,
    TemporalDetector,
    available_detectors,
    create_detector,
    get_detector_class,
    load_plugins,
    register_detector,
    unregister_detector,
)
from .replay import build_report as build_replay_report
from .replay import load_preregistration, load_replay
from .scorecard import CompareConfig, calibrate_threshold, run_compare
from .streams import StreamConfig, StreamRegime, generate_streams
from .stress import SCENARIOS, StressConfig, StressParams, run_stress

__all__ = [
    "PROJECT", "Config", "Event", "Regime", "ScoreWeights", "Status", "__version__", "assess_signal", "main",
    "run_benchmark", "run_trial", "score_signals", "threshold_sweep",
    "CUSUMDetector", "DetectionResult", "Detector", "EWMADetector", "IsolationForestDetector", "OES32Detector",
    "OES32EWMAHybridDetector", "MaxAbsDetector",
    "RobustZScoreDetector", "TemporalDetector", "available_detectors", "create_detector", "get_detector_class",
    "load_plugins", "register_detector", "unregister_detector",
    "CompareConfig", "calibrate_threshold", "run_compare", "StreamConfig", "StreamRegime", "generate_streams",
    "SCENARIOS", "StressConfig", "StressParams", "run_stress",
    "build_replay_report", "load_preregistration", "load_replay",
]
