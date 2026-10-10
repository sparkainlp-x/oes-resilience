# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the v0.6.0 adaptive threshold and robustness upgrades."""

from __future__ import annotations

import numpy as np
import pytest

import oes_resilience as package
from oes_resilience import core, detectors, scorecard, streams, stress
from oes_resilience.adaptive_threshold import POTThreshold


def test_pot_requires_clean_calibration_and_valid_inputs():
    pot = POTThreshold()
    with pytest.raises(RuntimeError, match="fitted"):
        pot.update(0.5)
    with pytest.raises(ValueError, match="20"):
        pot.fit_calibration(np.ones(19))
    with pytest.raises(ValueError, match="one-dimensional"):
        pot.fit_calibration(np.ones((20, 1)))
    with pytest.raises(ValueError, match="finite"):
        pot.fit_calibration(np.r_[np.ones(19), np.nan])
    with pytest.raises(ValueError, match="risk_level"):
        POTThreshold(risk_level=0.0)
    with pytest.raises(ValueError, match="exceedance fraction"):
        POTThreshold(risk_level=0.05, init_quantile=0.98)
    with pytest.raises(ValueError, match="window_size"):
        POTThreshold(window_size=2)


def test_pot_gpd_fit_masks_anomalies_and_updates_only_clean_history():
    rng = np.random.default_rng(45)
    calibration = rng.exponential(scale=1.0, size=600)
    pot = POTThreshold(risk_level=0.01, init_quantile=0.95, window_size=120, min_exceedances=8)
    assert pot.fit_calibration(calibration) is pot
    assert pot.fitted
    assert len(pot.history) == 120
    assert pot.exceedance_count >= 8
    assert pot.base_threshold < pot.threshold
    assert np.isfinite([pot.threshold, pot.shape, pot.scale]).all()

    before_history = tuple(pot.history)
    is_anomaly, used_threshold = pot.update(100.0)
    assert is_anomaly
    assert used_threshold == pot.threshold
    assert tuple(pot.history) == before_history

    is_anomaly, _ = pot.update(0.1)
    assert not is_anomaly
    assert pot.history[-1] == 0.1
    assert len(pot.history) == 120
    assert pot.params()["history_size"] == 120
    assert pot.params()["risk_level"] == 0.01


def test_pot_sparse_tail_falls_back_to_empirical_quantile_and_reset_refits():
    values = np.linspace(0.0, 1.0, 100)
    pot = POTThreshold(risk_level=0.01, init_quantile=0.95, window_size=50, min_exceedances=20)
    pot.fit_calibration(values)
    assert pot.exceedance_count < 20
    assert pot.shape == 0.0
    assert pot.threshold == pytest.approx(np.quantile(values, 0.99))

    shifted_clean = values + 0.25
    pot.reset(shifted_clean)
    assert pot.threshold > 0.25
    assert pot.base_threshold > 0.25
    with pytest.raises(ValueError, match="finite"):
        pot.update(float("inf"))


def test_pot_clean_stream_stays_near_its_configured_tail_budget():
    rng = np.random.default_rng(81)
    values = rng.exponential(size=1600)
    pot = POTThreshold(risk_level=0.01, init_quantile=0.95, window_size=400, min_exceedances=8)
    pot.fit_calibration(values[:800])
    decisions = [pot.update(float(value))[0] for value in values[800:]]
    assert np.mean(decisions) <= 0.04


def test_robust_oes32_preserves_single_channel_events_and_missing_masks():
    rng = np.random.default_rng(8)
    cfg = core.Config(channels=64, block_size=32)
    clean = rng.normal(0, 0.05, (400, 64))
    detector = detectors.RobustOES32Detector(cfg, top_k=1).fit(clean)
    event = np.zeros(64)
    event[7] = 0.5
    single_channel_score = detector.score(event)
    assert single_channel_score[0] > 2.0
    assert single_channel_score[1] < 1.0

    wider_topk = detectors.RobustOES32Detector(cfg, top_k=4).fit(clean)
    assert detector.score(event)[0] > wider_topk.score(event)[0]

    masked = event.copy()
    masked[32:64] = np.nan
    masked_score = detector.score(masked)
    assert masked_score[1] == 0.0
    assert np.isfinite(masked_score).all()
    assert detector.params()["scale"] == "1.4826 * per-channel MAD"


def test_robust_oes32_parameters_and_fit_validation():
    cfg = core.Config(channels=4, block_size=2)
    with pytest.raises(ValueError, match="top_k"):
        detectors.RobustOES32Detector(cfg, top_k=5)
    with pytest.raises(ValueError, match="huber_delta"):
        detectors.RobustOES32Detector(cfg, huber_delta=0)
    with pytest.raises(ValueError, match="weights must sum"):
        detectors.RobustOES32Detector(cfg, weights=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="weights must contain"):
        detectors.RobustOES32Detector(cfg, weights=(0.5, 0.5))
    with pytest.raises(RuntimeError, match="fitted"):
        detectors.RobustOES32Detector(cfg).score(np.zeros(4))
    with pytest.raises(ValueError, match="fit data must be complete"):
        detectors.RobustOES32Detector(cfg).fit(np.array([[0.0, np.nan, 0.0, 0.0]]))


def test_robust_oes32_reduces_calibrated_student_t_false_positives():
    config = scorecard.CompareConfig(
        trials=400,
        fit_trials=200,
        calibration_trials=500,
        streams=20,
        calibration_streams=20,
        fit_streams=20,
        detectors=("oes32", "oes32-robust"),
        tracks=("frame",),
    )
    fitted = scorecard.fit_detectors(config)
    thresholds, _ = scorecard.frame_calibration(config, fitted)
    frames, _, _ = stress.generate_stress_batch(
        "heavy_tail", "frame", range(500), config.config, config.stream, stress.StressParams()
    )
    false_positive_rates = {
        name: float((fitted[name].score(frames[:, 0]).max(axis=1) >= thresholds[name]).mean()) for name in fitted
    }
    assert false_positive_rates["oes32-robust"] < false_positive_rates["oes32"]


def test_change_point_cusum_resets_after_confirmed_regime_step():
    rng = np.random.default_rng(16)
    cfg = core.Config(channels=64, block_size=32)
    warmup = 8
    telemetry = rng.normal(0, 0.05, (1, 40, 64))
    telemetry[:, 18:, :] += 0.5
    detector = detectors.ChangePointCUSUMDetector(
        cfg, warmup=warmup, change_window=3, hazard_interval=50, change_probability=0.8
    )
    scores = detector.score(telemetry)[0]
    assert np.all(scores[:warmup] == 0)
    assert scores[18:24].max() > 1.0
    assert scores[30:].max() < scores[18:24].max()
    assert detector.params()["change_point_model"].startswith("finite-window")


def test_change_point_cusum_validation_and_masked_input():
    cfg = core.Config(channels=4, block_size=2)
    for kwargs in (
        {"change_window": 1},
        {"hazard_interval": 1},
        {"change_probability": 1.0},
        {"prior_mean_variance": 0.0},
    ):
        with pytest.raises((TypeError, ValueError)):
            detectors.ChangePointCUSUMDetector(cfg, warmup=3, **kwargs)

    telemetry = np.random.default_rng(4).normal(0, 0.05, (1, 12, 4))
    telemetry[0, 8:, 0:2] = np.nan
    detector = detectors.ChangePointCUSUMDetector(cfg, warmup=3, change_window=2)
    scores = detector.score(telemetry)
    assert scores.shape == (1, 12, 2)
    assert np.isfinite(scores).all()


def test_regime_steps_stress_scenario_is_seeded_and_stream_only():
    cfg = core.Config()
    stream = streams.StreamConfig(steps=32, warmup=8)
    params = stress.StressParams()
    x1, truth1, onsets1 = stress.generate_stress_batch("regime_steps", "stream", [2], cfg, stream, params)
    x2, truth2, onsets2 = stress.generate_stress_batch("regime_steps", "stream", [0, 2], cfg, stream, params)
    np.testing.assert_array_equal(x1[0], x2[1])
    np.testing.assert_array_equal(truth1, truth2[1:])
    np.testing.assert_array_equal(onsets1, onsets2[1:])
    assert not truth1.any()
    assert onsets1.tolist() == [-1]
    assert stress.get_scenario("regime_steps").tracks == ("stream",)
    segment_means = [x1[0, a:b].mean() for a, b in ((0, 8), (8, 16), (16, 24), (24, 32))]
    assert segment_means[1] > segment_means[0] + 0.1
    assert segment_means[2] < segment_means[1] - 0.2
    assert segment_means[3] > segment_means[2] + 0.1


def test_v06_api_is_public_and_registered():
    assert package.__version__ == "0.6.0"
    assert "oes32-robust" in package.available_detectors()
    assert "cusum-cp" in package.available_detectors()
    assert package.POTThreshold is POTThreshold
