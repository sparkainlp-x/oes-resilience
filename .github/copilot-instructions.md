# Copilot instructions for OES-Resilience

## Project scope
- This is a reproducible research benchmark for multichannel telemetry anomaly detection. Synthetic results describe only the software's behavior on its generators; do not claim hardware, spacecraft, navigation, safety, medical, or production validation.
- The only real-data evaluation is the labelled NASA SMAP/MSL analysis. Do not bundle or commit downloaded dataset files. CI must use small synthetic fixtures and must never fetch SMAP/MSL data.
- Preserve the detector plugin contract: finite, non-negative per-block scores; explicit fit requirements; declared NaN support; no silent zero-filling. Reject infinities and malformed inputs.

## Reproducibility and evaluation integrity
- Keep seed streams keyed by their documented purpose/regime/trial. Do not change existing seed derivation, draw order, score definitions, or historical benchmark outputs unless the task explicitly requires it and the change is documented.
- Keep evaluation, fitting, and calibration data disjoint. Do not tune thresholds on event labels, test splits, or replay event frames.
- Treat `reports/smap_msl_protocol.json`, its preregistration/hash records, and the published v0.5 result files as historical locked artifacts. Never edit them to improve a result; create a new, separately preregistered protocol for any future evaluation.
- Dynamic thresholds must be trained from clean calibration scores. Mask flagged observations; only reset/recalibrate after an independently confirmed clean regime change. Never silently feed suspected event intervals into calibration.
- Report negative and inconclusive results honestly. Do not present synthetic outcomes as evidence about real telemetry.

## Implementation conventions
- Keep NumPy as the only required runtime dependency. Put heavyweight or optional approaches behind an explicit optional extra.
- Add detector tests for shapes, validation, calibration leakage, missing-data behavior, deterministic output, and both positive and negative cases.
- New synthetic stress cases must have deterministic keyed seeds, documented ground truth, and their own non-colliding scenario key.
- Do not commit downloaded data, machine-dependent timing output, credentials, or generated local caches.

## Verification
- Install test tools with `python -m pip install -e ".[test]"` and `python -m pip install ruff`.
- Run `ruff check .` and `python -m pytest --cov --cov-report=xml --cov-report=term-missing --cov-fail-under=90`.
- Run the reproducibility commands in `.github/workflows/ci.yml` and update committed example CSVs only when a documented benchmark change intentionally changes them.
- Review `CHANGELOG.md`, `README.md`, `docs/detector-api.md`, and `docs/stress.md` whenever public behavior changes.
