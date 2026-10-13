# Test lock (OES-32-v2)

Written 2026-10-13 before any OES-32-v2 test evaluation.

- Code commit (evaluation-code version): `198c1d85a2ca83d1d98cce57163b9fe96b6c2574`
- Protocol-lock commit: `8b474b6`
- `configs/final_config.yaml` SHA-256: `e16034f602871f119031560d15d17e4890f8571441af95df562662a6496d2698` (trial 4193; frozen)
- `data/dataset_manifest.json` SHA-256: `2ffcc18690a73f480e904068a82f2d111ac54ff63a625bda38221f6072d1a032`; data verified against `reports/smap_msl_data.sha256` (165 files)
- Seeds: injection base 20261013 (+ channel index); Isolation Forest {0,1,2,3,4}; bootstrap 42 (10 000 channel-level resamples)
- Environment: Python 3.12.14, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, PyYAML 6.0.2, matplotlib 3.10.3

**Zero test-label influence.** The search (`oes32v2/search.py`), selection (`oes32v2/select.py`) and every threshold read only the TRAIN files: fit segments, clean validation segments and synthetically injected validation copies.
- No test file or test label was read by any OES-32-v2 code before this lock.
- The only previous contact with the test split is the published v1 evaluation (oes-resilience v0.5). Its pooled numbers are known from the v1 README. They were used only as the fixed v1 reference value, not for any selection.

**Known before the test:** hard constraint 4 (runtime ≤ 2× v1) failed on validation, with ratio 2.27. Under the locked rule the predeclared criterion therefore cannot be met, whatever the test result. The test is still run exactly once, as required, and reported.

Run exactly once with: `python -m oes32v2.run_test <data_dir>` (the runner refuses a second run).
