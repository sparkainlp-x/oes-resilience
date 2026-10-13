# Run log (validation and test executions, in order; times EDT, 2026-10-13, from file timestamps)

1. ~00:30: code smoke check before the logged search. It ran ONE validation evaluation (window 32, top-4 mean, smoothing 3, per-channel q0.99, PCA-residual max). It read only train files and is not part of the search or the selection.
2. 00:30:43–00:31:50: validation search (`python -m oes32v2.search`), 4223 trials, 0 failed, 67 s (`experiments/search_stdout.log`).
3. ~00:34:45–00:35:24: selection and validation comparison (`python -m oes32v2.select`). Selected trial 4193 (G: v2 + rolling MAD, max). Hard constraint 4 fails: runtime ratio 2.27 > 2.0 under the locked measure.
4. 00:35:24: `configs/final_config.yaml` frozen; committed as 198c1d8.
5. 00:36:19: `experiments/TEST_LOCK.md` written; committed as 5477ffb.
6. 00:36:22–00:45:13: the single test run (`python -m oes32v2.run_test`), exit 0, no rerun (`experiments/test_stdout.log`). Frozen v1 reproduced the published v0.5 pooled event F1 exactly.
