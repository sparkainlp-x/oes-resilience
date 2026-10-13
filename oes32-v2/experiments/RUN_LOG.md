# Run log (validation and test executions, in order)

1. 2026-10-13 ~00:35 EDT: code smoke check before the logged search. It ran ONE validation evaluation (window 32, top-4 mean, smoothing 3, per-channel q0.99, PCA-residual max). It touched only train files and is not part of the search or the selection. Listed here for completeness.
2. 2026-10-13 00:37 EDT: validation search (`python -m oes32v2.search`), 4223 trials, 0 failed, about 67 s (`experiments/search_stdout.log`).
3. 2026-10-13 ~00:45 EDT: selection and validation comparison (`python -m oes32v2.select`). Selected trial 4193 (G: v2 + rolling MAD, max). Hard constraint 4 fails: runtime ratio 2.27 > 2.0 under the locked measure (median of 3 passes).
