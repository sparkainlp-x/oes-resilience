# SMAP/MSL ablation preregistration status

## Status: NOT RUN

No preregistered ablations were defined in `reports/smap_msl_protocol.json`, and no ablation implementation exists in the repository. A repository code search found no `ablation`, `latch`, `residual`, or `calibration` experiment path.

Therefore, no ablation results are reported here. Any ablation run after this date is **exploratory** and cannot alter, weaken, strengthen, or reinterpret the locked primary result in `reports/smap_msl_results/BASELINE_COMPARISONS.md`.

## Locked primary result

At the primary target `target_fp = 0.01` on all channels, the preregistered headline success criterion was **not met**. OES-32 significantly outperformed EWMA on SMAP but did not significantly outperform maxabs or CUSUM on either SMAP or MSL.

This verdict remains the final primary result regardless of any future exploratory ablation.

## Why no ablations were run

Running new component-removal or threshold-sensitivity experiments after observing the primary result would create a multiple-testing and researcher-degrees-of-freedom problem. To preserve external checkability, ablations must be specified, implemented, and locked before their results are interpreted as confirmatory.

## Future exploratory variants

These are candidate specifications only. They are not preregistered confirmatory tests.

| Candidate | Proposed manipulation | Required comparison |
|---|---|---|
| Residual normalization | Replace or remove the residual-normalization component, if implemented | OES-32 primary configuration versus variant, identical data and event rules |
| Latch threshold | Sweep the latch threshold over a small, fixed grid declared before execution | OES-32 primary configuration versus each threshold variant |
| Channel grouping | Evaluate channel-level versus grouped-channel scoring | OES-32 primary configuration versus grouping variant |
| Calibration | Compare train-only threshold calibration with a fixed predeclared alternative | OES-32 primary calibration versus alternative |

## Rules for any future run

1. Create a new versioned protocol addendum before implementation.
2. Use the same SMAP/MSL splits, event definitions, primary metric, seed, and confidence-interval method as the locked evaluation.
3. Correct for multiplicity across all ablation comparisons.
4. Report every specified variant, including unfavorable results.
5. Label all outcomes `exploratory`; do not use them to revise the primary verdict.
6. Add results only to a separate `ABLATIONS_RESULTS.md` and machine-readable CSV, with checksums and run metadata.

## Required disclosure

If any candidate above is implemented, this file must be updated to record the addendum commit, implementation commit, run date, environment, seed, and exact command. Until then, the scientifically accurate status is: **no preregistered ablations exist; none were run.**
