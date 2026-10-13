# Review guide

A short path for reviewers. Synthetic data only unless you choose the optional SMAP/MSL path.

## 1. What this is

An open, reproducible telemetry-anomaly benchmark with detector comparisons, a stress suite, and a preregistered evaluation on one public dataset (NASA SMAP/MSL, release v0.5.0). On that evaluation, OES32 did **not** meet its stated success criterion. This is research software, not hardware or field software.

## 2. Ten-minute check

```bash
python -m pip install -r .github/ci-requirements.txt .
ruff check .
python -m pytest --cov --cov-fail-under=90
```

Expected: ruff reports all checks passed; pytest passes with total coverage at or above 90%. Counts depend on whether SciPy and scikit-learn are installed (see CI requirements).

## 3. Reproduce one scorecard

```bash
bash scripts/reproduce_compare.sh   # about 70 s on one 8-core Linux box; machine-dependent
```

Expected: exit 0, printed SHA-256 matching the value in the script, and a byte-identical `examples/compare_scorecard.csv`. Exit 1 means the rebuilt scorecard differs.

## 4. Reproduce all examples

From a package install (or `python -m oes_resilience` at the repo root), run the same sequence CI uses: `run`, `sweep`, `compare --verify`, `stress --verify`, the burst-stress case, then the two replay `--verify` paths. Each rebuilt CSV must `cmp` equal to the matching file under `examples/`. Wall times here were about 1 s (run), under 1 s (sweep), 65 s (compare), 3 min (stress), 1.5 min (burst stress).

## 5. Real data (optional)

```bash
oes-resilience smap-msl fetch --data-dir DIR
sha256sum -c reports/smap_msl_data.sha256   # from DIR after fetch, or use the committed manifest
oes-resilience smap-msl evaluate --data-dir DIR --out-dir OUT --verify
sha256sum -c reports/smap_msl_results/SHA256SUMS
```

Expected: evaluate exits 0 with "headline criterion met: False". Summary and comparison CSVs and the SVG match the committed results; `smap_msl_results.json` may differ only in the `"version"` field (documented exception). CI never downloads this dataset.

## 6. Where the evidence lives

- Synthetic examples: `examples/`
- SMAP/MSL protocol lock, data checksums, results and Evidence Passport page: `reports/`, `docs/`
- Detector API notes: `docs/detector-api.md`
- Original prototype review (historical): `REVIEW.md`, `original/`

## 7. Released vs on main

The latest release is **v0.5.1**. The adaptive methods section in the README is **v0.6.0 on `main` only** (unreleased; no v0.6.0 tag). `CITATION.cff` and the archived release still say 0.5.1. Cite the release, not `main`. Open work on other branches is separate from this review pass.

## 8. Known limits

- CI pins tools in `.github/ci-requirements.txt`; example CSV identity still depends on NumPy staying behaviour-compatible with that pin.
- The optional Isolation Forest job needs the `iforest` extra (scikit-learn).
- Evidence labels and the SMAP/MSL negative outcome are stated in the README; do not reframe them.
