# Reproducing OES-32-v2

Environment used: Python 3.12.14, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, PyYAML 6.0.2, matplotlib 3.10.3 (Linux, 8 cores). Requirements are pinned in `oes32-v2/requirements.txt`.

```bash
# from the repository root, on branch oes32-v2
python -m venv .venv-v2 && . .venv-v2/bin/activate
pip install -r oes32-v2/requirements.txt
oes-resilience smap-msl fetch --data-dir DATA        # or: python -m oes_resilience smap-msl fetch --data-dir DATA
python -m oes_resilience smap-msl verify --data-dir DATA   # 165 files vs reports/smap_msl_data.sha256
cd oes32-v2
python oes32v2/data.py DATA                  # rebuilds data/dataset_manifest.json (should be byte-identical)
python -m oes32v2.search DATA                # validation-only search: 4223 trials, about 1 min
python -m oes32v2.select DATA                # selection, validation comparison, writes configs/final_config.yaml
python -m oes32v2.analysis DATA              # failure atlas, figures and tables from the saved test outputs
```

**Expected outputs**
- `experiments/all_trials.csv` and the leaderboard reproduce exactly. They are deterministic; only `runtime_s` varies.
- `configs/final_config.yaml` should be trial 4193.

**The test run**
- `python -m oes32v2.run_test DATA` refuses to run while `results/final_test_summary.json` exists. This keeps the "exactly once" record.
- To replicate it independently, run it in a fresh clone after deleting that file. Expected pooled test event F1: v2 SMAP 0.212581 and MSL 0.433566; v1 SMAP 0.378601 and MSL 0.434783.
- Isolation Forest results depend on the scikit-learn version.

**What is not deterministic**
- Wall-clock timings, including the runtime-ratio constraint, depend on the machine.
- Memory peaks (tracemalloc) depend on the Python version.

**Run history:** see `experiments/RUN_LOG.md`.
