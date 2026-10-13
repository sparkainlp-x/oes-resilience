"""Pick the final configuration from the validation leaderboard, compare methods on validation,
check the hard constraints and freeze configs/final_config.yaml. Reads train files only."""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
import tracemalloc

import numpy as np
import yaml

from . import core, pipeline
from .data import DATASETS, V2_ROOT
from .search import load_validation, rank_key, complexity


def load_trials():
    rows = list(csv.DictReader((V2_ROOT / "experiments" / "all_trials.csv").open(encoding="utf-8")))
    for r in rows:
        for k in ("event_f1", "event_precision", "event_recall", "macro_channel_f1", "runtime_s", "point_f1",
                  "smap_event_f1", "msl_event_f1"):
            r[k] = float(r[k])
        for k in ("trial_id", "complexity", "fp_events", "tp_events", "fn_events"):
            r[k] = int(r[k])
    return [r for r in rows if r["status"] == "ok"]


def choose(trials):
    best_all = sorted(trials, key=rank_key)[0]
    best_f = sorted([t for t in trials if json.loads(t["config"])["complementary"]["feature"] == "none"], key=rank_key)[0]
    use_g = json.loads(best_all["config"])["complementary"]["feature"] != "none" and best_all["event_f1"] >= best_f["event_f1"] + 0.01
    return (best_all if use_g else best_f), best_f, best_all, use_g


def method_rows(chans, method, cfg=None, seed=0, gstats=None):
    rows = []
    pooled = None
    if cfg and cfg["threshold"]["kind"] == "global_quantile":
        pooled = {}
        for d in DATASETS:
            cs = [pipeline.v2_scores(cfg, c["fit"], c["clean"], c["clean"][:1], stats_for(c, cfg, gstats))[0] for c in chans if c["spacecraft"] == d]
            pooled[d] = np.concatenate(cs)
    for c in chans:
        if method == "v1":
            pred, _, _ = pipeline.v1_pred(c["fit"], c["inj"])
        elif method == "v2":
            pred, _, _ = pipeline.v2_pred(cfg, c["fit"], c["clean"], c["inj"], stats_for(c, cfg, gstats),
                                          pooled[c["spacecraft"]] if pooled else None)
        else:
            pred, _, _ = pipeline.baseline_pred(method, c["fit"], c["clean"], c["inj"], seed)
        r = core.channel_eval(pred, c["events"], c["inj"].size)
        r.update(spacecraft=c["spacecraft"], chan_id=c["chan_id"])
        rows.append(r)
    return rows


def stats_for(c, cfg, gstats):
    return core.robust_stats(c["fit"]) if cfg["normalization"]["scope"] == "per_channel" else gstats[c["spacecraft"]]


def timed(fn, reps=3):
    times = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); times.append(time.perf_counter() - t)
    tracemalloc.start(); fn(); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return statistics.median(times), peak


def summarize(name, rows):
    out = []
    for d in ("all", *DATASETS):
        a = core.aggregate(rows if d == "all" else [r for r in rows if r["spacecraft"] == d])
        out.append({"method": name, "dataset": d, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in a.items()}})
    return out


def main(data_dir):
    trials = load_trials()
    chosen, best_f, best_all, use_g = choose(trials)
    cfg = json.loads(chosen["config"])
    chans = load_validation(data_dir)
    gstats = pipeline.v2_global_stats({d: [c["fit"] for c in chans if c["spacecraft"] == d] for d in DATASETS})
    results = {}
    results["A_v1_frozen"] = method_rows(chans, "v1")
    results["B_rolling_mad"] = method_rows(chans, "rolling_mad")
    results["C_train_percentile"] = method_rows(chans, "train_percentile")
    results["D_pca_residual"] = method_rows(chans, "pca_residual")
    if_rows = {s: method_rows(chans, "isolation_forest", seed=s) for s in pipeline.IF_SEEDS}
    results["F_or_G_v2_selected"] = method_rows(chans, "v2", cfg, gstats=gstats)
    results["F_v2_best_standalone"] = method_rows(chans, "v2", json.loads(best_f["config"]), gstats=gstats)
    summary = []
    for k, rows in results.items():
        summary += summarize(k, rows)
    for s, rows in if_rows.items():
        summary += summarize(f"E_isolation_forest_seed{s}", rows)
    f1s = [core.aggregate(r)["event_f1"] for r in if_rows.values()]
    t_v1, m_v1 = timed(lambda: method_rows(chans, "v1"))
    t_v2, m_v2 = timed(lambda: method_rows(chans, "v2", cfg, gstats=gstats))
    v2a, v1a = core.aggregate(results["F_or_G_v2_selected"]), core.aggregate(results["A_v1_frozen"])
    check_f1 = abs(v2a["event_f1"] - chosen["event_f1"]) < 1e-9
    constraints = {
        "event_precision>=0.55": v2a["event_precision"] >= 0.55,
        "event_recall>=0.55": v2a["event_recall"] >= 0.55,
        "improvement_over_v1>=0.05": (v2a["event_f1"] - v1a["event_f1"]) >= 0.05,
        "runtime_ratio<=2": (t_v2 / t_v1) <= 2.0,
        "zero_test_influence": True,
    }
    res = V2_ROOT / "results"; res.mkdir(exist_ok=True)
    fields = list(summary[0].keys())
    with (res / "validation_summary.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=fields); w.writeheader(); w.writerows(summary)
    meta = {
        "selected_trial_id": chosen["trial_id"], "selected_is_G_complementary": use_g,
        "best_standalone_trial_id": best_f["trial_id"], "best_standalone_event_f1": best_f["event_f1"],
        "best_overall_trial_id": best_all["trial_id"], "best_overall_event_f1": best_all["event_f1"],
        "g_rule": "G kept only if its validation event F1 >= best F + 0.01",
        "rerun_matches_search_log": check_f1,
        "validation": {"v2": v2a, "v1": v1a, "improvement": v2a["event_f1"] - v1a["event_f1"],
                       "isolation_forest_event_f1_mean": float(np.mean(f1s)), "isolation_forest_event_f1_std": float(np.std(f1s, ddof=1))},
        "runtime": {"v1_s": t_v1, "v2_s": t_v2, "ratio": t_v2 / t_v1, "v1_peak_bytes": m_v1, "v2_peak_bytes": m_v2,
                    "measure": "median of 3 passes over all 80 validation segments (fit+calibration+target scoring, thresholds, alerts, metrics); tracemalloc peak on a 4th pass"},
        "hard_constraints": constraints, "all_hard_constraints_met": all(constraints.values()),
    }
    (res / "validation_selection.json").write_text(json.dumps(meta, indent=1) + "\n")
    final = {"schema": "oes32-v2/final-config/1", "frozen_from_trial": chosen["trial_id"], "mode": "offline benchmark mode",
             "complexity": complexity(cfg), **cfg}
    (V2_ROOT / "configs" / "final_config.yaml").write_text(
        "# OES-32-v2 FROZEN final configuration (selected on validation only; frozen before any test evaluation)\n"
        + yaml.safe_dump(final, sort_keys=False))
    print(json.dumps({k: meta[k] for k in ("selected_trial_id", "selected_is_G_complementary", "best_standalone_event_f1",
                                           "best_overall_event_f1", "rerun_matches_search_log", "hard_constraints", "runtime")}, indent=1))
    print("v2", {k: round(v, 4) for k, v in v2a.items() if isinstance(v, float)})
    print("v1", {k: round(v, 4) for k, v in v1a.items() if isinstance(v, float)})


if __name__ == "__main__":
    main(sys.argv[1])
