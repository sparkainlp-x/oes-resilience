"""The single OES-32-v2 test evaluation. Refuses to run twice or without a valid test lock."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import yaml

from . import core, pipeline
from .data import DATASETS, V2_ROOT, channels, split_for, v1

PUBLISHED_V1 = {"SMAP": 0.378601, "MSL": 0.434783}   # reports/smap_msl_results/smap_msl_summary.csv, all_channels, 0.01


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def guard() -> dict:
    out = V2_ROOT / "results" / "final_test_summary.json"
    if out.exists():
        sys.exit("REFUSED: results/final_test_summary.json exists; the test set is evaluated exactly once.")
    lock = V2_ROOT / "experiments" / "TEST_LOCK.md"
    if not lock.exists():
        sys.exit("REFUSED: experiments/TEST_LOCK.md is missing.")
    text = lock.read_text()
    cfg_hash = sha(V2_ROOT / "configs" / "final_config.yaml")
    if cfg_hash not in text:
        sys.exit("REFUSED: final_config.yaml hash does not match TEST_LOCK.md.")
    return {"final_config_sha256": cfg_hash, "test_lock_sha256": sha(lock),
            "manifest_sha256": sha(V2_ROOT / "data" / "dataset_manifest.json")}


def main(data_dir: str) -> None:
    lockinfo = guard()
    data_dir = Path(data_dir)
    v1.verify_data(data_dir, V2_ROOT.parent / "reports" / "smap_msl_data.sha256")
    cfg = yaml.safe_load((V2_ROOT / "configs" / "final_config.yaml").read_text())
    chans = channels(data_dir)
    prepared = []
    for c in chans:
        train, test = v1.load_series(data_dir, c)
        sp = split_for(train.size)
        fit, calib = (train[: sp.fit_end], train[sp.val_start:]) if sp.has_val else (train, train)
        prepared.append({"c": c, "train": train, "test": test, "fit": fit, "calib": calib, "has_val": sp.has_val})
    gstats = pipeline.v2_global_stats({d: [p["fit"] for p in prepared if p["c"].spacecraft == d and p["has_val"]] for d in DATASETS})
    pooled = None
    if cfg["threshold"]["kind"] == "global_quantile":
        pooled = {d: np.concatenate([pipeline.v2_scores(cfg, p["fit"], p["calib"], p["calib"][:1], st(p, cfg, gstats))[0]
                                     for p in prepared if p["c"].spacecraft == d and p["has_val"]]) for d in DATASETS}
    methods = ["A_v1_frozen", "B_rolling_mad", "C_train_percentile", "D_pca_residual",
               *[f"E_isolation_forest_seed{s}" for s in pipeline.IF_SEEDS], "F_or_G_v2_selected"]
    rows = {m: [] for m in methods}
    matches, keep = [], {}
    timing = {}
    for m in methods:
        tracemalloc.start()
        t0 = time.perf_counter()
        for p in prepared:
            c = p["c"]
            if m == "A_v1_frozen":
                pred, s, T = pipeline.v1_pred(p["train"], p["test"])
            elif m == "F_or_G_v2_selected":
                pred, s, T = pipeline.v2_pred(cfg, p["fit"], p["calib"], p["test"], st(p, cfg, gstats),
                                              pooled[c.spacecraft] if pooled else None)
            else:
                name = {"B": "rolling_mad", "C": "train_percentile", "D": "pca_residual", "E": "isolation_forest"}[m[0]]
                seed = int(m.rsplit("seed", 1)[1]) if m.startswith("E") else 0
                pred, s, T = pipeline.baseline_pred(name, p["fit"], p["calib"], p["test"], seed)
            r = core.channel_eval(pred, c.events, p["test"].size)
            r.update(spacecraft=c.spacecraft, chan_id=c.chan_id, threshold=T)
            rows[m].append(r)
            if m in ("A_v1_frozen", "F_or_G_v2_selected"):
                keep[(m, c.chan_id)] = (s.astype(np.float32), T, pred)
                for gi, (gs, ge) in enumerate(c.events):
                    hit = [list(map(int, q)) for q in pred if q[0] <= ge and q[1] >= gs]
                    matches.append({"method": m, "spacecraft": c.spacecraft, "chan_id": c.chan_id, "kind": "ground_truth",
                                    "index": gi, "start": gs, "end": ge, "class": c.classes[gi], "matched": bool(hit),
                                    "n_matching_predictions": len(hit), "max_score": float(s[gs:ge + 1].max()), "threshold": T})
                for pi, (ps, pe) in enumerate(pred):
                    ok = any(ps <= ge and pe >= gs for gs, ge in c.events)
                    matches.append({"method": m, "spacecraft": c.spacecraft, "chan_id": c.chan_id, "kind": "prediction",
                                    "index": pi, "start": int(ps), "end": int(pe), "class": "", "matched": ok,
                                    "n_matching_predictions": "", "max_score": float(s[ps:pe + 1].max()), "threshold": T})
        timing[m] = {"wall_s": time.perf_counter() - t0, "peak_bytes": tracemalloc.get_traced_memory()[1]}
        tracemalloc.stop()

    res = V2_ROOT / "results"
    summary = {}
    for m in methods:
        summary[m] = {d: core.aggregate([r for r in rows[m] if r["spacecraft"] == d]) for d in DATASETS}
    boot = {}
    for d in DATASETS:
        sub = {m: [r for r in rows[m] if r["spacecraft"] == d] for m in methods}
        b = core.bootstrap_f1(sub)
        boot[d] = {m: [float(np.percentile(b[m], 2.5)), float(np.percentile(b[m], 97.5))] for m in methods}
        diff = b["F_or_G_v2_selected"] - b["A_v1_frozen"]
        boot[d]["v2_minus_v1"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    v1_repro = {d: round(summary["A_v1_frozen"][d]["event_f1"], 6) for d in DATASETS}
    ifs = {d: [summary[f"E_isolation_forest_seed{s}"][d]["event_f1"] for s in pipeline.IF_SEEDS] for d in DATASETS}
    h2 = {d: summary["F_or_G_v2_selected"][d]["event_f1"] - summary["A_v1_frozen"][d]["event_f1"] for d in DATASETS}
    out = {
        "schema": "oes32-v2/final-test-summary/1", "mode": "offline benchmark mode", "run_once": True,
        "lock": lockinfo, "final_config": cfg,
        "v1_reproduces_published_v0.5": {"published": PUBLISHED_V1, "reproduced": v1_repro,
                                         "match": all(abs(v1_repro[d] - PUBLISHED_V1[d]) < 1e-6 for d in DATASETS)},
        "summary": summary, "bootstrap_95ci_event_f1_channel_level": boot,
        "isolation_forest_event_f1_mean_std": {d: [float(np.mean(v)), float(np.std(v, ddof=1))] for d, v in ifs.items()},
        "timing_test": timing,
        "confirmatory_H2": {"v2_minus_v1_event_f1": h2, "required": 0.05, "met": all(v >= 0.05 for v in h2.values())},
    }
    (res / "final_test_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    pcf = ["method", "spacecraft", "chan_id", "events", "tp", "fp", "fn", "pred_events", "point_tp", "point_fp", "point_fn", "threshold", "event_f1"]
    with (res / "per_channel_metrics.csv").open("w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=pcf, extrasaction="ignore"); w.writeheader()
        for m in methods:
            for r in rows[m]:
                w.writerow({"method": m, **r, "event_f1": round(core.prf(r["tp"], r["fp"], r["fn"])[2], 6)})
    with (res / "event_matches.csv").open("w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(matches[0].keys())); w.writeheader(); w.writerows(matches)
    np.savez_compressed(res / "test_scores_v1_v2.npz", **{f"{m}|{cid}|score": v[0] for (m, cid), v in keep.items()},
                        **{f"{m}|{cid}|threshold": np.array([v[1]]) for (m, cid), v in keep.items()},
                        **{f"{m}|{cid}|pred": np.asarray(v[2], dtype=np.int64).reshape(-1, 2) for (m, cid), v in keep.items()})
    print(json.dumps({"v1_repro": out["v1_reproduces_published_v0.5"], "H2": out["confirmatory_H2"],
                      "v2": {d: {k: round(summary["F_or_G_v2_selected"][d][k], 4) for k in ("event_precision", "event_recall", "event_f1", "macro_channel_f1")} | {"fp": summary["F_or_G_v2_selected"][d]["fp"]} for d in DATASETS}}, indent=1))


def st(p, cfg, gstats):
    return core.robust_stats(p["fit"]) if cfg["normalization"]["scope"] == "per_channel" else gstats[p["c"].spacecraft]


if __name__ == "__main__":
    main(sys.argv[1])
