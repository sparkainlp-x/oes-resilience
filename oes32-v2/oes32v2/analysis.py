"""Post-test analysis: failure atlas, figures, comparison tables. Reads the saved test outputs; no tuning."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import core  # noqa: E402
from .data import DATASETS, V2_ROOT, channels, v1  # noqa: E402

RES, FIG = V2_ROOT / "results", V2_ROOT / "figures"
METHODS_SHOW = {"A_v1_frozen": "v1 (frozen)", "F_or_G_v2_selected": "v2 (selected)"}


def categorize(kind, start, end, x, train, max_score, T):
    """Locked heuristic rules (protocol section 9), first match wins."""
    med = np.median(train)
    rsd = 1.4826 * np.median(np.abs(train - med))
    seg = x[start:end + 1]
    pre = x[max(0, start - 128):start]
    if rsd == 0 or np.ptp(seg) == 0:
        return "data-quality issue"
    if end - start + 1 <= 5:
        return "isolated spike"
    if kind == "FN" and max_score >= 0.8 * T:
        return "threshold miss"
    if seg.size >= 3:
        slope = np.polyfit(np.arange(seg.size), seg, 1)[0]
        if abs(slope) * seg.size >= 2 * rsd:
            return "drift"
    if pre.size and abs(seg.mean() - pre.mean()) >= 2 * rsd:
        return "regime shift"
    if pre.size > 1:
        prsd = 1.4826 * np.median(np.abs(pre - np.median(pre)))
        if seg.std() >= 2 * max(prsd, 1e-12):
            return "noisy segment"
    pp = x[max(0, start - 256):start]
    if pp.size > 70 and pp.std() > 0:
        z = pp - pp.mean()
        ac = [np.dot(z[:-k], z[k:]) / np.dot(z, z) for k in range(1, 65)]
        if max(ac) >= 0.5:
            return "periodic mismatch"
    if kind == "FN":
        return "score failure"
    return "unknown"


def main(data_dir):
    data_dir = Path(data_dir)
    summary = json.loads((RES / "final_test_summary.json").read_text())
    npz = np.load(RES / "test_scores_v1_v2.npz")
    chans = {c.chan_id: c for c in channels(data_dir)}
    series = {cid: v1.load_series(data_dir, c) for cid, c in chans.items()}
    matches = list(csv.DictReader((RES / "event_matches.csv").open()))
    atlas = []
    for r in matches:
        if r["method"] not in METHODS_SHOW:
            continue
        unmatched = r["matched"] == "False"
        if not unmatched:
            continue
        kind = "FN" if r["kind"] == "ground_truth" else "FP"
        train, test = series[r["chan_id"]]
        cat = categorize(kind, int(r["start"]), int(r["end"]), test, train, float(r["max_score"]), float(r["threshold"]))
        atlas.append({"method": r["method"], "spacecraft": r["spacecraft"], "chan_id": r["chan_id"], "type": kind,
                      "start": r["start"], "end": r["end"], "length": int(r["end"]) - int(r["start"]) + 1,
                      "label_class": r["class"], "max_score": r["max_score"], "threshold": r["threshold"], "category": cat})
    with (RES / "failure_atlas.csv").open("w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(atlas[0].keys())); w.writeheader(); w.writerows(atlas)
    top = {m: Counter((a["type"], a["category"]) for a in atlas if a["method"] == m).most_common(5) for m in METHODS_SHOW}
    (RES / "failure_atlas_top5.json").write_text(json.dumps({m: [[f"{t} / {c}", n] for (t, c), n in v] for m, v in top.items()}, indent=1) + "\n")

    # overlays: first example of each of v2's top-5 patterns (plus v1 for the same window)
    od = FIG / "score_label_alert_overlays"; od.mkdir(parents=True, exist_ok=True)
    for rank, ((t, cat), _) in enumerate(top["F_or_G_v2_selected"], 1):
        ex = next(a for a in atlas if a["method"] == "F_or_G_v2_selected" and a["type"] == t and a["category"] == cat)
        cid, s0, e0 = ex["chan_id"], int(ex["start"]), int(ex["end"])
        test = series[cid][1]
        lo, hi = max(0, s0 - 300), min(test.size, e0 + 300)
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(range(lo, hi), test[lo:hi], lw=0.8, color="k"); ax[0].set_ylabel("telemetry (col 0)")
        for gs, ge in chans[cid].events:
            for a in ax:
                if ge >= lo and gs <= hi:
                    a.axvspan(max(gs, lo), min(ge, hi), color="tab:red", alpha=0.15)
        for i, (m, lab) in enumerate(METHODS_SHOW.items(), 1):
            sc, T, pred = npz[f"{m}|{cid}|score"], float(npz[f"{m}|{cid}|threshold"][0]), npz[f"{m}|{cid}|pred"]
            ax[i].plot(range(lo, hi), sc[lo:hi], lw=0.8); ax[i].axhline(T, color="tab:orange", ls="--", lw=1)
            for ps, pe in pred:
                if pe >= lo and ps <= hi:
                    ax[i].axvspan(max(ps, lo), min(pe, hi), ymin=0, ymax=0.06, color="tab:blue")
            ax[i].set_ylabel(f"{lab} score"); ax[i].set_yscale("symlog")
        ax[0].set_title(f"{cid} test [{lo},{hi}): v2 {t} '{cat}' at [{s0},{e0}]. Red = label, blue bar = alert, dashed = threshold. Offline benchmark; SMAP/MSL test split.", fontsize=8)
        fig.tight_layout(); fig.savefig(od / f"{rank}_{t}_{cat.replace(' ', '_')}_{cid}.png", dpi=110); plt.close(fig)

    # per-channel F1 distribution
    pc = list(csv.DictReader((RES / "per_channel_metrics.csv").open()))
    fd = FIG / "per_channel_f1_distribution"; fd.mkdir(parents=True, exist_ok=True)
    meths = ["A_v1_frozen", "B_rolling_mad", "C_train_percentile", "D_pca_residual", "E_isolation_forest_seed0", "F_or_G_v2_selected"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, d in zip(axs, DATASETS):
        data = [[float(r["event_f1"]) for r in pc if r["method"] == m and r["spacecraft"] == d] for m in meths]
        ax.boxplot(data, tick_labels=[m.split("_", 1)[0] for m in meths]); ax.set_title(f"{d}: per-channel test event F1"); ax.set_ylim(-0.05, 1.05)
    fig.suptitle("A v1, B rolling MAD, C train percentile, D PCA residual, E IForest (seed 0), F/G v2", fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "per_channel_event_f1_test.png", dpi=110); plt.close(fig)

    # precision-recall
    prd = FIG / "precision_recall"; prd.mkdir(parents=True, exist_ok=True)
    val = list(csv.DictReader((RES / "validation_summary.csv").open()))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, getter) in zip(axs, [("validation (synthetic injected events, pooled)", lambda m: next(((float(r["event_precision"]), float(r["event_recall"])) for r in val if r["method"] == m and r["dataset"] == "all"), None)),
                                         ("test (pooled SMAP+MSL counts)", lambda m: test_pr(summary, m))]):
        for m in meths:
            pr = getter(m)
            if pr:
                ax.scatter(pr[1], pr[0]); ax.annotate(m.split("_", 1)[0], (pr[1], pr[0]), fontsize=8)
        ax.set_xlabel("event recall"); ax.set_ylabel("event precision"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(prd / "event_precision_recall.png", dpi=110); plt.close(fig)

    # runtime/memory
    rd = FIG / "runtime_memory"; rd.mkdir(parents=True, exist_ok=True)
    tt = summary["timing_test"]
    names = list(tt)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].barh(names, [tt[n]["wall_s"] for n in names]); axs[0].set_xscale("log"); axs[0].set_xlabel("wall s, all 81 test channels")
    axs[1].barh(names, [tt[n]["peak_bytes"] / 2**20 for n in names]); axs[1].set_xlabel("tracemalloc peak MiB"); axs[1].set_yticks([])
    fig.suptitle("Single test run, one machine (8 cores); includes fit+calibration; machine-dependent", fontsize=8)
    fig.tight_layout(); fig.savefig(rd / "test_runtime_memory.png", dpi=110); plt.close(fig)

    # comparison tables
    pub = list(csv.DictReader((V2_ROOT.parent / "reports" / "smap_msl_results" / "smap_msl_summary.csv").open()))
    comp = []
    for r in val:
        comp.append({"split": "validation (synthetic injected events)", "method": r["method"], "dataset": r["dataset"], "event_precision": r["event_precision"],
                     "event_recall": r["event_recall"], "event_f1": r["event_f1"], "macro_channel_f1": r["macro_channel_f1"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                     "point_f1": r["point_f1"], "source": "oes32-v2 results/validation_summary.csv"})
    for m, v in summary["summary"].items():
        for d, a in v.items():
            ci = summary["bootstrap_95ci_event_f1_channel_level"][d][m]
            comp.append({"split": "test (single run)", "method": m, "dataset": d, **{k: round(a[k], 6) for k in ("event_precision", "event_recall", "event_f1", "macro_channel_f1", "point_f1")},
                         "tp": a["tp"], "fp": a["fp"], "fn": a["fn"], "event_f1_ci95_low": round(ci[0], 6), "event_f1_ci95_high": round(ci[1], 6), "source": "oes32-v2 results/final_test_summary.json"})
    for r in pub:
        if r["subset"] == "all_channels" and r["target_fp"] == "0.01":
            comp.append({"split": "test (published oes-resilience v0.5, target FP 0.01)", "method": f"v0.5_{r['method']}", "dataset": r["dataset"],
                         "event_precision": r["event_precision"], "event_recall": r["event_recall"], "event_f1": r["event_f1"], "macro_channel_f1": r["mean_channel_event_f1"],
                         "tp": r["tp_events"], "fp": r["fp_runs"], "fn": int(r["events"]) - int(r["tp_events"]), "point_f1": "",
                         "event_f1_ci95_low": r["event_f1_ci_low"], "event_f1_ci95_high": r["event_f1_ci_high"], "source": "reports/smap_msl_results/smap_msl_summary.csv"})
    fields = ["split", "method", "dataset", "event_precision", "event_recall", "event_f1", "event_f1_ci95_low", "event_f1_ci95_high", "macro_channel_f1", "tp", "fp", "fn", "point_f1", "source"]
    with (RES / "baseline_comparison.csv").open("w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=fields, restval=""); w.writeheader(); w.writerows(comp)
    sel = json.loads((RES / "validation_selection.json").read_text())
    (RES / "v1_reference.json").write_text(json.dumps({
        "method": "OES-32-v1, frozen (oes-resilience v0.5 code, imported read-only)",
        "rule": "score = 0.45 max|z| + 0.35 RMS(z) + 0.20 mean|z| over a 32-sample trailing window; z standardised on train mean/std; threshold by v1 calibration at 1 % train FP; alarm score >= T",
        "published_v0.5_test_event_f1": summary["v1_reproduces_published_v0.5"]["published"],
        "reproduced_in_v2_test_run": summary["v1_reproduces_published_v0.5"]["reproduced"],
        "reproduction_match": summary["v1_reproduces_published_v0.5"]["match"],
        "validation_split_metrics": sel["validation"]["v1"],
        "test_metrics": summary["summary"]["A_v1_frozen"],
        "test_bootstrap_ci95": {d: summary["bootstrap_95ci_event_f1_channel_level"][d]["A_v1_frozen"] for d in DATASETS},
    }, indent=1) + "\n")
    print(json.dumps(top, indent=1))


def test_pr(summary, m):
    if m not in summary["summary"]:
        return None
    tp = sum(summary["summary"][m][d]["tp"] for d in DATASETS); fp = sum(summary["summary"][m][d]["fp"] for d in DATASETS); fn = sum(summary["summary"][m][d]["fn"] for d in DATASETS)
    p, r, _ = core.prf(tp, fp, fn)
    return p, r


if __name__ == "__main__":
    main(sys.argv[1])
