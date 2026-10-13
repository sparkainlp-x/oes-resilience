"""Validation-only staged search for OES-32-v2. Never reads test files."""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from . import core
from .data import V2_ROOT, channels, inject, split_for, DATASETS
from .data import v1

SPACE = yaml.safe_load((V2_ROOT / "configs" / "search_space.yaml").read_text())
DEFAULT_NORM = {"scope": "per_channel", "epsilon": 1e-9, "clip": "none"}
DEFAULT_POST = {"hysteresis_low_high_ratio": 0.90, "min_alert_duration": 1, "max_merge_gap": 0, "cooldown": 0}
DEFAULT_COMP = {"feature": "none"}
TRIAL_FIELDS = ["trial_id", "stage", "status", "config", "complexity", "event_f1", "event_precision", "event_recall",
                "macro_channel_f1", "fp_events", "tp_events", "fn_events", "smap_event_f1", "msl_event_f1",
                "point_f1", "runtime_s", "error"]


# ------------------------------------------------------------ validation data (train files only)
def load_validation(data_dir: str | Path) -> list[dict]:
    out = []
    for i, c in enumerate(channels(data_dir)):
        train = np.asarray(np.load(Path(data_dir) / "train" / f"{c.chan_id}.npy", allow_pickle=False)[:, 0], dtype=np.float64)
        sp = split_for(train.size)
        if not sp.has_val:
            continue
        fit, clean = train[: sp.fit_end], train[sp.val_start:]
        inj, events, kinds = inject(clean, fit, i)
        out.append({"spacecraft": c.spacecraft, "chan_id": c.chan_id, "index": i, "fit": fit,
                    "clean": clean, "inj": inj, "events": events, "kinds": kinds})
    return out


def norm_stats(chans: list[dict], norm: dict) -> dict:
    if norm["scope"] == "per_channel":
        return {c["chan_id"]: core.robust_stats(c["fit"]) for c in chans}
    glob = {d: core.robust_stats(np.concatenate([c["fit"] for c in chans if c["spacecraft"] == d])) for d in DATASETS}
    return {c["chan_id"]: glob[c["spacecraft"]] for c in chans}


def complexity(cfg: dict) -> int:
    n = sum(cfg["normalization"][k] != DEFAULT_NORM[k] for k in DEFAULT_NORM)
    n += sum(cfg["postprocess"][k] != DEFAULT_POST[k] for k in DEFAULT_POST)
    n += cfg["window"]["smoothing"] != "none"
    n += cfg["window"]["aggregation"] in ("median", "topk_mean")
    n += 2 * (cfg["complementary"]["feature"] != "none")
    return int(n)


# ------------------------------------------------------------ scoring with caches
class Scorer:
    def __init__(self, chans: list[dict]):
        self.chans = chans
        self.cache: dict = {}
        self.comp_cache: dict = {}

    def scaled(self, norm: dict) -> dict:
        key = ("x", json.dumps(norm, sort_keys=True))
        if key not in self.cache:
            st = norm_stats(self.chans, norm)
            self.cache[key] = {c["chan_id"]: tuple(core.scale(c[s], *st[c["chan_id"]], norm["epsilon"], norm["clip"])
                                                   for s in ("fit", "clean", "inj")) for c in self.chans}
        return self.cache[key]

    def base(self, norm: dict, window: dict) -> dict:
        key = ("b", json.dumps(norm, sort_keys=True), json.dumps(window, sort_keys=True))
        if key not in self.cache:
            xs = self.scaled(norm)
            self.cache[key] = {cid: (core.oes_v2_score(x[1], {"window": window}), core.oes_v2_score(x[2], {"window": window}))
                               for cid, x in xs.items()}
        return self.cache[key]

    def comp(self, norm: dict, feature: str) -> dict:
        key = (json.dumps(norm, sort_keys=True), feature)
        if key not in self.comp_cache:
            xs = self.scaled(norm)
            res = {}
            for cid, (xf, xc, xi) in xs.items():
                if feature == "rolling_mad":
                    res[cid] = (core.rolling_mad_score(xc), core.rolling_mad_score(xi))
                else:
                    m = core.PCAModel.fit(xf)
                    res[cid] = (m.score(xc), m.score(xi))
            self.comp_cache[key] = res
        return self.comp_cache[key]

    def scores(self, cfg: dict) -> dict:
        b = self.base(cfg["normalization"], cfg["window"])
        comp = cfg["complementary"]
        if comp["feature"] == "none":
            return b
        cs = self.comp(cfg["normalization"], comp["feature"])
        out = {}
        for cid, (bc, bi) in b.items():
            cc, ci = cs[cid]
            na, nb = max(float(np.quantile(bc, 0.99)), 1e-12), max(float(np.quantile(cc, 0.99)), 1e-12)
            a_c, a_i, b_c, b_i = bc / na, bi / na, cc / nb, ci / nb
            how = comp["combination"]
            if how == "normalized_mean":
                out[cid] = ((a_c + b_c) / 2, (a_i + b_i) / 2)
            elif how == "max":
                out[cid] = (np.maximum(a_c, b_c), np.maximum(a_i, b_i))
            else:
                w = float(comp["weight"])
                out[cid] = (w * a_c + (1 - w) * b_c, w * a_i + (1 - w) * b_i)
        return out


def evaluate(scorer: Scorer, cfg: dict, return_rows: bool = False) -> dict:
    sc = scorer.scores(cfg)
    rule = cfg["threshold"]
    pooled = {d: np.concatenate([sc[c["chan_id"]][0] for c in scorer.chans if c["spacecraft"] == d]) for d in DATASETS} \
        if rule["kind"] == "global_quantile" else {}
    rows = []
    for c in scorer.chans:
        clean, inj = sc[c["chan_id"]]
        T = core.threshold_from(clean, rule, pooled.get(c["spacecraft"]))
        pred = core.alerts(inj, T, cfg["postprocess"])
        r = core.channel_eval(pred, c["events"], inj.size)
        r.update(spacecraft=c["spacecraft"], chan_id=c["chan_id"], threshold=T)
        rows.append(r)
    res = {"all": core.aggregate(rows), **{d: core.aggregate([r for r in rows if r["spacecraft"] == d]) for d in DATASETS}}
    if return_rows:
        res["rows"] = rows
    return res


def rank_key(t: dict):
    return (-round(t["event_f1"], 3), t["complexity"], -t["event_precision"], -t["macro_channel_f1"],
            t["fp_events"], t["runtime_s"], t["trial_id"])


# ------------------------------------------------------------ enumeration
def windows():
    for L in SPACE["window"]["length"]:
        for agg in SPACE["window"]["aggregation"]:
            ks = SPACE["window"]["topk"] if agg == "topk_mean" else [1]
            for k in ks:
                for sm in SPACE["window"]["smoothing"]:
                    yield {"length": L, "aggregation": agg, "topk": k, "smoothing": sm}


def thresholds():
    t = SPACE["threshold"]
    for q in t["per_channel_quantile"]:
        yield {"kind": "per_channel_quantile", "value": q}
    for q in t["global_quantile"]:
        yield {"kind": "global_quantile", "value": q}
    for m in t["robust_multiplier"]:
        yield {"kind": "robust_multiplier", "value": m}


def norms():
    n = SPACE["normalization"]
    for s, e, c in itertools.product(n["scope"], n["epsilon"], n["clip"]):
        yield {"scope": s, "epsilon": e, "clip": c}


def posts():
    p = SPACE["postprocess"]
    for r, d, g, c in itertools.product(p["hysteresis_low_high_ratio"], p["min_alert_duration"], p["max_merge_gap"], p["cooldown"]):
        yield {"hysteresis_low_high_ratio": r, "min_alert_duration": d, "max_merge_gap": g, "cooldown": c}


def comps():
    yield {"feature": "none"}
    for f in ("rolling_mad", "pca_residual"):
        for how in ("normalized_mean", "max"):
            yield {"feature": f, "combination": how}
        for w in SPACE["complementary"]["weight"]:
            yield {"feature": f, "combination": "weighted_sum", "weight": w}


# ------------------------------------------------------------ driver
def run(data_dir: str) -> None:
    chans = load_validation(data_dir)
    scorer = Scorer(chans)
    trials: list[dict] = []
    out = V2_ROOT / "experiments" / "all_trials.csv"
    out.parent.mkdir(exist_ok=True)
    f = out.open("w", newline="", encoding="utf-8")
    wr = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
    wr.writeheader()

    def trial(stage: str, cfg: dict) -> dict:
        t0 = time.perf_counter()
        rec = {"trial_id": len(trials) + 1, "stage": stage, "config": json.dumps(cfg, sort_keys=True), "complexity": complexity(cfg)}
        try:
            r = evaluate(scorer, cfg)
            a = r["all"]
            rec.update(status="ok", event_f1=a["event_f1"], event_precision=a["event_precision"], event_recall=a["event_recall"],
                       macro_channel_f1=a["macro_channel_f1"], fp_events=a["fp"], tp_events=a["tp"], fn_events=a["fn"],
                       smap_event_f1=r["SMAP"]["event_f1"], msl_event_f1=r["MSL"]["event_f1"], point_f1=a["point_f1"], error="")
        except Exception as exc:  # logged, never hidden
            rec.update(status="failed", event_f1=0.0, event_precision=0.0, event_recall=0.0, macro_channel_f1=0.0,
                       fp_events=10**9, tp_events=0, fn_events=0, smap_event_f1=0.0, msl_event_f1=0.0, point_f1=0.0,
                       error=f"{type(exc).__name__}: {exc}"[:300])
            traceback.print_exc()
        rec["runtime_s"] = round(time.perf_counter() - t0, 4)
        wr.writerow(rec)
        f.flush()
        trials.append(rec)
        return rec

    def top(stage: str, n: int) -> list[dict]:
        return [json.loads(t["config"]) for t in sorted((t for t in trials if t["stage"] == stage and t["status"] == "ok"), key=rank_key)[:n]]

    start = time.time()
    for w in windows():
        for th in thresholds():
            trial("stage1", {"normalization": DEFAULT_NORM, "window": w, "threshold": th, "postprocess": DEFAULT_POST, "complementary": DEFAULT_COMP})
    print("stage1 done", len(trials), round(time.time() - start), flush=True)
    for base in top("stage1", 5):
        for nm in norms():
            trial("stage2a", {**base, "normalization": nm})
    print("stage2a done", len(trials), round(time.time() - start), flush=True)
    for base in top("stage2a", 5):
        for p in posts():
            trial("stage2b", {**base, "postprocess": p})
    print("stage2b done", len(trials), round(time.time() - start), flush=True)
    for base in top("stage2b", 3):
        for cp in comps():
            trial("stage2c", {**base, "complementary": cp})
    print("stage2c done", len(trials), round(time.time() - start), flush=True)
    f.close()
    ok = sorted((t for t in trials if t["status"] == "ok"), key=rank_key)
    with (V2_ROOT / "experiments" / "validation_leaderboard.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=["rank"] + TRIAL_FIELDS)
        w.writeheader()
        for i, t in enumerate(ok, 1):
            w.writerow({"rank": i, **t})
    (V2_ROOT / "experiments" / "search_meta.json").write_text(json.dumps({
        "trials": len(trials), "failed": sum(t["status"] != "ok" for t in trials),
        "wall_clock_s": round(time.time() - start, 1), "validation_channels": len(chans),
        "validation_events": {d: sum(len(c["events"]) for c in chans if c["spacecraft"] == d) for d in DATASETS},
        "injected_types": {k: sum(c["kinds"].count(k) for c in chans) for k in ("spike", "level_shift", "drift", "variance_burst")},
    }, indent=1) + "\n")


if __name__ == "__main__":
    run(sys.argv[1])
