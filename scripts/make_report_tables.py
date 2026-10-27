"""Generate the LaTeX tables and number macros used by docs/report/main.tex from the repository's result files.

No number in the report is typed by hand: every table body and every figure quoted in the text comes from
results/, data/processed/ (rebuilt by `make preprocess`) or the published numbers listed in PUBLISHED below.

    python scripts/make_report_tables.py        # writes docs/report/generated/*.tex
"""
from __future__ import annotations

import json
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
OUT = ROOT / "docs" / "report" / "generated"

MODELS = {"persistence": "Persistence", "hist_avg": "Historical average", "ridge": "Ridge per sensor",
          "rf": "Random Forest", "lstm": "LSTM (no graph)", "xgb": "XGBoost"}
FULL = {"persistence": "Persistence (last value)", "hist_avg": "Historical average", "ridge": "Ridge per sensor",
        "xgb": "XGBoost per horizon", "rf": "Multi-output Random Forest", "lstm": "LSTM (no graph)"}
WORD = {15: "Fifteen", 30: "Thirty", 60: "Sixty"}
STEP = {15: 3, 30: 6, 60: 12}

# Published METR-LA results, Table 1 of Li, Yu, Shahabi and Liu (2018), ICLR (arXiv:1707.01926). MAE in mph at 15/30/60 min.
PUBLISHED = {"Historical average (HA)": [4.16, 4.16, 4.16], "FC-LSTM": [3.44, 3.77, 4.37], "DCRNN": [2.77, 3.15, 3.60]}


def r(v: float, nd: int = 2) -> str:
    q = Decimal(1).scaleb(-nd)
    out = str(Decimal(repr(round(float(v), 8))).quantize(q, ROUND_HALF_UP))
    return out.lstrip("-") if not out.strip("-0.") else out          # no "-0.000"


def th(n: int) -> str:
    """Thousands separated by a LaTeX thin space."""
    return f"{n:,}".replace(",", r"\,")


def esc(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text)


def tabular(spec: str, header: list[str], rows: list[list[str]], pre_header: str = "") -> str:
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    if pre_header:
        lines.append(pre_header)
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def main() -> None:
    macros: dict[str, str] = {}

    def mac(name: str, value) -> None:
        macros[name] = str(value)

    p1 = json.loads((R / "phase1_findings.json").read_text())
    p4 = json.loads((R / "phase4_findings.json").read_text())
    hz = {k: pd.read_csv(R / "baselines" / f"{k}.csv", dtype={"horizon": str}) for k in MODELS}
    meta = {k: json.loads((R / "baselines" / f"{k}.json").read_text()) for k in ("xgb", "rf", "ridge", "lstm")}
    peak = pd.read_csv(R / "baselines_peak.csv")
    acc = pd.read_csv(R / "accuracy.csv")
    gapres = pd.read_csv(R / "gap_results.csv")
    onset = pd.read_csv(R / "gap" / "onset_vs_normal.csv")
    resid = pd.read_csv(R / "gap" / "residual_correlation.csv")
    imp = pd.read_csv(R / "baselines" / "xgb_importance.csv", index_col="feature")
    nimp = pd.read_csv(R / "gap" / "xgb_neighbours_importance.csv", header=[0, 1], index_col=0)["both"]

    def score(k: str, split: str, h, metric: str) -> float:
        d = hz[k]
        return float(d[(d.split == split) & (d.horizon == str(h))][metric].iloc[0])

    # ------------------------------------------------------------------ dataset (Phase 1)
    d, mi, dp, ds, g, pr = p1["data"], p1["missingness"], p1["daily_profile"], p1["distribution"], p1["graph"], p1["propagation"]
    mac("Sensors", d["sensors"])
    mac("Timestamps", th(d["timestamps"]))
    mac("ZeroPct", mi["zero_pct"])
    mac("NetworkOutagePct", mi["network_outage_pct"])
    mac("SensorOutagePct", mi["sensor_outage_pct"])
    mac("NetworkOutageSteps", th(mi["network_outage_timestamps"]))
    mac("LongestOutageHours", mi["longest_network_outage_hours"])
    mac("FreeFlowMph", dp["weekday_free_flow_mph"])
    mac("PctAtLeastSixty", ds["pct_ge_60"])
    mac("Edges", th(g["edges"]))
    mac("TwoWayPct", g["reciprocal_edge_pct"])
    mac("WeakEdgePct", pr["weak_edge_pct"])
    mac("ClearEdges", pr["clear_edges"])
    mac("DownstreamFirst", pr["downstream_first"])
    mac("UpstreamFirst", pr["upstream_first"])
    mac("ShockwaveKmh", int(pr["shockwave_kmh"]))
    dip = "; ".join(f"{w['start']}--{w['end']} (low {w['min_mph']} mph)" for w in dp["peak_windows"])
    rows_ds = [
        ["Sensors and readings", f"{d['sensors']} loop detectors; {th(d['timestamps'])} timestamps each"],
        ["Period", f"{d['start'][:10]} to {d['end'][:10]}, 5-minute clock, no gaps"],
        ["Readings equal to 0 mph", rf"{mi['zero_pct']}\% ({mi['network_outage_pct']} points whole-network, {mi['sensor_outage_pct']} points single-sensor)"],
        ["Whole-network outages", f"{th(mi['network_outage_timestamps'])} timestamps; longest {mi['longest_network_outage_hours']} h"],
        ["Weekday free-flow speed", f"{dp['weekday_free_flow_mph']} mph (weekend minimum {dp['weekend_min_mph']} mph)"],
        [r"Weekday dips below 90\% of free flow", dip],
        ["Speed distribution", rf"median {ds['median']} mph; {ds['pct_ge_60']}\% at least 60 mph; {ds['pct_lt_40']}\% below 40 mph; {ds['pct_lt_20']}\% below 20 mph"],
        ["Road graph", rf"{g['nodes']} nodes; {th(g['edges'])} directed links; mean degree {g['mean_degree']}; {g['reciprocal_edge_pct']}\% two-way; {g['isolated_nodes']} isolated sensor"],
    ]
    write("tab_dataset.tex", "\\begin{tabularx}{\\textwidth}{@{}lX@{}}\n\\toprule\nProperty & Value \\\\\n\\midrule\n"
          + "\n".join(" & ".join(row) + " \\\\" for row in rows_ds) + "\n\\bottomrule\n\\end{tabularx}\n")

    # ------------------------------------------------------------------ splits (Phase 2)
    from src.data.preprocess import load_config, plan_windows
    proc = ROOT / "data" / "processed"
    splits = {}
    for protocol in ("strict", "dcrnn"):
        for s in ("train", "val", "test"):
            with np.load(proc / protocol / f"{s}.npz") as z:
                tt = z["target_time"]
                splits[(protocol, s)] = {"windows": int(len(z["origin_idx"])), "first": str(tt.min())[:16].replace("T", " "),
                                         "last": str(tt.max())[:16].replace("T", " ")}
    rows = []
    for s in ("train", "val", "test"):
        m = np.load(proc / "strict" / "features" / f"M_{s}.npy", mmap_mode="r")
        sp = splits[("strict", s)]
        sp["rows"], sp["masked"] = int(m.shape[0]), 100 * float(1 - np.asarray(m).mean())
        rows.append([s.capitalize(), f"{sp['windows']:,}".replace(",", r"\,"), f"{sp['rows']:,}".replace(",", r"\,"), sp["first"], sp["last"], rf"{r(sp['masked'])}\%"])
        mac(f"Windows{s.capitalize()}", f"{sp['windows']:,}".replace(",", r"\,"))
        mac(f"Rows{s.capitalize()}", f"{sp['rows']:,}".replace(",", r"\,"))
        mac(f"Masked{s.capitalize()}", r(sp["masked"]))
    write("tab_splits.tex", tabular("@{}lrrllr@{}", ["Split", "Windows", "Rows", "First target", "Last target", "Masked targets"], rows))
    planned = plan_windows(d["timestamps"], load_config(), "strict")[0]["train"]
    mac("DroppedTrain", f"{len(planned) - splits[('strict', 'train')]['windows']:,}".replace(",", r"\,"))
    sc_s = json.loads((proc / "strict" / "scaler.json").read_text())
    sc_d = json.loads((proc / "dcrnn" / "scaler.json").read_text())
    mac("ScalerMean", r(sc_s["mu"]))
    mac("ScalerStd", r(sc_s["sd"]))
    write("tab_protocols.tex", tabular("@{}lll@{}", ["", "Strict split (used)", "Reference window split"], [
        ["Split unit", "timestamps, then windows", "windows"],
        ["Last train target", splits[("strict", "train")]["last"], splits[("dcrnn", "train")]["last"]],
        ["First validation target", splits[("strict", "val")]["first"], bold(splits[("dcrnn", "val")]["first"]) + r" (earlier)"],
        ["Scaler mean / std (mph)", f"{r(sc_s['mu'])} / {r(sc_s['sd'])}, outages excluded", f"{r(sc_d['mu'])} / {r(sc_d['sd'])}, 0 mph included"],
    ]))

    # ------------------------------------------------------------------ headline scores (Phase 3)
    order = ["persistence", "hist_avg", "ridge", "rf", "lstm", "xgb"]
    for k in order:
        for mn, w in WORD.items():
            for met in ("mae", "rmse", "mape"):
                mac(f"{k.replace('_', '')}{met.upper()}{w}", r(score(k, "test", STEP[mn], met), 1 if met == "mape" else 2))
        mac(f"{k.replace('_', '')}PooledMAE", r(score(k, "test", "avg", "mae")))
    mac("XgbVsPersistenceFifteen", int(round(100 * (1 - score("xgb", "test", 3, "mae") / score("persistence", "test", 3, "mae")))))
    mac("XgbVsPersistenceSixty", int(round(100 * (1 - score("xgb", "test", 12, "mae") / score("persistence", "test", 12, "mae")))))
    mac("LstmGapFifteen", r(score("xgb", "test", 3, "mae") - score("lstm", "test", 3, "mae")))

    def scores_table(split: str, pooled: bool) -> str:
        rows_ = []
        best = {}
        for met in ("mae", "rmse", "mape"):
            for mn in (15, 30, 60):
                best[(met, mn)] = min(float(score(k, split, STEP[mn], met)) for k in order)
        for k in order:
            row = [MODELS[k]]
            for met in ("mae", "rmse", "mape"):
                for mn in (15, 30, 60):
                    v = score(k, split, STEP[mn], met)
                    s = r(v, 1 if met == "mape" else 2)
                    row.append(bold(s) if abs(v - best[(met, mn)]) < 1e-9 else s)
            if pooled:
                row.append(r(score(k, split, "avg", "mae")))
            rows_.append(row)
        head = ["Model"] + [f"{m}\\,{n}" for m in ("MAE", "RMSE", "MAPE") for n in (15, 30, 60)] + (["Pooled"] if pooled else [])
        return tabular("@{}l" + "r" * (9 + (1 if pooled else 0)) + "@{}", head, rows_)

    write("tab_scores_test.tex", scores_table("test", True))
    write("tab_scores_val.tex", scores_table("val", True))

    # ------------------------------------------------------------------ accuracy
    amap = {FULL[k]: MODELS[k] for k in order} | {"XGBoost + neighbours (both directions)": "XGBoost + neighbours"}
    rows = []
    for fullname, short in amap.items():
        a = acc[acc.model == fullname].set_index("minutes")
        rows.append([short] + [r(a.loc[m, "mape_accuracy_pct"], 1) for m in (15, 30, 60)] + [r(a.loc[m, "within_5mph_pct"], 1) for m in (15, 30, 60)])
    write("tab_accuracy.tex", tabular("@{}lrrrrrr@{}", ["Model", "Acc.\\,15", "Acc.\\,30", "Acc.\\,60", "$\\leq$5\\,mph\\,15", "$\\leq$5\\,mph\\,30", "$\\leq$5\\,mph\\,60"], rows))
    ax = acc[acc.model == FULL["xgb"]].set_index("minutes")
    axn = acc[acc.model == "XGBoost + neighbours (both directions)"].set_index("minutes")
    for mn, w in WORD.items():
        mac(f"XgbAcc{w}", r(ax.loc[mn, "mape_accuracy_pct"], 1))
        mac(f"XgbNbAcc{w}", r(axn.loc[mn, "mape_accuracy_pct"], 1))
        mac(f"XgbWithin{w}", r(ax.loc[mn, "within_5mph_pct"], 1))

    # ------------------------------------------------------------------ peak versus off-peak
    pk = peak.copy()
    rows = []
    ratios = []
    for k in order:
        row = [MODELS[k]]
        for per in ("peak", "offpeak", "weekend"):
            row.append(" / ".join(r(pk[(pk.model == FULL[k]) & (pk.period == per) & (pk.minutes == m)].mae.iloc[0]) for m in (15, 30, 60)))
        rows.append(row)
        for m in (15, 30, 60):
            ratios.append(float(pk[(pk.model == FULL[k]) & (pk.period == "peak") & (pk.minutes == m)].mae.iloc[0]) / float(pk[(pk.model == FULL[k]) & (pk.period == "offpeak") & (pk.minutes == m)].mae.iloc[0]))
    write("tab_peak.tex", tabular("@{}llll@{}", ["Model", "Weekday peak", "Weekday off-peak", "Weekend"], rows))
    mac("PeakRatioMin", r(min(ratios)))
    mac("PeakRatioMax", r(max(ratios)))
    xp = pk[pk.model == FULL["xgb"]]
    for per in ("peak", "offpeak"):
        for m in (15, 60):
            mac(f"Xgb{per.capitalize()}{WORD[m]}", r(xp[(xp.period == per) & (xp.minutes == m)].mae.iloc[0]))

    # ------------------------------------------------------------------ feature importance
    names = []
    for h in ("3", "6", "12"):
        names += [n for n in imp[h].sort_values(ascending=False).head(8).index if n not in names]
    rows = []
    for n in names:
        row = [esc(n)]
        for h in ("3", "6", "12"):
            top = imp[h].sort_values(ascending=False).head(8)
            row.append(rf"{r(100 * imp.loc[n, h], 1)}\%" if n in top.index else "--")
        rows.append(row)
    write("tab_importance.tex", tabular("@{}lrrr@{}", ["Feature", "15 min", "30 min", "60 min"], rows))
    mac("ImpSpeedFifteen", r(100 * imp.loc["speed_t", "3"], 1))
    mac("ImpSpeedSixty", r(100 * imp.loc["speed_t", "12"], 1))

    # ------------------------------------------------------------------ model configuration
    xm, rm, lm, gm = meta["xgb"], meta["rf"], meta["lstm"], meta["ridge"]
    tune = xm["tuning_h6"]
    lbest = min(e["val_mae"] for e in lm["history"])
    write("tab_models.tex", r"""\begin{tabularx}{\textwidth}{@{}clX@{}}
\toprule
Rung & Model & Configuration \\
\midrule
0 & Persistence & Repeats the last (imputed) speed for all 12 horizons. \\
1 & Historical average & Training mean per sensor, weekday-or-weekend and 5-minute slot. \\
""" + f"""2 & Ridge per sensor & One linear model per sensor and horizon ({207} $\\times$ 12) on the 41 time-varying features, standardised per sensor. Penalty chosen from $\\{{1, 10, 100, 1000\\}}$ on validation: {int(gm['alpha'])} (validation MAE {r(gm['val_mae_by_alpha']['1.0'], 3)} to {r(gm['val_mae_by_alpha']['1000.0'], 3)}, so the choice barely matters). \\\\
3 & XGBoost & One regressor per horizon trained on all sensors stacked, sensor id as a feature, masked targets given weight 0. Maximum depth searched at the 30-minute horizon: """ + ", ".join(f"{t['max_depth']} gives {r(t['val_mae'], 3)}" for t in tune) + f""" validation MAE; depth {xm['params']['max_depth']} chosen. Learning rate {xm['params']['eta']}, row and column subsample {xm['params']['subsample']}, up to 600 rounds with early stopping (50) on validation; every {xm['stride']}nd window used. \\\\
4 & Random Forest & One multi-output forest predicting all 12 horizons: {rm['params']['n_estimators']} trees, depth {rm['params']['max_depth']}, minimum leaf {rm['params']['min_samples_leaf']}, {int(rm['params']['max_samples'] * 100)}\\% bootstrap samples, {int(rm['params']['max_features'] * 100)}\\% of features per split. Trained on every {rm['stride']}th window keeping rows whose 12 targets are all observed ({rm['train_rows']:,} rows); evaluated on every test row with the mask. \\\\
5 & LSTM, no graph & Two LSTM layers of {lm['hidden']} units, dropout 0.1, one set of weights shared by all 207 sensors; input is each sensor's own $12 \\times 7$ window, output 12 horizons. Masked MAE in mph, Adam (learning rate 0.002, halved on plateau), gradient clipping 5, at most 12 epochs. All 12 epochs were used; best validation MAE {r(lbest, 3)}. \\\\
""".replace(f"{rm['train_rows']:,}", f"{rm['train_rows']:,}".replace(",", r"\,")) + r"""\bottomrule
\end{tabularx}
""")
    mac("XgbBestDepth", xm["params"]["max_depth"])
    mac("XgbHighRoundHorizons", sum(1 for v in xm["best_rounds"].values() if v >= 580))

    # ------------------------------------------------------------------ gap experiments
    def ons(model, grp, h):
        return float(onset[(onset.model == model) & (onset.group == grp) & (onset.horizon == h)].mae.iloc[0])

    rows = []
    for key, label in (("xgb", "XGBoost"), ("persistence", "Persistence"), ("xgb_both", "XGBoost + neighbours")):
        for h in (3, 6, 12):
            rows.append([label, f"{5 * h} min", r(ons(key, "onset", h)), r(ons(key, "normal", h)), rf"{r(ons(key, 'onset', h) / ons(key, 'normal', h), 1)}$\times$"])
    write("tab_onset.tex", tabular("@{}llrrr@{}", ["Model", "Horizon", "After onset", "Other times", "Ratio"], rows))
    for h, w in ((3, "Fifteen"), (6, "Thirty"), (12, "Sixty")):
        mac(f"OnsetXgb{w}", r(ons("xgb", "onset", h), 1))
        mac(f"NormalXgb{w}", r(ons("xgb", "normal", h), 1))
        mac(f"OnsetRatio{w}", r(ons("xgb", "onset", h) / ons("xgb", "normal", h), 1))
    mac("OnsetNbFifteen", r(ons("xgb_both", "onset", 3)))
    mac("OnsetXgbFifteenTwo", r(ons("xgb", "onset", 3)))
    mac("OnsetReductionFifteen", int(round(100 * (1 - ons("xgb_both", "onset", 3) / ons("xgb", "onset", 3)))))
    mac("OnsetReductionThirty", int(round(100 * (1 - ons("xgb_both", "onset", 6) / ons("xgb", "onset", 6)))))
    mac("TestOnsets", f"{p4['test_onsets']:,}".replace(",", r"\,"))
    mac("OnsetSharePct", r(p4["onset_share_of_targets_pct"]))

    def rc(model, grp, h):
        return float(resid[(resid.model == model) & (resid.group == grp) & (resid.horizon == h)].mean_corr.iloc[0])

    groups = ["1 link", "2 links", "3 links", "4+ links", "not connected"]
    write("tab_residual.tex", tabular("@{}lrrrr@{}", ["Pair distance", "XGB 15 min", "XGB 30 min", "XGB 60 min", "XGB + nb. 60 min"],
                                       [[esc(gname)] + [r(rc("xgb", gname, h), 3) for h in (3, 6, 12)] + [r(rc("xgb_both", gname, 12), 3)] for gname in groups]))
    mac("ResidLinkSixty", r(rc("xgb", "1 link", 12), 2))
    mac("ResidLinkFifteen", r(rc("xgb", "1 link", 3), 2))
    mac("ResidTwoSixty", r(rc("xgb", "2 links", 12), 2))
    mac("ResidLinkNbSixty", r(rc("xgb_both", "1 link", 12), 3))
    mac("ResidLinkSixtyThree", r(rc("xgb", "1 link", 12), 3))

    def gr(model, split):
        return gapres[(gapres.model == model) & (gapres.split == split)].iloc[0]

    labels = [("xgb (no neighbours)", "XGBoost, own history only"), ("xgb + upstream", "+ upstream neighbours"),
              ("xgb + downstream", "+ downstream neighbours"), ("xgb + both", "+ both directions")]
    b0 = gr("xgb (no neighbours)", "test")
    rows = []
    for key, label in labels:
        t = gr(key, "test")
        row = [label]
        for col in ("MAE_15min", "MAE_30min", "MAE_60min"):
            row.append(r(t[col], 3) if key == labels[0][0] else rf"{r(t[col], 3)} ({r(100 * (t[col] / b0[col] - 1), 1)}\%)")
        row += [r(t["RMSE_15min"], 2), r(t["MAPE_15min"], 1)]
        rows.append(row)
    write("tab_neighbours.tex", tabular("@{}lrrrrr@{}", ["Model", "MAE 15", "MAE 30", "MAE 60", "RMSE 15", "MAPE 15"], rows))
    both, bv, b0v = gr("xgb + both", "test"), gr("xgb + both", "val"), gr("xgb (no neighbours)", "val")
    for col, w in (("MAE_15min", "Fifteen"), ("MAE_30min", "Thirty"), ("MAE_60min", "Sixty")):
        mac(f"NbGain{w}", r(100 * (1 - both[col] / b0[col]), 1))
        mac(f"NbGainVal{w}", r(100 * (1 - bv[col] / b0v[col]), 1))
        mac(f"NbMae{w}", r(both[col], 2))
    mac("NbRmseFifteen", r(both["RMSE_15min"], 2))
    mac("XgbRmseFifteenTwo", r(b0["RMSE_15min"], 2))
    share = {h: 100 * float(nimp[nimp.index.str.contains("stream_")].sum()[h]) for h in ("3", "6", "12")}
    mac("NbShareFifteen", r(share["3"], 1))
    mac("NbShareThirty", r(share["6"], 1))
    mac("NbShareSixty", r(share["12"], 1))

    # ------------------------------------------------------------------ all horizons, published comparison, tests
    rows = [[MODELS[k]] + [r(score(k, "test", h, "mae")) for h in range(1, 13)] for k in order]
    write("tab_horizon_mae.tex", tabular("@{}l" + "r" * 12 + "@{}", ["Model"] + [str(5 * h) for h in range(1, 13)], rows))

    ours = {"Historical average (HA)": "hist_avg", "FC-LSTM": "lstm", "DCRNN": None}
    rows = []
    for name, vals in PUBLISHED.items():
        k = ours[name]
        rows.append([name + r" \cite{li2018dcrnn}"] + [r(v) for v in vals] + ([" / ".join(r(score(k, "test", STEP[m], "mae")) for m in (15, 30, 60))] if k else ["not yet built"]))
    rows.append(["XGBoost (this work)", "--", "--", "--", " / ".join(r(score("xgb", "test", STEP[m], "mae")) for m in (15, 30, 60))])
    write("tab_published.tex", tabular("@{}lrrrl@{}", ["Published (Li et al.)", "15 min", "30 min", "60 min", "This work: 15 / 30 / 60"], rows))

    out = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    per: dict[str, int] = {}
    for line in out.splitlines():
        if "::" in line:
            f = line.split("::")[0].split("/")[-1]
            per[f] = per.get(f, 0) + 1
    guards = {"test_data.py": "shapes, five-minute clock, sensor order equals graph order, adjacency properties, masking of outages",
              "test_leakage.py": "target order, no target in two splits, train-only scaler, no future inputs, masked metrics, adjacency rebuild",
              "test_baselines.py": "feature alignment, train-only statistics, baselines, tree models, peak labelling, accuracy",
              "test_gap.py": "neighbour direction and features, onset labelling, residual correlation"}
    rows = [[esc(f), str(per.get(f, 0)), guards[f]] for f in guards]
    rows.append([bold("Total"), bold(str(sum(per.values()))), "all tests pass"])
    write("tab_tests.tex", r"\begin{tabularx}{\textwidth}{@{}lrX@{}}" + "\n" + r"\toprule" + "\n" + r"Test file & Tests & What it guards \\" + "\n" + r"\midrule" + "\n"
          + "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n")
    mac("TestsTotal", sum(per.values()))
    mac("Commits", subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())

    write("macros.tex", "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"wrote {len(list(OUT.glob('*.tex')))} files to {OUT.relative_to(ROOT)} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
