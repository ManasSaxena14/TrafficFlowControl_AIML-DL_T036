"""Collect every number used in the project report from the files in this repository.

Nothing in the report is typed by hand: tables and figures are generated from results/, reports/figures/,
data/processed/ (when present) and git history. Run from the repository root:

    python scripts/report/collect_data.py            # writes docs/report/report_data.json
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
MODELS = {"persistence": "Persistence (last value)", "hist_avg": "Historical average", "ridge": "Ridge per sensor",
          "xgb": "XGBoost per horizon", "rf": "Multi-output Random Forest", "lstm": "LSTM (no graph)"}


def sh(*args) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    out: dict = {}
    out["phase1"] = json.loads((R / "phase1_findings.json").read_text())
    out["phase4_findings"] = json.loads((R / "phase4_findings.json").read_text())

    # ---- Phase 2: splits, scaler, masking (needs data/processed; produced by `make preprocess`)
    proc = ROOT / "data" / "processed" / "strict"
    splits = {}
    for s in ("train", "val", "test"):
        with np.load(proc / f"{s}.npz") as z:
            tt = z["target_time"]
            splits[s] = {"windows": int(len(z["origin_idx"])), "first_target": str(tt.min())[:16].replace("T", " "),
                         "last_target": str(tt.max())[:16].replace("T", " ")}
        m = np.load(proc / "features" / f"M_{s}.npy", mmap_mode="r")
        splits[s]["masked_target_pct"] = round(100 * float(1 - np.asarray(m).mean()), 2)
        splits[s]["rows"] = int(m.shape[0])
    out["splits"] = splits
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.preprocess import load_config, plan_windows
    planned = plan_windows(out["phase1"]["data"]["timestamps"], load_config(), "strict")[0]["train"]
    out["dropped_train_windows"] = int(len(planned) - splits["train"]["windows"])   # windows removed by the 90%-masked rule
    out["scaler_strict"] = json.loads((proc / "scaler.json").read_text())
    out["scaler_dcrnn"] = json.loads((ROOT / "data" / "processed" / "dcrnn" / "scaler.json").read_text())
    dc = {}
    for s in ("train", "val", "test"):
        with np.load(ROOT / "data" / "processed" / "dcrnn" / f"{s}.npz") as z:
            tt = z["target_time"]
            dc[s] = {"windows": int(len(z["origin_idx"])), "first_target": str(tt.min())[:16].replace("T", " "),
                     "last_target": str(tt.max())[:16].replace("T", " ")}
    out["splits_dcrnn"] = dc

    # ---- Phase 3 scores
    b = pd.read_csv(R / "baselines.csv")
    out["scores"] = b.to_dict("records")
    horizon = {}
    for k, label in MODELS.items():
        d = pd.read_csv(R / "baselines" / f"{k}.csv", dtype={"horizon": str})
        horizon[label] = {sp: {m: [round(float(d[(d.split == sp) & (d.horizon == str(h))][m].iloc[0]), 3) for h in range(1, 13)]
                               for m in ("mae", "rmse", "mape")} for sp in ("val", "test")}
        pooled = d[d.horizon == "avg"].set_index("split")
        horizon[label]["pooled"] = {sp: {m: round(float(pooled.loc[sp, m]), 3) for m in ("mae", "rmse", "mape")}
                                    for sp in ("val", "test")}
    out["horizon"] = horizon
    out["peak"] = pd.read_csv(R / "baselines_peak.csv").to_dict("records")
    out["model_meta"] = {k: json.loads((R / "baselines" / f"{k}.json").read_text()) for k in ("xgb", "rf", "ridge", "lstm")}
    imp = pd.read_csv(R / "baselines" / "xgb_importance.csv", index_col="feature")
    out["xgb_importance"] = {h: {k: round(float(v), 4) for k, v in imp[h].sort_values(ascending=False).head(8).items()}
                             for h in ("3", "6", "12")}

    # ---- Phase 4
    out["onset"] = pd.read_csv(R / "gap" / "onset_vs_normal.csv").to_dict("records")
    out["residual"] = pd.read_csv(R / "gap" / "residual_correlation.csv").to_dict("records")
    out["gap_results"] = pd.read_csv(R / "gap_results.csv").to_dict("records")
    ni = pd.read_csv(R / "gap" / "xgb_neighbours_importance.csv", header=[0, 1], index_col=0)["both"]
    share = ni[ni.index.str.contains("stream_")].sum()
    out["neighbour_feature_gain_pct"] = {h: round(100 * float(share[h]), 1) for h in ("3", "6", "12")}

    # ---- repository facts
    log = sh("git", "log", "--reverse", "--format=%h|%an|%ad|%s", "--date=short").splitlines()
    out["commits"] = [dict(zip(("hash", "author", "date", "subject"), line.split("|", 3))) for line in log]
    out["head"] = sh("git", "rev-parse", "--short", "HEAD")
    tests = sh(str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "--collect-only", "-q").splitlines()
    per_file: dict[str, int] = {}
    for line in tests:
        if "::" in line:
            f = line.split("::")[0].split("/")[-1]
            per_file[f] = per_file.get(f, 0) + 1
    out["tests"] = per_file
    loc = {}
    for d in ("src", "tests"):
        loc[d] = sum(len(p.read_text().splitlines()) for p in (ROOT / d).rglob("*.py"))
    out["loc"] = loc
    out["loc"]["app.py"] = len((ROOT / "app.py").read_text().splitlines())
    (ROOT / "docs" / "report" / "report_data.json").write_text(json.dumps(out, indent=1))
    print("wrote docs/report/report_data.json;", len(out["commits"]), "commits;", sum(per_file.values()), "tests")


if __name__ == "__main__":
    main()
