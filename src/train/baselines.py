"""Phase 3 runner: fit one baseline, score it on val and test, store per-horizon results and predictions.

    python -m src.train.baselines persistence hist_avg ridge      # (also: xgb rf)
    python -m src.train.baselines --table                         # merge results/baselines/*.csv -> results/baselines.csv

Hyper-parameters are chosen on the validation block only. Test numbers are written but never used to choose anything.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.data.dataset import PROCESSED, load_scaler, load_split
from src.data.load import load_speed
from src.data.preprocess import load_config, split_bounds
from src.eval.evaluate import REPORT_HORIZONS, evaluate_rows
from src.features.build import load_features
from src.models import simple, trees

ROOT = PROCESSED.parents[1]
RESULTS = ROOT / "results" / "baselines"
PRED_DIR = PROCESSED / "strict" / "preds"
LADDER = {"persistence": (0, "Persistence (last value)"), "hist_avg": (1, "Historical average"),
          "ridge": (2, "Ridge per sensor"), "xgb": (3, "XGBoost per horizon"),
          "rf": (4, "Multi-output Random Forest"), "lstm": (5, "LSTM (no graph)")}


def train_end() -> int:
    return split_bounds(len(load_speed()), load_config()["split"])["train"][1]


def score(name: str, predict, extra: dict | None = None) -> pd.DataFrame:
    """predict(split) -> rows (S*N, 12) in mph. Scores val and test, stores results + predictions."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("val", "test"):
        _, Y, M = load_features(split)
        P = np.asarray(predict(split), dtype=np.float32)
        np.save(PRED_DIR / f"{name}_{split}.npy", P)
        df = evaluate_rows(Y, P, M)
        df.insert(0, "split", split)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / f"{name}.csv", index=False)
    if extra:
        (RESULTS / f"{name}.json").write_text(json.dumps(extra, indent=2, default=float))
    v, t = (out[(out.split == s) & (out.horizon == "avg")].iloc[0] for s in ("val", "test"))
    print(f"[{name}] avg MAE val {v.mae:.3f} | test {t.mae:.3f}  RMSE test {t.rmse:.3f}  MAPE test {t.mape:.2f}%")
    return out


def run_persistence():
    return score("persistence", lambda split: simple.persistence(load_features(split)[0]))


def run_hist_avg():
    model = simple.HistoricalAverage().fit(load_speed(), train_end())
    return score("hist_avg", lambda split: model.predict(load_split(split, keys=["target_time"])["target_time"]))


def run_ridge():
    F, Y, M = load_features("train")
    Fv, Yv, Mv = load_features("val")
    model = simple.RidgePerSensor().fit(F, Y, M, Fv, Yv, Mv)
    return score("ridge", lambda split: model.predict(load_features(split)[0]),
                 {"alpha": model.alpha, "val_mae_by_alpha": model.val_mae})


def run_xgb():
    train, val = load_features("train"), load_features("val")
    model = trees.XGBPerHorizon()
    tuning = model.tune(train, val)
    print("  tuning on h=6 (val MAE):\n" + tuning.to_string(index=False))
    model.fit(train, val)
    model.save(ROOT / "checkpoints" / "xgb")
    imp = model.importance()
    imp.to_csv(RESULTS / "xgb_importance.csv")
    return score("xgb", lambda split: model.predict(load_features(split)[0]),
                 {"stride": model.stride, "params": model.params, "best_rounds": model.best_rounds,
                  "tuning_h6": tuning.to_dict("records")})


def run_rf():
    model = trees.RandomForestMulti().fit(load_features("train"))
    return score("rf", lambda split: model.predict(load_features(split)[0]),
                 {"stride": model.stride, "train_rows": model.n_train_rows, "params": model.params})


def make_table() -> pd.DataFrame:
    """results/baselines.csv: MAE / RMSE / MAPE at 15 / 30 / 60 min (and pooled) for every rung that has been run."""
    rows = []
    for name, (rung, label) in LADDER.items():
        f = RESULTS / f"{name}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype={"horizon": str})
        for split in ("val", "test"):
            d = df[df.split == split].set_index("horizon")
            r = {"rung": rung, "model": label, "split": split}
            for h, minutes in REPORT_HORIZONS.items():
                for m in ("mae", "rmse", "mape"):
                    r[f"{m.upper()}_{minutes}min"] = round(float(d.loc[str(h), m]), 3)
            for m in ("mae", "rmse", "mape"):
                r[f"{m.upper()}_avg"] = round(float(d.loc["avg", m]), 3)
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "baselines.csv", index=False)
    return table


RUNNERS = {"persistence": run_persistence, "hist_avg": run_hist_avg, "ridge": run_ridge,
           "xgb": run_xgb, "rf": run_rf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", choices=list(RUNNERS))
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    load_scaler()   # fail early if Phase 2 has not been run
    for m in args.models:
        t0 = time.time()
        RUNNERS[m]()
        print(f"   {m}: {time.time() - t0:.0f}s")
    if args.table or args.models:
        print(make_table().to_string(index=False))


if __name__ == "__main__":
    main()
