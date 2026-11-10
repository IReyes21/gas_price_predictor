#!/usr/bin/env python3
"""Gas Price Predictor - DS 305 prototype.

Forecasts next week's U.S. regular gasoline price from crude oil prices, recent pump
prices and seasonality, and explains every forecast.

    python src/gas_predictor.py              # uses data/*.csv if present, else the synthetic demo
    python src/gas_predictor.py --demo       # force the synthetic demo
    python src/gas_predictor.py --no-figures # skip PNG charts

Outputs land in outputs/ (metrics.csv, latest_forecast.json, dashboard.html, figures/).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_sources import load_weekly  # noqa: E402
from explain import feature_table, forest_explain, linear_explain, sentence, to_groups, top_drivers  # noqa: E402
from features import FEATURE_GROUPS, FEATURE_LABELS, FEATURES, make_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE, LINEAR, FOREST = "Last-week baseline", "Linear regression", "Random forest"


def fit_models(Xtr, ytr, seed):
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(Xtr)
    lin = LinearRegression().fit(scaler.transform(Xtr), ytr)
    t_lin = time.perf_counter() - t0
    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=8, max_features=0.5, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    t_rf = time.perf_counter() - t0
    return scaler, lin, rf, t_lin, t_rf


def predict_change(name, models, X):
    scaler, lin, rf = models
    if name == BASELINE:
        return np.zeros(len(X))
    if name == LINEAR:
        return lin.predict(scaler.transform(X))
    return rf.predict(X)


def explain(name, models, X):
    scaler, lin, rf = models
    return linear_explain(lin, scaler, X) if name == LINEAR else forest_explain(rf, X)


def score(price_true, price_pred, chg_true, chg_pred, spike_mask):
    err = price_pred - price_true
    moved = chg_true != 0
    direction = float(np.mean(np.sign(chg_pred[moved]) == np.sign(chg_true[moved]))) if chg_pred.any() else None
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_pct": float(np.mean(np.abs(err / price_true)) * 100),
        "direction_hit_rate": direction,
        "MAE_on_spike_weeks": float(np.mean(np.abs(err[spike_mask]))),
        "MAE_on_normal_weeks": float(np.mean(np.abs(err[~spike_mask]))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--demo", action="store_true", help="force synthetic demo data")
    ap.add_argument("--test-frac", type=float, default=0.2, help="share of weeks held out, chronologically")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data + features ---------------------------------------------------------------
    df, source, is_demo = load_weekly(args.data_dir, args.demo)
    feats = make_features(df).dropna(subset=FEATURES).reset_index(drop=True)
    forecast_row = feats.iloc[[-1]]
    assert forecast_row["target_chg"].isna().all(), "last row must be the not-yet-observed forecast week"
    labelled = feats.dropna(subset=["target_chg"]).reset_index(drop=True)
    n_test = int(round(len(labelled) * args.test_frac))
    train, test = labelled.iloc[:-n_test], labelled.iloc[-n_test:]  # chronological: never shuffle a time series
    Xtr, ytr = train[FEATURES].to_numpy(), train["target_chg"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test["target_chg"].to_numpy()
    Xfc = forecast_row[FEATURES].to_numpy()

    scaler, lin, rf, t_lin, t_rf = fit_models(Xtr, ytr, args.seed)
    models = (scaler, lin, rf)

    # ---- evaluation --------------------------------------------------------------------
    price_te, price_next = test["gas"].to_numpy(), test["next_price"].to_numpy()
    spike_mask = np.abs(yte) >= np.quantile(np.abs(yte), 0.90)  # biggest 10% weekly moves
    rows, preds = [], {}
    for name, fit_s in [(BASELINE, 0.0), (LINEAR, t_lin), (FOREST, t_rf)]:
        t0 = time.perf_counter()
        chg = predict_change(name, models, Xte)
        pred_s = time.perf_counter() - t0
        preds[name] = price_te + chg
        rows.append({"model": name, **score(price_next, price_te + chg, yte, chg, spike_mask), "fit_seconds": fit_s, "predict_seconds": pred_s})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)

    # runtime vs. training-set size (the "scalability" part of the proposal)
    scaling = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(Xtr) * frac)
        _, _, _, a, b = fit_models(Xtr[:k], ytr[:k], args.seed)
        scaling.append({"train_rows": k, "linear_fit_seconds": a, "forest_fit_seconds": b})
    pd.DataFrame(scaling).to_csv(args.out_dir / "runtime_scaling.csv", index=False)

    # ---- explanations: local (every test week + the live forecast) and global ----------
    all_X = np.vstack([Xte, Xfc])
    all_price = np.concatenate([price_te, forecast_row["gas"].to_numpy()])
    all_dates = list(test["date"].dt.strftime("%Y-%m-%d")) + [forecast_row["date"].iloc[0].strftime("%Y-%m-%d")]
    actual_next = list(price_next) + [None]
    explained, group_mean_abs = {}, {}
    for name in (LINEAR, FOREST):
        base, contrib = explain(name, models, all_X)
        total = base + contrib.sum(axis=1)
        assert np.allclose(total, predict_change(name, models, all_X), atol=1e-6), f"{name}: contributions do not add up to the prediction"
        out = []
        for i in range(len(all_X)):
            fc = dict(zip(FEATURES, contrib[i]))
            fv = dict(zip(FEATURES, all_X[i]))
            chg = float(total[i])
            drivers = top_drivers(fc, fv)
            out.append(
                {
                    "date": all_dates[i],
                    "price_now": float(all_price[i]),
                    "predicted_next": float(all_price[i] + chg),
                    "actual_next": None if actual_next[i] is None else float(actual_next[i]),
                    "base_cents": round(float(base) * 100, 3),
                    "groups_cents": {g: round(c * 100, 3) for g, c in to_groups(fc).items()},
                    "features": feature_table(fc, fv),
                    "drivers": drivers,
                    "summary": sentence(float(all_price[i]), chg, drivers),
                }
            )
        explained[name] = out
        group_mean_abs[name] = {g: round(float(np.mean([abs(r["groups_cents"][g]) for r in out[:-1]])), 3) for g in FEATURE_GROUPS}

    perm = {}
    for name, model, X in [(LINEAR, lin, scaler.transform(Xte)), (FOREST, rf, Xte)]:
        r = permutation_importance(model, X, yte, scoring="neg_mean_absolute_error", n_repeats=20, random_state=args.seed)
        perm[name] = [
            {"feature": f, "label": FEATURE_LABELS[f], "mae_increase_cents": round(float(m) * 100, 3), "std_cents": round(float(s) * 100, 3)}
            for f, m, s in sorted(zip(FEATURES, r.importances_mean, r.importances_std), key=lambda t: -t[1])
        ]

    # ---- what-if: crude moves by $X/barrel this week, rebuild the features, re-forecast --
    scenarios = []
    for shock in (-10, -5, 0, 5, 10):
        shocked = df.copy()
        shocked.loc[shocked.index[-1], "crude"] += shock
        row = make_features(shocked).dropna(subset=FEATURES).iloc[[-1]]
        x = row[FEATURES].to_numpy()
        scenarios.append(
            {
                "crude_shock": shock,
                "crude_now": float(shocked["crude"].iloc[-1]),
                LINEAR: float(row["gas"].iloc[0] + predict_change(LINEAR, models, x)[0]),
                FOREST: float(row["gas"].iloc[0] + predict_change(FOREST, models, x)[0]),
            }
        )

    # ---- export ------------------------------------------------------------------------
    last = forecast_row.iloc[0]
    meta = {
        "source": source,
        "is_demo": is_demo,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_observed_week": last["date"].strftime("%Y-%m-%d"),
        "forecast_week": (last["date"] + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        "current_price": float(last["gas"]),
        "current_crude": float(last["crude"]),
        "train_weeks": len(train),
        "test_weeks": len(test),
        "train_range": [train["date"].iloc[0].strftime("%Y-%m-%d"), train["date"].iloc[-1].strftime("%Y-%m-%d")],
        "test_range": [test["date"].iloc[0].strftime("%Y-%m-%d"), test["date"].iloc[-1].strftime("%Y-%m-%d")],
    }
    payload = {
        "meta": meta,
        "history": [[d.strftime("%Y-%m-%d"), float(p)] for d, p in zip(df["date"], df["gas"])],
        "test_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in test["date"] + pd.Timedelta(days=7)],
            "actual": [float(x) for x in price_next],
            BASELINE: [float(x) for x in preds[BASELINE]],
            LINEAR: [float(x) for x in preds[LINEAR]],
            FOREST: [float(x) for x in preds[FOREST]],
        },
        "metrics": json.loads(metrics.to_json(orient="records")),
        "runtime_scaling": scaling,
        "explanations": explained,
        "group_mean_abs_cents": group_mean_abs,
        "permutation_importance": perm,
        "scenarios": scenarios,
        "feature_groups": FEATURE_GROUPS,
    }
    (args.out_dir / "dashboard_data.json").write_text(json.dumps(payload))
    latest = {"meta": meta, **{m: {k: v for k, v in explained[m][-1].items() if k != "features"} for m in (LINEAR, FOREST)}, "scenarios": scenarios}
    (args.out_dir / "latest_forecast.json").write_text(json.dumps(latest, indent=2))

    tpl = (Path(__file__).resolve().parent / "dashboard_template.html")
    if tpl.exists():
        (args.out_dir / "dashboard.html").write_text(tpl.read_text().replace("__DATA__", json.dumps(payload)))

    if not args.no_figures:
        from plots import make_figures  # noqa: E402

        make_figures(payload, args.out_dir / "figures")

    # ---- console summary ---------------------------------------------------------------
    pd.set_option("display.width", 160)
    print(f"\n{source}")
    print(f"Weeks: {len(train)} train ({meta['train_range'][0]} -> {meta['train_range'][1]}), {len(test)} test ({meta['test_range'][0]} -> {meta['test_range'][1]})")
    print("\nHeld-out test results (1-week-ahead price, $/gal):")
    print(metrics.round(4).to_string(index=False))
    print(f"\nForecast for week of {meta['forecast_week']} (this week ${meta['current_price']:.2f}):")
    for m in (LINEAR, FOREST):
        print(f"  {m}: {explained[m][-1]['summary']}")
    print("\nBeat the baseline on MAE and RMSE?", {m: bool(metrics.set_index('model').loc[m, 'MAE'] < metrics.set_index('model').loc[BASELINE, 'MAE'] and metrics.set_index('model').loc[m, 'RMSE'] < metrics.set_index('model').loc[BASELINE, 'RMSE']) for m in (LINEAR, FOREST)})
    print(f"\nWrote {args.out_dir}/")


if __name__ == "__main__":
    main()
