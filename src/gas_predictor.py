#!/usr/bin/env python3
"""Gas Price Predictor - DS 305 prototype.

Forecasts next week's U.S. regular gasoline price from crude oil prices, recent pump
prices and seasonality, and explains every forecast. Optionally does the same for
individual states / regions (see data/regions/ and the README).

    python src/gas_predictor.py              # uses data/*.csv if present, else the synthetic demo
    python src/gas_predictor.py --demo       # force the synthetic demo
    python src/gas_predictor.py --no-figures # skip PNG charts

Outputs land in outputs/ (metrics.csv, latest_forecast.json, dashboard.html, figures/).
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
from data_sources import load_regions, load_weekly  # noqa: E402
from analysis_extras import crude_asymmetry, flat_crude_shift, passthrough_speed  # noqa: E402
from eia_fetch import DEFAULT_AREAS, EiaError, fetch_all  # noqa: E402
from explain import feature_table, forest_explain, linear_explain, sentence, to_groups, top_drivers  # noqa: E402
from features import FEATURE_GROUPS, FEATURE_LABELS, FEATURES, make_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE, LINEAR, FOREST = "Last-week baseline", "Linear regression", "Random forest"
US = "United States"


class FastForest(RandomForestRegressor):
    """Same forest, same numbers; predict() averages the trees directly.

    scikit-learn's own predict() starts a thread pool on every call. The importance and what-if steps call
    predict hundreds of times on tiny inputs, so that overhead (not the math) dominated the run time.
    """

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([e.predict(X, check_input=False) for e in self.estimators_], axis=0)


def fit_models(Xtr, ytr, seed):
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(Xtr)
    lin = LinearRegression().fit(scaler.transform(Xtr), ytr)
    t_lin = time.perf_counter() - t0
    t0 = time.perf_counter()
    rf = FastForest(n_estimators=300, min_samples_leaf=8, max_features=0.5, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
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


def data_warnings(df, is_demo):
    """Plain-English data-quality flags, shown in the console and on the dashboard."""
    out = []
    if is_demo:
        return out
    last_gas, first = df["date"].iloc[-1], df["date"].iloc[0]
    crude_last = df.attrs.get("crude_last_obs")
    if crude_last is not None and (last_gas - crude_last).days > 2:
        n = df.attrs.get("crude_days_in_last_week", 0)
        out.append(
            f"Crude data ends {crude_last:%Y-%m-%d}, but gas prices run through {last_gas:%Y-%m-%d}. The newest week's crude value averages only "
            f"{n} trading day(s), so the upcoming forecast leans on a stale crude reading. Re-run with --fetch-eia (or re-download DCOILWTICO.csv) once newer days are published."
        )
    if (last_gas - first).days < 8 * 365:
        out.append(
            f"Only {len(df)} weeks of overlapping data (starting {first:%Y-%m-%d}). the crude file covers only a short window: run with --fetch-eia, "
            "or on FRED's DCOILWTICO page set the date range to the maximum and re-download, for far more training weeks."
        )
    return out


RANGE_LEVEL = 0.8   # the forecast range is built to contain the real price about 80% of the time


def _recent_vol(X):
    """Typical size of the last three weekly pump-price moves ($/gal), known at forecast time. Floor of 1 cent."""
    idx = [FEATURES.index(f) for f in ("gas_chg_1w", "gas_chg_2w", "gas_chg_3w")]
    return np.maximum(np.abs(X[:, idx]).mean(axis=1), 0.01)


def conformal_scale(Xtr, ytr, seed, level=RANGE_LEVEL):
    """Volatility-scaled split-conformal range: half-width = q x (recent volatility).

    Fit on the first 75% of the TRAINING weeks, measure each model's error on the last 25% (weeks it never saw) as a
    multiple of the recent volatility, and take the multiple that 80% of those weeks stayed under. Scaling by
    volatility makes the range widen when prices get jumpy (a fixed width calibrated on calm weeks is far too
    narrow in a shock). No held-out test week is touched.
    """
    n = len(Xtr)
    m = int(round(n * 0.75))
    if n - m < 20:
        return None
    models = fit_models(Xtr[:m], ytr[:m], seed)[:3]
    vol = _recent_vol(Xtr[m:])
    out = {}
    for name in (LINEAR, FOREST):
        res = np.sort(np.abs(ytr[m:] - predict_change(name, models, Xtr[m:])) / vol)
        out[name] = float(res[min(len(res), math.ceil((len(res) + 1) * level)) - 1])
    return {"q": out, "calibration_weeks": int(n - m)}


def walk_forward(labelled, n_test, seed, window=8):
    """Honest back-test: retrain on everything known so far, forecast the next block of weeks, then roll forward.

    The single train/test split trains once and hopes the model stays good. A real forecaster would retrain as new
    weeks arrive, so this does that (about 16 refits across the test period, forest uses 100 trees to stay fast).
    """
    X = labelled[FEATURES].to_numpy()
    y = labelled["target_chg"].to_numpy()
    price, nxt = labelled["gas"].to_numpy(), labelled["next_price"].to_numpy()
    s0 = len(labelled) - n_test
    step = max(4, math.ceil(n_test / 16))
    chg = {LINEAR: [], FOREST: []}
    for j in range(s0, len(labelled), step):
        hi = min(j + step, len(labelled))
        assert X[:j].shape[0] == j and j <= len(X)   # origin week j: only rows 0..j-1 (targets already observed) are used
        sc = StandardScaler().fit(X[:j])
        lin = LinearRegression().fit(sc.transform(X[:j]), y[:j])
        rf = FastForest(n_estimators=100, min_samples_leaf=8, max_features=0.5, random_state=seed, n_jobs=-1).fit(X[:j], y[:j])
        chg[LINEAR].append(lin.predict(sc.transform(X[j:hi])))
        chg[FOREST].append(rf.predict(X[j:hi]))
    chg = {k: np.concatenate(v) for k, v in chg.items()}
    chg[BASELINE] = np.zeros(n_test)
    yt, pt, actual = y[s0:], price[s0:], nxt[s0:]
    spike = np.abs(yt) >= np.quantile(np.abs(yt), 0.90)
    metrics, rolling = [], {}
    for name in (BASELINE, LINEAR, FOREST):
        sc_ = score(actual, pt + chg[name], yt, chg[name], spike)
        metrics.append({"model": name, "MAE": sc_["MAE"], "RMSE": sc_["RMSE"], "MAPE_pct": sc_["MAPE_pct"], "direction_hit_rate": sc_["direction_hit_rate"]})
        ae = np.abs(pt + chg[name] - actual)
        rolling[name] = [float(ae[i - window + 1 : i + 1].mean()) for i in range(window - 1, n_test)]
    dates = [d.strftime("%Y-%m-%d") for d in labelled["date"].iloc[s0:] + pd.Timedelta(days=7)]
    return {"step_weeks": step, "refits": len(range(s0, len(labelled), step)), "window": window, "metrics": metrics,
            "dates": dates[window - 1 :], "rolling_mae": rolling}


def flat_crude(df, models):
    """What change in this week's crude price would make each model forecast an unchanged pump price?"""
    tail = df.tail(80).reset_index(drop=True)
    now = float(tail["crude"].iloc[-1])
    lo = -int(min(40, max(now - 10, 0)))
    shifts = [float(s) for s in np.arange(lo, 41)]
    rows = []
    for s in shifts:
        t = tail.copy()
        t.loc[t.index[-1], "crude"] += s
        rows.append(make_features(t).iloc[-1][FEATURES].to_numpy(dtype=float))
    X = np.vstack(rows)
    vals = {name: dict(zip(shifts, predict_change(name, models, X))) for name in (LINEAR, FOREST)}
    return {name: flat_crude_shift(lambda s, n=name: vals[n][float(s)], now, lo, 40) for name in (LINEAR, FOREST)}


def analyze(df, args, label, source, is_demo, warnings):
    """Full pipeline for ONE price series (the U.S., or one state / region)."""
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

    # runtime vs. training-set size (the "scalability" part of the proposal)
    scaling = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(Xtr) * frac)
        _, _, _, a, b = fit_models(Xtr[:k], ytr[:k], args.seed)
        scaling.append({"train_rows": k, "linear_fit_seconds": a, "forest_fit_seconds": b})

    # ---- explanations: local (every test week + the live forecast) and global ----------
    all_X = np.vstack([Xte, Xfc])
    all_price = np.concatenate([price_te, forecast_row["gas"].to_numpy()])
    all_dates = list(test["date"].dt.strftime("%Y-%m-%d")) + [forecast_row["date"].iloc[0].strftime("%Y-%m-%d")]
    actual_next = list(price_next) + [None]
    explained, group_mean_abs = {}, {}
    for name in (LINEAR, FOREST):
        base, contrib = explain(name, models, all_X)
        total = base + contrib.sum(axis=1)
        assert np.allclose(total, predict_change(name, models, all_X), atol=1e-6), f"{label} / {name}: contributions do not add up to the prediction"
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

    # ---- extras: forecast ranges, walk-forward back-test, crude pass-through, flat-price crude ------------
    cal = conformal_scale(Xtr, ytr, args.seed)
    intervals = None
    if cal:
        vol_te, vol_fc = _recent_vol(Xte), _recent_vol(Xfc)
        half_te = {n: cal["q"][n] * vol_te for n in (LINEAR, FOREST)}
        intervals = {
            "level": RANGE_LEVEL, "calibration_weeks": cal["calibration_weeks"],
            "half_test": {n: [float(x) for x in half_te[n]] for n in half_te},
            "half_forecast": {n: float(cal["q"][n] * vol_fc[0]) for n in cal["q"]},
            "typical_half": {n: float(np.median(half_te[n])) for n in half_te},
            "coverage": {n: float(np.mean(np.abs(preds[n] - price_next) <= half_te[n])) for n in half_te},
        }
    wf = None if getattr(args, "no_walkforward", False) else walk_forward(labelled, n_test, args.seed)
    asym = crude_asymmetry(df["gas"].to_numpy(), df["crude"].to_numpy())
    speed = passthrough_speed(df["gas"].to_numpy(), df["crude"].to_numpy())
    flat = flat_crude(df, models)

    # ---- package -----------------------------------------------------------------------
    last = forecast_row.iloc[0]
    meta = {
        "label": label,
        "source": source,
        "is_demo": is_demo,
        "warnings": warnings,
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
        "intervals": intervals,
        "walkforward": wf,
        "asymmetry": asym,
        "passthrough": speed,
        "flat_crude": flat,
    }
    return {"payload": payload, "metrics": metrics, "scaling": scaling, "explained": explained, "meta": meta, "n_train": len(train), "n_test": len(test)}


def beats_baseline(metrics, model):
    m = metrics.set_index("model")
    return bool(m.loc[model, "MAE"] < m.loc[BASELINE, "MAE"] and m.loc[model, "RMSE"] < m.loc[BASELINE, "RMSE"])


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--demo", action="store_true", help="force synthetic demo data")
    ap.add_argument("--test-frac", type=float, default=0.2, help="share of weeks held out, chronologically")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-walkforward", action="store_true", help="skip the rolling-retrain back-test (saves time on many areas)")
    ap.add_argument("--fetch-eia", action="store_true",
                    help="first download fresh data from the EIA API (U.S. gas, WTI crude, states, cities) into the data folder")
    ap.add_argument("--eia-key", default=None, help="EIA API key (or set the EIA_API_KEY environment variable)")
    ap.add_argument("--eia-areas", default=DEFAULT_AREAS,
                    help='areas to download: "states", "regions", "metros", "all", or names, comma separated '
                         '(default: "%s"; EIA publishes only 9 states weekly)' % DEFAULT_AREAS)
    ap.add_argument("--open", action="store_true", help="open the finished dashboard in your browser")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252/cp437; never crash on a symbol
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- optional: refresh the data folder from the EIA API ---------------------------
    if args.fetch_eia:
        import os
        try:
            fetch_all(args.eia_key or os.environ.get("EIA_API_KEY", ""), args.data_dir, args.eia_areas)
        except EiaError as e:
            sys.exit("EIA download failed: %s" % e)
        print()

    # ---- data + one analysis per area --------------------------------------------------
    df, source, is_demo = load_weekly(args.data_dir, args.demo)
    regions, skipped = ({}, []) if is_demo else load_regions(args.data_dir)
    results = {US: analyze(df, args, US, source, is_demo, data_warnings(df, is_demo))}
    print(f"Modeling {len(regions) + 1} areas (about 10-20 seconds each) ...", flush=True)
    for i, (label, rdf) in enumerate(regions.items(), 1):
        print(f"  [{i}/{len(regions)}] {label}", flush=True)
        results[label] = analyze(rdf, args, label, "Real data: " + label + " weekly retail gasoline (EIA) + WTI crude (DCOILWTICO)", False, data_warnings(rdf, False))

    us = results[US]
    meta, metrics, explained = us["meta"], us["metrics"], us["explained"]

    # ---- export ------------------------------------------------------------------------
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    pd.DataFrame(us["scaling"]).to_csv(args.out_dir / "runtime_scaling.csv", index=False)
    if regions:
        pd.concat([r["metrics"].assign(region=label) for label, r in results.items()])[["region"] + list(metrics.columns)].to_csv(args.out_dir / "metrics_by_region.csv", index=False)

    def extras_row(label, r):
        p = r["payload"]
        a, sp, iv, wf = p["asymmetry"], p["passthrough"], p["intervals"], p["walkforward"]
        row = {"area": label, "weeks_used": r["n_train"] + r["n_test"]}
        if a:
            row.update({"rise_cents_per_1usd_bbl_6wk": a["rise"][-1], "fall_cents_per_1usd_bbl_6wk": a["fall"][-1], "rise_minus_fall_6wk": a["diff"][-1], "p_total_gap": a["p_value"],
                        "speed_gap_same_week": a["diff"][0], "p_speed_gap_same_week": a["diff_p"][0], "speed_signal": a["speed_dir"] or "none"})
        if sp:
            row.update({"passthrough_cents_per_1usd_bbl": sp["total"], "passthrough_pct_of_crude_cost": sp["total_pct_of_full"], "same_week_share": sp["same_week_share"], "weeks_to_half": sp["weeks_to_half"]})
        if iv:
            row.update({"range_coverage_linear": iv["coverage"][LINEAR], "range_coverage_forest": iv["coverage"][FOREST], "range_target": iv["level"]})
        if wf:
            m = {x["model"]: x["MAE"] for x in wf["metrics"]}
            row.update({"walkforward_MAE_baseline": m[BASELINE], "walkforward_MAE_linear": m[LINEAR], "walkforward_MAE_forest": m[FOREST]})
        return row

    pd.DataFrame([extras_row(label, r) for label, r in results.items()]).to_csv(args.out_dir / "extras_by_area.csv", index=False)

    def latest(res):
        return {"meta": res["meta"], **{m: {k: v for k, v in res["explained"][m][-1].items() if k != "features"} for m in (LINEAR, FOREST)}, "scenarios": res["payload"]["scenarios"]}

    latest_json = latest(us)
    if regions:
        latest_json["regions"] = {label: latest(r) for label, r in results.items() if label != US}
    (args.out_dir / "latest_forecast.json").write_text(json.dumps(latest_json, indent=2), encoding="utf-8")

    bundle = {"default": US, "order": list(results), "regions": {label: r["payload"] for label, r in results.items()}}
    (args.out_dir / "dashboard_data.json").write_text(json.dumps(bundle), encoding="utf-8")
    tpl = Path(__file__).resolve().parent / "dashboard_template.html"
    if not tpl.exists():
        sys.exit(f"Dashboard template not found: {tpl}\nCopy dashboard_template.html into the src folder next to gas_predictor.py and run again.")
    dash = args.out_dir / "dashboard.html"
    dash.write_text(tpl.read_text(encoding="utf-8").replace("__DATA__", json.dumps(bundle)), encoding="utf-8")
    print(f"Dashboard written: {dash.resolve()}  ({dash.stat().st_size / 1e6:.1f} MB, {len(results)} areas)", flush=True)

    if not args.no_figures:
        from plots import make_figures  # noqa: E402

        make_figures(us["payload"], args.out_dir / "figures")
        for label, r in results.items():
            if label != US:
                make_figures(r["payload"], args.out_dir / "figures" / slug(label))

    # ---- console summary ---------------------------------------------------------------
    pd.set_option("display.width", 160)
    print(f"\n{source}")
    for w in us["payload"]["meta"]["warnings"]:
        print("WARNING:", w)
    print(f"Weeks: {us['n_train']} train ({meta['train_range'][0]} -> {meta['train_range'][1]}), {us['n_test']} test ({meta['test_range'][0]} -> {meta['test_range'][1]})")
    print("\nHeld-out test results (1-week-ahead price, $/gal):")
    print(metrics.round(4).to_string(index=False))
    print(f"\nForecast for week of {meta['forecast_week']} (this week ${meta['current_price']:.2f}):")
    for m in (LINEAR, FOREST):
        print(f"  {m}: {explained[m][-1]['summary']}")
    print("\nBeat the baseline on MAE and RMSE?", {m: beats_baseline(metrics, m) for m in (LINEAR, FOREST)})
    p_us = us["payload"]
    iv, wf, a, sp = p_us["intervals"], p_us["walkforward"], p_us["asymmetry"], p_us["passthrough"]
    if iv:
        hw = iv["half_forecast"]
        print(f"\nLikely range for next week (about {iv['level']:.0%}): linear ${explained[LINEAR][-1]['predicted_next'] - hw[LINEAR]:.2f} to ${explained[LINEAR][-1]['predicted_next'] + hw[LINEAR]:.2f}; "
              f"forest ${explained[FOREST][-1]['predicted_next'] - hw[FOREST]:.2f} to ${explained[FOREST][-1]['predicted_next'] + hw[FOREST]:.2f}. "
              f"On the test weeks the ranges held {iv['coverage'][LINEAR]:.0%} (linear) and {iv['coverage'][FOREST]:.0%} (forest) of the time.")
    if wf:
        m = {x["model"]: x["MAE"] for x in wf["metrics"]}
        print(f"Walk-forward (retrained about every {wf['step_weeks']} weeks) MAE: baseline {m[BASELINE]*100:.2f}c, linear {m[LINEAR]*100:.2f}c, forest {m[FOREST]*100:.2f}c")
    if a:
        speed = {"rise_faster": "rises pass through faster than falls", "fall_faster": "falls pass through faster than rises"}.get(a["speed_dir"], "no clear speed difference")
        print(f"Rockets and feathers: after 6 weeks a $1/bbl crude rise = {a['rise'][-1]:.2f}c, a fall = {a['fall'][-1]:.2f}c (p={a['p_value']:.2f}); speed: {speed}"
              + (f" (same week: {a['rise'][0]:.2f}c vs {a['fall'][0]:.2f}c, p={a['diff_p'][0]:.3f})" if a["speed_dir"] else ""))
    if sp and sp["reliable"]:
        print(f"Pass-through: {sp['total']:.2f}c per $1/bbl within 6 weeks ({sp['total_pct_of_full']:.0f}% of the crude-cost benchmark); {sp['same_week_share']:.0%} arrives the same week")
    if regions or skipped:
        print("\nState / regional forecasts:")
        print(f"  {'area':<28}{'now':>7}{'linear':>8}{'forest':>8}   beats baseline (lin/forest)   weeks (train/test)")
        for label, r in results.items():
            if label == US:
                continue
            e = r["explained"]
            print(f"  {label:<28}{r['meta']['current_price']:>7.2f}{e[LINEAR][-1]['predicted_next']:>8.2f}{e[FOREST][-1]['predicted_next']:>8.2f}   "
                  f"{beats_baseline(r['metrics'], LINEAR)!s:>5}/{beats_baseline(r['metrics'], FOREST)!s:<5}   {r['n_train']}/{r['n_test']}")
        for s in skipped:
            print(f"  skipped '{s}': fewer than 120 weeks overlap with the crude file")
        print("  (per-area range coverage, walk-forward errors and pass-through are in outputs/extras_by_area.csv)")
    print(f"\nWrote {args.out_dir}/")
    if args.open:
        import webbrowser
        webbrowser.open(dash.resolve().as_uri())


if __name__ == "__main__":
    main()
