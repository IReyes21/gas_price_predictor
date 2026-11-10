# Gas Price Predictor (DS 305 prototype)

Forecasts next week's U.S. regular gasoline price from recent pump prices, crude oil prices and seasonality,
and **explains every forecast** in plain English.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib     # nothing else needed
python src/gas_predictor.py                          # real data if data/ has the CSVs, else synthetic demo
open outputs/dashboard.html                          # interactive dashboard (double-click works too)
```

## Switch from demo data to real data (do this before writing the report)

The demo data is synthetic. Its numbers are for showing how the prototype works, not for the report.

1. Download two CSVs from FRED (free, no account needed) into `data/`:
   - `GASREGW.csv`: https://fred.stlouisfed.org/series/GASREGW (EIA weekly U.S. regular gasoline price; same EIA source as the proposal)
   - `DCOILWTICO.csv`: https://fred.stlouisfed.org/series/DCOILWTICO (daily WTI crude)
   Use the **Download -> CSV** button on each page and keep those file names.
2. Run `python src/gas_predictor.py`. The dashboard banner changes from "demo data" to "real data".

Options: `--demo` forces demo data, `--test-frac 0.25` changes the held-out share, `--no-figures` skips PNGs.

## What is in the box

| File | Job |
|---|---|
| `src/data_sources.py` | Loads FRED CSVs, aligns daily crude to weekly dates (no look-ahead), makes the demo data |
| `src/features.py` | The lag-feature pipeline: crude moves (1-4 weeks, in $/gal), pump momentum, markup gap vs 52-week norm, season and holidays |
| `src/explain.py` | The explainable-AI layer (below) |
| `src/gas_predictor.py` | Chronological split, baseline vs linear regression vs random forest, MAE/RMSE/MAPE, runtime, spike analysis, what-if scenarios, export |
| `src/plots.py`, `src/dashboard_template.html` | Report figures and the interactive dashboard |
| `outputs/` | `metrics.csv`, `runtime_scaling.csv`, `latest_forecast.json`, `dashboard.html`, `figures/*.png` |

## How the forecast and the explanation work

The models predict the **change** in the pump price one week ahead (not the price itself), so
`forecast = this week's price + predicted change`. That makes the explanation natural: start at this week's price,
and each group of inputs pushes it up or down by some number of cents.

- **Linear regression:** each input's push is `coefficient x (input - training average)`. This is exactly the SHAP value for a linear model.
- **Random forest:** every split in every tree changes the node's average prediction; that change is credited to the input the split used, then averaged over trees (the "tree-path" method, as in `treeinterpreter`).
- Both satisfy `base + sum of pushes = prediction`. The script asserts this on every run.
- **Global view:** average size of each group's push over all test weeks, plus permutation importance per input.
- **What-if:** re-runs the upcoming forecast with this week's crude shifted by -$10 to +$10 per barrel.

## Success criterion from the proposal

Beat the last-week baseline on MAE and RMSE on a chronological hold-out. The script prints whether each model does, and the dashboard scorecard shows it.

## Known gaps / next steps

- `statsmodels` and `shap` were not available where this was built, so the optional **ARIMA** check is not included and the forest explanation uses the tree-path method. If you install them, `shap.TreeExplainer(rf)` is a drop-in upgrade for the forest explanation, and ARIMA can be added as a fourth model in `gas_predictor.py`.
- Brent crude (also named in the proposal) is not used yet; it is highly correlated with WTI, so expect little gain. Add a `DCOILBRENTEU.csv` loader in `data_sources.py` if you want it.
- Shocks (hurricanes, refinery outages, policy) are not inputs, so the models lag big spikes. The scorecard's "biggest 10% moves" column quantifies this for the error-analysis slide.
- Everything is plain Python files so the two of you can split them. To turn it into a notebook, paste `main()`'s steps into cells.
