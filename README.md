# Gas Price Predictor (DS 305 prototype)

## Quick start

```bash
pip install pandas numpy scikit-learn
python src/gas_predictor.py
```

If the data is missing, the program asks you to paste a free EIA API key (get one at https://www.eia.gov/opendata/register.php),
downloads everything, and writes one file: `outputs/dashboard.html` (open it in your browser).
Press Enter at the prompt to skip the download and see demo data instead. The key is not saved.

The older notes below still apply; ignore the parts about CSV, JSON and PNG outputs, which are no longer written.


Forecasts next week's U.S. regular gasoline price from recent pump prices, crude oil prices and seasonality,
and **explains every forecast** in plain English.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib     # nothing else needed
python src/gas_predictor.py                          # real data if data/ has the CSVs, else synthetic demo
open outputs/dashboard.html                          # interactive dashboard (double-click works too)
```

## Get live data from EIA (recommended): one command, includes states

EIA's free API gives the newest weekly prices for the U.S., regions and the 9 states EIA surveys, plus the full daily WTI crude history. This replaces downloading CSVs by hand.

1. Get a free key (instant): https://www.eia.gov/opendata/register.php
2. Run (macOS/Linux shown; on Windows PowerShell use `$env:EIA_API_KEY="YOUR_KEY"`):
   ```bash
   export EIA_API_KEY=YOUR_KEY
   python src/gas_predictor.py --fetch-eia
   ```
   or pass `--eia-key YOUR_KEY`. The key is sent only to api.eia.gov; it is never saved or printed.
3. It writes `data/GASREGW.csv`, `data/DCOILWTICO.csv` and `data/regions/<Area>.csv`, then runs the model on everything. Your previous `GASREGW.csv` / `DCOILWTICO.csv` are kept once as `*.csv.bak`.

Choose areas with `--eia-areas`. The default is `"states,metros"`: the 9 states EIA surveys plus the cities it also publishes weekly (Boston, Chicago, Houston, Los Angeles, Miami, New York City, San Francisco, Seattle and so on, whatever EIA lists). PADD regions are left out unless you add `regions` (for example `"states,metros,regions"`). Other values: `all`, or names such as `"Texas, Chicago"`. Region files from an earlier fetch (the old `Padd_*.csv`) are moved to `data/regions/_old/` so they are not modeled again. If a name is not found, the console lists the areas EIA does offer. EIA does not publish weekly prices for the other 41 states, and this project does not invent them.

Runtime: each area takes about 7 seconds, so 20 areas take roughly 2-3 minutes. Add `--no-walkforward` to skip the rolling-retrain test and cut that by about half.

Run it again every Tuesday or so for the newest week (EIA publishes weekly, so "live" means weekly). To refresh later without re-downloading, run plain `python src/gas_predictor.py`.

Troubleshooting: "rejected the API key" means the key was copied wrong; "could not reach api.eia.gov" means no internet (the CSV route below still works offline).

Heads-up: the fetch code was tested against a mock of EIA's API, not the live service. If EIA's reply format differs, the error message says what it saw.

## What the dashboard adds beyond the forecast

| Section | What it answers | How it is computed |
|---|---|---|
| Likely range (hero, back-test band, scorecard) | "How sure is the forecast?" | Split-conformal ranges: calibrated on the last 25% of the training weeks, scaled by recent price volatility, built to contain the real price about 80% of the time. The scorecard reports how often they actually did on the held-out weeks. |
| Walk-forward test | "Does it hold up if you retrain as new weeks arrive?" | Expanding-window back-test: retrain on all weeks known so far (about every 4 weeks), forecast the next block, roll forward. Only weeks already observed are ever used (checked with a scrambled-future test). Forest uses 100 trees here to stay fast. |
| Rockets and feathers | "Do pump prices follow crude rises faster than crude falls?" | Distributed-lag regression of the weekly pump change on the last 6 weekly crude changes, rises and falls separate, Newey-West standard errors (plain numpy). Compares the cumulative response after a $1/bbl rise vs fall, both in total and in speed (week by week, with a Bonferroni-corrected bar for the speed claim). |
| Which areas react first | "How much and how fast does crude reach the pump, per area?" | Same regression, rises and falls combined: total cents per $1/bbl within 6 weeks, share arriving the same week. Shown when more than one area is loaded. |
| Flat-price crude | "What crude price would make the forecast flat?" | Searches shifts of this week's crude from about -$40 to +$40 for a zero predicted change. |
| Do the models agree? | "Is the explanation trustworthy this week?" | Compares each input group's push in the linear and forest models. |

Notes for the report: the p-values are approximate (about 5-9% false alarms in simulation at 5 years of data, against 5% nominal), so call results "evidence", not proof. Pass-through can exceed 100% of the crude-cost benchmark because pump margins and costs also move with crude. `outputs/extras_by_area.csv` has these numbers for every area. The methods were checked on simulated data with known answers (a planted rise/fall gap is recovered, a symmetric world rarely triggers the test).

## Switch from demo data to real data (do this before writing the report)

The demo data is synthetic. Its numbers are for showing how the prototype works, not for the report.

1. Download two CSVs from FRED (free, no account needed) into `data/`:
   - `GASREGW.csv`: https://fred.stlouisfed.org/series/GASREGW (EIA weekly U.S. regular gasoline price; same EIA source as the proposal)
   - `DCOILWTICO.csv`: https://fred.stlouisfed.org/series/DCOILWTICO (daily WTI crude)
   Use the **Download -> CSV** button on each page and keep those file names. FRED's default window for the crude series is only the last 5 years: on its page set the date range to the **maximum** before downloading, or the model will only have about 4 years of weeks to learn from.
2. Run `python src/gas_predictor.py`. The dashboard banner changes from "demo data" to "real data".

Options: `--demo` forces demo data, `--test-frac 0.25` changes the held-out share, `--no-figures` skips PNGs.

## Add states and regions (optional)

Weekly pump prices by state are **not** published for all 50 states. EIA's weekly survey covers **9 states** (California, Colorado, Florida, Massachusetts, Minnesota, New York, Ohio, Texas, Washington) plus about ten cities. Pennsylvania has no weekly series of its own. If you ever need a regional stand-in, `--eia-areas states,metros,regions` adds the PADD regions (Central Atlantic, PADD 1B, covers PA, NY, NJ, DE, MD and DC); label it a regional proxy in the report, not Pennsylvania.


How to add one:
1. Get its weekly regular-gasoline history: on https://www.eia.gov/petroleum/gasdiesel/ use the "full history" XLS link for the area, and save the two columns (date, price) as a CSV with a header row. (FRED mirrors some of these too, e.g. `GASREGECW` for the East Coast and `GASREGWCW` for the West Coast, in the same two-column format as `GASREGW.csv`.)
2. Put the file in `data/regions/`. The file name becomes the label: `data/regions/Central_Atlantic.csv` shows as "Central Atlantic", `California.csv` as "California".
   Alternatively use one wide `data/regions.csv` with a date column and one column per area.
3. Run `python src/gas_predictor.py` again. The dashboard gets an **Area** selector and a "States and cities at a glance" table; clicking a row (or using the selector) re-scopes every chart, explanation and scorecard to that area. `outputs/metrics_by_region.csv` holds the scorecards side by side.

Each area is modeled separately with the same method as the U.S. (its own pump-price moves, national WTI crude, season). An area needs at least 120 weeks that overlap the crude file, otherwise it is skipped and the console says so. With no regional files, nothing changes.

## What is in the box

| File | Job |
|---|---|
| `src/analysis_extras.py` | Rockets-and-feathers test, crude pass-through speed, flat-crude search (numpy only) |
| `src/eia_fetch.py` | Optional `--fetch-eia`: pulls U.S./state/region gas and WTI crude from the EIA API into `data/` (standard library only) |
| `src/data_sources.py` | Loads FRED CSVs, aligns daily crude to weekly dates (no look-ahead), makes the demo data |
| `src/features.py` | The lag-feature pipeline: crude moves (1-4 weeks, in $/gal), pump momentum, markup gap vs 52-week norm, season and holidays |
| `src/explain.py` | The explainable-AI layer (below) |
| `src/gas_predictor.py` | Chronological split, baseline vs linear regression vs random forest, MAE/RMSE/MAPE, runtime, spike analysis, what-if scenarios, export |
| `src/plots.py`, `src/dashboard_template.html` | Report figures and the interactive dashboard |
| `outputs/` | `metrics.csv`, `metrics_by_region.csv` (if regions), `extras_by_area.csv`, `runtime_scaling.csv`, `latest_forecast.json`, `dashboard.html`, `figures/*.png` (U.S.) and `figures/<area>/` |

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

- **Wholesale gasoline is not an input yet.** It moves before pump prices, so it would likely improve accuracy most. EIA publishes daily spot prices, but the exact series ID needs checking against the live API, so it was left out rather than guessed.
- **Only 9 states.** EIA's weekly survey covers 9 states, a few cities and the PADD regions. Other states would have to be estimates, which should never be mixed into the forecasts.
- `statsmodels` and `shap` were not available where this was built, so the optional **ARIMA** check is not included and the forest explanation uses the tree-path method. If you install them, `shap.TreeExplainer(rf)` is a drop-in upgrade for the forest explanation, and ARIMA can be added as a fourth model in `gas_predictor.py`.
- Brent crude (also named in the proposal) is not used yet; it is highly correlated with WTI, so expect little gain. Add a `DCOILBRENTEU.csv` loader in `data_sources.py` if you want it.
- Shocks (hurricanes, refinery outages, policy) are not inputs, so the models lag big spikes. The scorecard's "biggest 10% moves" column quantifies this for the error-analysis slide.
- Everything is plain Python files so the two of you can split them. To turn it into a notebook, paste `main()`'s steps into cells.
