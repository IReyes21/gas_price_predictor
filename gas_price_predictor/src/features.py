"""Feature engineering: the lag-feature pipeline that links crude oil moves to pump prices.

Every feature is computed from information known on the Monday we forecast from
(time t). The target is the change in the pump price one week later:
    target_chg = gas[t+1] - gas[t]
so  forecast = this week's price + predicted change.
Predicting the change (not the level) keeps the explanation simple: it starts at
"this week's price" and each feature group pushes it up or down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARREL_TO_GALLON = 42.0

# Features grouped the way a commuter would think about them (used in the "why" chart).
FEATURE_GROUPS = {
    "Recent pump-price momentum": ["gas_chg_1w", "gas_chg_2w", "gas_chg_3w"],
    "Crude oil moves": ["crude_chg_1w", "crude_chg_2w", "crude_chg_3w", "crude_chg_4w"],
    "Pump-vs-crude markup gap": ["margin_gap_vs_52w"],
    "Season & holidays": ["summer_season", "holiday_week", "season_sin", "season_cos"],
}
FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]
FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

FEATURE_LABELS = {
    "gas_chg_1w": "Pump price change, this week",
    "gas_chg_2w": "Pump price change, last week",
    "gas_chg_3w": "Pump price change, 2 weeks ago",
    "crude_chg_1w": "Crude change, this week",
    "crude_chg_2w": "Crude change, last week",
    "crude_chg_3w": "Crude change, 2 weeks ago",
    "crude_chg_4w": "Crude change, 3 weeks ago",
    "margin_gap_vs_52w": "Pump price minus crude vs 52-wk norm",
    "summer_season": "Summer driving season (May-Sep)",
    "holiday_week": "Holiday in forecast week",
    "season_sin": "Seasonal cycle (sin)",
    "season_cos": "Seasonal cycle (cos)",
}


def _holidays(year: int) -> list[pd.Timestamp]:
    memorial = pd.date_range(f"{year}-05-01", f"{year}-05-31", freq="W-MON")[-1]
    labor = pd.date_range(f"{year}-09-01", f"{year}-09-30", freq="W-MON")[0]
    thanksgiving = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="W-THU")[3]
    return [
        pd.Timestamp(f"{year}-01-01"),
        memorial,
        pd.Timestamp(f"{year}-07-04"),
        labor,
        thanksgiving,
        pd.Timestamp(f"{year}-12-25"),
    ]


def has_holiday(week_start: pd.Timestamp) -> bool:
    """True if a major U.S. holiday falls in the 7 days after `week_start` (the forecast week)."""
    week_start = pd.Timestamp(week_start)
    end = week_start + pd.Timedelta(days=7)
    for yr in {week_start.year, end.year}:
        for h in _holidays(yr):
            if week_start < h <= end:
                return True
    return False


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df has columns date (weekly Monday), gas ($/gal), crude (weekly avg WTI, $/bbl)."""
    d = df.sort_values("date").reset_index(drop=True)
    g = d["gas"]
    c = d["crude"] / BARREL_TO_GALLON  # crude in $ per gallon, same units as pump price

    f = pd.DataFrame({"date": d["date"], "gas": g, "crude": d["crude"]})
    f["gas_chg_1w"] = g - g.shift(1)
    f["gas_chg_2w"] = g.shift(1) - g.shift(2)
    f["gas_chg_3w"] = g.shift(2) - g.shift(3)
    for k in (1, 2, 3, 4):
        f[f"crude_chg_{k}w"] = c.shift(k - 1) - c.shift(k)

    spread = g - c  # what the pump charges on top of the crude cost (taxes, refining, retail margin)
    f["margin_gap_vs_52w"] = spread - spread.rolling(52, min_periods=52).mean()

    nxt = f["date"] + pd.Timedelta(days=7)  # calendar features describe the week being forecast
    f["summer_season"] = nxt.dt.month.isin([5, 6, 7, 8, 9]).astype(int)
    f["holiday_week"] = [int(has_holiday(x)) for x in f["date"]]
    woy = nxt.dt.isocalendar().week.astype(int)
    f["season_sin"] = np.sin(2 * np.pi * woy / 52)
    f["season_cos"] = np.cos(2 * np.pi * woy / 52)

    f["next_price"] = g.shift(-1)
    f["target_chg"] = f["next_price"] - g
    return f
