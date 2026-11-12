"""Load weekly gas + crude data.

Real data: drop two FRED CSV downloads into the data/ folder
    GASREGW.csv      weekly U.S. regular gasoline price, $/gal  (EIA series, hosted on FRED)
    DCOILWTICO.csv   daily WTI crude oil price, $/barrel
and the pipeline uses them automatically.

No files? A synthetic demo dataset is generated so the prototype still runs.
Demo numbers are NOT real prices and must not be used in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GAS_FILE = "GASREGW.csv"
WTI_FILE = "DCOILWTICO.csv"
CRUDE_FLOOR = 5.0  # WTI briefly went negative on 2020-04-20; floor daily prices so one day cannot wreck a weekly average


def _read_fred(path: Path) -> pd.DataFrame:
    """FRED CSV: first column = date, second = value ('.' or blank when missing)."""
    raw = pd.read_csv(path, na_values=[".", ""])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.iloc[:, 0]),
            "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
        }
    )
    return out.dropna().sort_values("date").reset_index(drop=True)


def _weekly_crude(daily: pd.DataFrame, week_dates) -> np.ndarray:
    """Average daily crude over the 7 days ending on each gas-price date (inclusive).

    Gas prices are surveyed on Mondays, so each weekly crude value only uses
    information that was already known on that Monday (no look-ahead).
    """
    s = daily.set_index("date")["value"].clip(lower=CRUDE_FLOOR)
    vals = []
    for d in week_dates:
        window = s.loc[d - pd.Timedelta(days=6) : d]
        vals.append(window.mean() if len(window) else np.nan)
    return np.array(vals)


def load_real(data_dir: Path, start: str = "2010-01-01") -> pd.DataFrame:
    gas = _read_fred(data_dir / GAS_FILE)
    wti = _read_fred(data_dir / WTI_FILE)
    gas = gas[gas["date"] >= start].reset_index(drop=True)
    df = pd.DataFrame({"date": gas["date"], "gas": gas["value"]})
    df["crude"] = _weekly_crude(wti, df["date"])
    df = df.dropna().reset_index(drop=True)
    last_gas = df["date"].iloc[-1]
    df.attrs["crude_last_obs"] = wti["date"].iloc[-1]
    df.attrs["crude_days_in_last_week"] = int(((wti["date"] > last_gas - pd.Timedelta(days=7)) & (wti["date"] <= last_gas)).sum())
    return df


def _bump(n: int, centre: int, up: int, down: int, amp: float) -> np.ndarray:
    """A smooth shock: ramps to `amp` over `up` days, then decays over `down` days."""
    x = np.arange(n)
    out = np.zeros(n)
    a = (x >= centre - up) & (x < centre)
    out[a] = amp * (x[a] - (centre - up)) / up
    b = (x >= centre) & (x < centre + down)
    out[b] = amp * (1 - (x[b] - centre) / down)
    return out


def make_demo(seed: int = 7, start: str = "2015-01-05", end: str = "2026-11-02") -> pd.DataFrame:
    """Synthetic weekly data with realistic structure: gas follows lagged crude,
    rises faster than it falls ("rockets and feathers"), has a summer bump,
    holiday bumps and two big shocks (so error analysis has spikes to look at)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(pd.Timestamp(start) - pd.Timedelta(days=60), end)
    n = len(days)

    logp = np.empty(n)
    logp[0] = np.log(60.0)
    mu = np.log(62.0)
    for i in range(1, n):
        logp[i] = logp[i - 1] + 0.004 * (mu - logp[i - 1]) + rng.normal(0, 0.016)
    for when, up, down, amp in [("2020-04-01", 25, 150, -0.55), ("2022-03-10", 30, 140, 0.38), ("2025-06-15", 15, 60, 0.18)]:
        logp += _bump(n, int(days.searchsorted(pd.Timestamp(when))), up, down, amp)
    daily = pd.DataFrame({"date": days, "value": np.exp(logp)})

    weeks = pd.date_range(start, end, freq="W-MON")
    crude = _weekly_crude(daily, weeks)

    from features import has_holiday  # local import avoids a circular dependency at module load

    gas = np.empty(len(weeks))
    gas[0] = crude[0] / 42 + 1.2
    for t in range(1, len(weeks)):
        woy = weeks[t].isocalendar().week
        summer = 1.0 if weeks[t].month in (5, 6, 7, 8, 9) else 0.0
        target = crude[t - 1] / 42 + 0.95 + 0.08 * np.sin(2 * np.pi * (woy - 10) / 52) + 0.07 * summer
        target += 0.03 * has_holiday(weeks[t - 1])
        a = 0.40 if target > gas[t - 1] else 0.18  # up fast, down slow
        gas[t] = gas[t - 1] + a * (target - gas[t - 1]) + rng.normal(0, 0.014)
    return pd.DataFrame({"date": weeks, "gas": gas.round(3), "crude": crude.round(2)})


def load_weekly(data_dir: Path, force_demo: bool = False, start: str = "2010-01-01"):
    """Returns (dataframe[date, gas, crude], source_label, is_demo)."""
    have_real = (data_dir / GAS_FILE).exists() and (data_dir / WTI_FILE).exists()
    if have_real and not force_demo:
        return load_real(data_dir, start), "Real data: EIA weekly gasoline (via FRED GASREGW) + FRED WTI (DCOILWTICO)", False
    return make_demo(), "SYNTHETIC DEMO DATA (not real prices)", True
