"""PNG figures for the report / slides. Same palette and chart rules as the dashboard."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # model identity: linear = blue, forest = orange, in every chart
UP, DOWN = "#e34948", "#2a78d6"  # diverging poles for "pushes price up / down"
LINEAR, FOREST, BASELINE = "Linear regression", "Random forest", "Last-week baseline"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _fig(w=9, h=4.4, left=0.08):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=left, right=0.97, top=0.80, bottom=0.2)
    _style(ax)
    return fig, ax


def _title(ax, title, sub):
    """Title block pinned to the figure's left edge, above the plot (not indented by y labels)."""
    fig = ax.figure
    y1 = ax.get_position().y1
    fig.text(0.012, y1 + 0.115, title, fontsize=13, fontweight="semibold", color=INK, va="bottom")
    fig.text(0.012, y1 + 0.045, sub, fontsize=9, color=INK2, va="bottom")


def make_figures(payload: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    demo = payload["meta"]["is_demo"]
    tag = "  |  SYNTHETIC DEMO DATA" if demo else ""

    # 1. forecast vs actual on the held-out weeks ------------------------------------------------
    ts = payload["test_series"]
    dates = pd.to_datetime(ts["dates"])
    fig, ax = _fig(9.5, 4.6)
    ax.plot(dates, ts["actual"], color=INK, lw=2, label="Actual price", solid_capstyle="round")
    ax.plot(dates, ts[BASELINE], color=MUTED, lw=2, label=BASELINE, solid_capstyle="round")
    ax.plot(dates, ts[LINEAR], color=BLUE, lw=2, label=LINEAR, solid_capstyle="round")
    ax.plot(dates, ts[FOREST], color=ORANGE, lw=2, label=FOREST, solid_capstyle="round")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:.2f}")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, -0.22), fontsize=9, labelcolor=INK2)
    _title(ax, "1-week-ahead forecasts vs. actual pump price", f"Held-out test weeks, $ per gallon{tag}")
    fig.savefig(out / "forecast_vs_actual.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 2. why this forecast: contributions for the upcoming week (random forest) ------------------
    row = payload["explanations"][FOREST][-1]
    items = [("Typical weekly drift", row["base_cents"])] + list(row["groups_cents"].items())
    net = row["predicted_next"] - row["price_now"]
    fig, ax = _fig(9.5, 3.6, left=0.27)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    labels = [n for n, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    bars = ax.barh(labels, vals, height=0.5, color=[UP if v >= 0 else DOWN for v in vals])
    lim = max(abs(v) for v in vals + [net * 100]) * 1.35 or 1
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color=AXIS, lw=1)
    for b, v in zip(bars, vals):
        ax.text(v + (lim * 0.02 if v >= 0 else -lim * 0.02), b.get_y() + b.get_height() / 2, f"{'↑' if v >= 0 else '↓'} {abs(v):.1f}¢",
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.1f}¢" if v else "0¢")
    ax.tick_params(axis="y", colors=INK2, labelsize=10)
    _title(
        ax,
        f"Why we forecast ${row['predicted_next']:.2f} for the week of {payload['meta']['forecast_week']}",
        f"Random forest. Net change {net * 100:+.1f}¢ from this week's ${row['price_now']:.2f}. Red pushes the price up, blue pushes it down.{tag}",
    )
    fig.savefig(out / "forecast_explanation.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 3. what each model relies on ---------------------------------------------------------------
    gm = payload["group_mean_abs_cents"]
    groups = sorted(gm[FOREST], key=lambda g: gm[FOREST][g])
    fig, ax = _fig(9.5, 3.8, left=0.27)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    h = 0.32
    ys = range(len(groups))
    ax.barh([y + h / 2 + 0.01 for y in ys], [gm[LINEAR][g] for g in groups], height=h, color=BLUE, label=LINEAR)
    ax.barh([y - h / 2 - 0.01 for y in ys], [gm[FOREST][g] for g in groups], height=h, color=ORANGE, label=FOREST)
    ax.set_yticks(list(ys), groups)
    ax.tick_params(axis="y", colors=INK2, labelsize=10)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}¢")
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, -0.22), fontsize=9, labelcolor=INK2)
    _title(ax, "What each model relies on", f"Average size of each factor's push on the weekly forecast, across all test weeks{tag}")
    fig.savefig(out / "feature_importance.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
