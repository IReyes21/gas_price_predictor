"""Builds the small dashboard.html: US forecast, a map of region prices, and a gas cost calculator."""
from pathlib import Path

MODEL = "Random forest"   # the model whose forecast is shown

# Each state gets a (row, column) square on a grid, so the "map" is just colored squares.
TILES = {
    "AK": (0, 0), "ME": (0, 10),
    "VT": (1, 9), "NH": (1, 10),
    "WA": (2, 0), "ID": (2, 1), "MT": (2, 2), "ND": (2, 3), "MN": (2, 4), "IL": (2, 5),
    "WI": (2, 6), "MI": (2, 7), "NY": (2, 8), "RI": (2, 9), "MA": (2, 10),
    "OR": (3, 0), "NV": (3, 1), "WY": (3, 2), "SD": (3, 3), "IA": (3, 4), "IN": (3, 5),
    "OH": (3, 6), "PA": (3, 7), "NJ": (3, 8), "CT": (3, 9),
    "CA": (4, 0), "UT": (4, 1), "CO": (4, 2), "NE": (4, 3), "MO": (4, 4), "KY": (4, 5),
    "WV": (4, 6), "VA": (4, 7), "MD": (4, 8), "DE": (4, 9),
    "AZ": (5, 1), "NM": (5, 2), "KS": (5, 3), "AR": (5, 4), "TN": (5, 5), "NC": (5, 6),
    "SC": (5, 7), "DC": (5, 8),
    "OK": (6, 3), "LA": (6, 4), "MS": (6, 5), "AL": (6, 6), "GA": (6, 7),
    "HI": (7, 0), "TX": (7, 3), "FL": (7, 8),
}

# PADD code: (names the data file may have, states in the region)
REGIONS = {
    "1A": (["New England", "Padd 1A"], "CT ME MA NH RI VT"),
    "1B": (["Central Atlantic", "Padd 1B"], "DE DC MD NJ NY PA"),
    "1C": (["Lower Atlantic", "Padd 1C"], "FL GA NC SC VA WV"),
    "2": (["Midwest", "Padd 2"], "IL IN IA KS KY MI MN MO NE ND OH OK SD TN WI"),
    "3": (["Gulf Coast", "Padd 3"], "AL AR LA MS NM TX"),
    "4": (["Rocky Mountain", "Padd 4"], "CO ID MT UT WY"),
    "5": (["West Coast", "Padd 5"], "AK AZ CA HI NV OR WA"),
}
PADD_NAMES = [names[0] for names, states in REGIONS.values()]

TEMPLATE = Path(__file__).resolve().parent / "simple_template.html"


def get_forecast(result):
    """This week's price and next week's forecast for one area."""
    latest = result["explained"][MODEL][-1]
    return {"now": latest["price_now"], "next": latest["predicted_next"], "summary": latest["summary"]}


def region_prices(results):
    """Forecast price for each PADD region we have data for: {"1A": 3.80, ...}."""
    prices = {}
    for code, (names, states) in REGIONS.items():
        for name in names:
            if name in results:
                prices[code] = get_forecast(results[name])["next"]
    return prices


def price_color(price, low, high):
    """Light yellow for the cheapest region, orange for the most expensive."""
    share = 0.5 if high == low else (price - low) / (high - low)
    red = int(255 - share * 25)
    green = int(240 - share * 120)
    blue = int(170 - share * 120)
    return f"rgb({red},{green},{blue})"


def map_html(prices):
    """The tile map plus a small list of region prices."""
    if prices:
        low, high = min(prices.values()), max(prices.values())
    code_of = {}
    for code, (names, states) in REGIONS.items():
        for state in states.split():
            code_of[state] = code

    tiles = ""
    for state, (row, col) in TILES.items():
        code = code_of[state]
        if code in prices:
            color = price_color(prices[code], low, high)
            tip = f"{state}: PADD {code}, ${prices[code]:.2f}"
        else:
            color = "#dddddd"
            tip = state
        tiles += f'<div class="tile" style="grid-row:{row + 1}; grid-column:{col + 1}; background:{color}" title="{tip}">{state}</div>\n'

    if not prices:
        return f'<div class="map">{tiles}</div>\n<p class="small">No region data yet. Run again with an EIA API key to fill in the map.</p>'

    rows = ""
    for code, price in sorted(prices.items(), key=lambda item: item[1]):
        name = REGIONS[code][0][0]
        color = price_color(price, low, high)
        rows += f'<tr><td><span class="dot" style="background:{color}"></span>{name} (PADD {code})</td><td>${price:.2f}</td></tr>\n'
    return f'<div class="map">{tiles}</div>\n<table>{rows}</table>'


def backtest_svg(series):
    """Line chart of actual price vs the model's one-week-ahead forecasts on past weeks."""
    width, height, left, top, bottom = 680, 240, 45, 10, 25
    lines = [("Actual price", "actual", "#222222"),
             ("Assume no change", "Last-week baseline", "#aaaaaa"),
             ("Model forecast", MODEL, "#e8742a")]
    all_prices = series["actual"] + series[MODEL] + series["Last-week baseline"]
    low, high = min(all_prices), max(all_prices)
    count = len(series["actual"])

    def x_of(i):
        return left + i / (count - 1) * (width - left - 10)

    def y_of(price):
        return top + (high - price) / (high - low) * (height - top - bottom)

    svg = f'<svg viewBox="0 0 {width} {height}" width="100%">'
    for k in range(5):   # light horizontal lines with dollar labels
        price = low + (high - low) * k / 4
        y = y_of(price)
        svg += f'<line x1="{left}" x2="{width - 10}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eee"/>'
        svg += f'<text x="{left - 5}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#666">${price:.2f}</text>'
    for name, key, color in lines:
        points = " ".join(f"{x_of(i):.1f},{y_of(price):.1f}" for i, price in enumerate(series[key]))
        svg += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
    svg += f'<text x="{left}" y="{height - 6}" font-size="11" fill="#666">{series["dates"][0]}</text>'
    svg += f'<text x="{width - 10}" y="{height - 6}" font-size="11" text-anchor="end" fill="#666">{series["dates"][-1]}</text>'
    svg += "</svg>"

    legend = ""
    for name, key, color in lines:
        legend += f'<span class="dot" style="background:{color}"></span>{name} &nbsp; '
    return svg + f'<p class="small">{legend}</p>'


def write_dashboard(results, path):
    us = results["United States"]
    meta = us["meta"]
    now = get_forecast(us)
    change = (now["next"] - now["now"]) * 100

    # The dropdown for the calculator: one option per area, value = forecast price
    options = ""
    for label, result in results.items():
        f = get_forecast(result)
        options += f'<option value="{f["next"]:.3f}" data-now="{f["now"]:.3f}">{label} (${f["next"]:.2f})</option>\n'

    # Likely range and back-test accuracy for the U.S.
    range_text = ""
    intervals = us["payload"]["intervals"]
    if intervals:
        half = intervals["half_forecast"][MODEL]
        range_text = f"Likely range: ${now['next'] - half:.2f} to ${now['next'] + half:.2f}"
    mae = us["metrics"].set_index("model")["MAE"] * 100
    accuracy = (f"On {meta['test_weeks']} past weeks the model was off by {mae[MODEL]:.1f}&cent; on average, "
                f"versus {mae['Last-week baseline']:.1f}&cent; if you just assume prices stay the same.")

    banner = "Demo data, not real prices." if meta["is_demo"] else "Source: U.S. Energy Information Administration (EIA)."
    fills = {
        "{{WEEK}}": meta["forecast_week"],
        "{{PRICE}}": f"{now['next']:.2f}",
        "{{NOW}}": f"{now['now']:.2f}",
        "{{CHANGE}}": f"{'up' if change >= 0 else 'down'} {abs(change):.1f}&cent;",
        "{{RANGE}}": range_text,
        "{{SUMMARY}}": now["summary"],
        "{{MAP}}": map_html(region_prices(results)),
        "{{OPTIONS}}": options,
        "{{ACCURACY}}": accuracy,
        "{{BACKTEST}}": backtest_svg(us["payload"]["test_series"]),
        "{{BANNER}}": banner,
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fills.items():
        html = html.replace(key, value)
    Path(path).write_text(html, encoding="utf-8")
