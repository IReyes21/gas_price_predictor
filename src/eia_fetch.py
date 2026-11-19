"""Pull live weekly data from the EIA Open Data API (v2) into the data/ folder.

What it writes (same CSV shape the FRED downloads use, so nothing else changes):
    data/GASREGW.csv          weekly U.S. regular gasoline price           (area "U.S.")
    data/DCOILWTICO.csv       daily WTI crude oil spot price               (series RWTC)
    data/regions/<Area>.csv   weekly regular gasoline price per area       (states and/or cities; PADD regions only if asked for)

Needs a free API key: https://www.eia.gov/opendata/register.php
Pass it with --eia-key or the EIA_API_KEY environment variable. The key is only sent to api.eia.gov
and is never written to disk or printed.

Standard library only (urllib), so no extra packages.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = os.environ.get("EIA_API_BASE", "https://api.eia.gov/v2")  # overridable so the code can be tested offline
PAGE = 5000  # EIA's maximum rows per request

# EIA publishes weekly retail prices for exactly these 9 states (plus regions). Not all 50.
STATES = ["California", "Colorado", "Florida", "Massachusetts", "Minnesota", "New York", "Ohio", "Texas", "Washington"]
DEFAULT_AREAS = "states,metros,regions"   # 9 states + cities + the 7 PADD regions (the dashboard map uses regions only to shade the 41 other states)


class EiaError(RuntimeError):
    pass


def _get(path: str, key: str, params: dict | None = None) -> dict:
    q = [("api_key", key)]
    for k, v in (params or {}).items():
        if isinstance(v, (list, tuple)):
            q += [(k, x) for x in v]
        else:
            q.append((k, v))
    url = f"{BASE}/{path}?{urllib.parse.urlencode(q)}"
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:   # build our own message: the URL contains the key
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            if e.code in (401, 403):
                raise EiaError("EIA rejected the API key (HTTP %d). Check that it is copied exactly; keys come from "
                               "https://www.eia.gov/opendata/register.php. %s" % (e.code, body)) from None
            if e.code == 429 or e.code >= 500:
                last = "HTTP %d" % e.code
                time.sleep(2 * (attempt + 1))
                continue
            raise EiaError("EIA returned HTTP %d for /%s. %s" % (e.code, path, body)) from None
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(getattr(e, "reason", e))
            time.sleep(2 * (attempt + 1))
    raise EiaError("Could not reach api.eia.gov (%s). Check your internet connection and try again." % last)


def _rows(path: str, key: str, params: dict) -> list[dict]:
    """Fetch every page of a /data endpoint."""
    out, offset = [], 0
    while True:
        resp = _get(path, key, {**params, "data[0]": "value", "length": PAGE, "offset": offset,
                                "sort[0][column]": "period", "sort[0][direction]": "asc"})
        body = resp.get("response")
        if body is None:
            raise EiaError("Unexpected reply from EIA for /%s: %s" % (path, str(resp)[:200]))
        page = body.get("data", [])
        out += page
        total = int(body.get("total", len(out)))
        offset += PAGE
        if not page or offset >= total:
            return out


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", name or "")).strip().lower()


def _label(area_name: str) -> str:
    """'Central Atlantic (PADD 1B)' -> 'Central Atlantic'; 'STATE OF TEXAS' -> 'Texas'."""
    s = re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", area_name)).strip()
    s = re.sub(r"^state of ", "", s, flags=re.I)
    return s.title() if s.isupper() else s


def _wanted(areas: str) -> tuple[set[str], bool, bool]:
    """Parse --eia-areas: tokens 'states', 'regions', 'metros', 'all', or area names, comma separated.

    Returns (names wanted, want all PADD regions, want metro areas). 'metros' means every area EIA lists that is
    not the U.S., a state, or a PADD region: EIA publishes about ten cities weekly.
    """
    toks = [t.strip() for t in areas.split(",") if t.strip()]
    low = {t.lower() for t in toks}
    all_ = "all" in low
    names = {_norm(t) for t in toks if t.lower() not in ("states", "regions", "metros", "all")}
    if all_ or "states" in low:
        names |= {s.lower() for s in STATES}
    return names, all_ or "regions" in low, all_ or "metros" in low


def _is_region(area_name: str) -> bool:
    return bool(re.search(r"padd", area_name or "", re.I)) and "less" not in (area_name or "").lower()


def _write(path: Path, col: str, dates, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"observation_date": pd.to_datetime(list(dates)).strftime("%Y-%m-%d"), col: list(values)}).to_csv(path, index=False)


def _backup_once(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)


def fetch_all(key: str, data_dir: Path, areas: str = DEFAULT_AREAS, start: str = "2010-01-01", log=print) -> dict:
    """Download U.S. gas, WTI crude and the requested areas into data_dir. Returns a small summary dict."""
    data_dir = Path(data_dir)
    if not key:
        raise EiaError("No EIA API key. Get a free one at https://www.eia.gov/opendata/register.php, then pass "
                       "--eia-key YOUR_KEY or set the EIA_API_KEY environment variable.")
    want_names, want_regions, want_metros = _wanted(areas)

    # 1) weekly regular gasoline, every area (filtered on our side so we do not depend on EIA's area codes)
    log("EIA: downloading weekly regular gasoline prices ...")
    rows = _rows("petroleum/pri/gnd/data", key, {"frequency": "weekly", "facets[product][]": ["EPMR"],
                                                 "facets[process][]": ["PTE"], "start": start})
    rows = [r for r in rows if r.get("value") not in (None, "") and r.get("product", "EPMR") == "EPMR" and r.get("process", "PTE") == "PTE"]
    by_area: dict[str, list[dict]] = {}
    for r in rows:
        by_area.setdefault(r.get("area-name") or r.get("duoarea") or "?", []).append(r)
    if not by_area:
        raise EiaError("EIA returned no regular-gasoline rows. Try again later or check your key.")

    def series(rs):
        s = pd.Series({pd.Timestamp(r["period"]): float(r["value"]) for r in rs}).sort_index()
        return s[~s.index.duplicated(keep="last")]

    summary = {"us_weeks": 0, "areas": [], "missing": [], "crude_days": 0}

    # U.S. national
    us_name = next((n for n in by_area if _norm(n) in ("u.s.", "us", "united states", "u.s")), None)
    if us_name is None:
        raise EiaError("Could not find the U.S. series in EIA's reply. Areas seen: " + ", ".join(sorted(by_area)[:12]))
    us = series(by_area[us_name])
    gas_path = data_dir / "GASREGW.csv"
    _backup_once(gas_path)
    _write(gas_path, "GASREGW", us.index, us.values)
    summary["us_weeks"] = len(us)
    log(f"  U.S.: {len(us)} weeks, latest {us.index[-1].date()} at ${us.iloc[-1]:.3f}  ->  data/GASREGW.csv")

    # states / regions
    reg_dir = data_dir / "regions"
    picked = {}
    for name, rs in by_area.items():
        if name == us_name:
            continue
        is_state = _norm(_label(name)) in {s.lower() for s in STATES}
        is_metro = not is_state and not _is_region(name) and "less" not in name.lower()
        if _norm(_label(name)) in want_names or (want_regions and _is_region(name)) or (want_metros and is_metro):
            picked[_label(name)] = rs
    # Move region files left by an earlier fetch (e.g. the old Padd_*.csv) out of the way, so they are not modeled again.
    keep = {label.replace(" ", "_") + ".csv" for label in picked}
    manifest = reg_dir / ".fetched.json"
    old = set(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.exists() else set()
    if reg_dir.is_dir():
        for f in sorted(reg_dir.glob("*.csv")):
            if f.name not in keep and (f.name in old or f.name.lower().startswith("padd")):
                (reg_dir / "_old").mkdir(exist_ok=True)
                shutil.move(str(f), str(reg_dir / "_old" / f.name))
                log(f"  moved aside (no longer requested): data/regions/{f.name}  ->  data/regions/_old/")
    found = {_norm(k) for k in picked}
    summary["missing"] = sorted(w for w in want_names if w not in found)
    for label, rs in sorted(picked.items()):
        s = series(rs)
        _write(reg_dir / (label.replace(" ", "_") + ".csv"), "price", s.index, s.values)
        summary["areas"].append(label)
        log(f"  {label}: {len(s)} weeks, latest {s.index[-1].date()} at ${s.iloc[-1]:.3f}  ->  data/regions/{label.replace(' ', '_')}.csv")
    reg_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(sorted(keep)), encoding="utf-8")
    if summary["missing"]:
        log("  NOT FOUND in EIA's reply: " + ", ".join(m.title() for m in summary["missing"]))
        log("  Areas EIA does list: " + ", ".join(sorted(_label(n) for n in by_area if n != us_name)[:40]))

    # 2) daily WTI crude (full history from `start`, which also fixes the short/stale crude file)
    log("EIA: downloading daily WTI crude oil prices ...")
    crude = _rows("petroleum/pri/spt/data", key, {"frequency": "daily", "facets[series][]": ["RWTC"], "start": start})
    crude = [r for r in crude if r.get("value") not in (None, "") and r.get("series", "RWTC") == "RWTC"]
    if not crude:
        raise EiaError("EIA returned no WTI crude rows (series RWTC). The gasoline files were still written.")
    c = series(crude)
    wti_path = data_dir / "DCOILWTICO.csv"
    _backup_once(wti_path)
    _write(wti_path, "DCOILWTICO", c.index, c.values)
    summary["crude_days"] = len(c)
    log(f"  WTI: {len(c)} days, latest {c.index[-1].date()} at ${c.iloc[-1]:.2f}  ->  data/DCOILWTICO.csv")
    return summary