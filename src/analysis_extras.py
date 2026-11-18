"""Extra analyses that sit on top of the forecasting models.

1. crude_asymmetry   "Rockets and feathers": do pump prices follow crude RISES faster than crude FALLS?
2. passthrough_speed How much, and how fast, does a crude move reach the pump (per area)?
3. flat_crude_shift  What crude price move would make a model forecast a flat pump price?

Everything is plain numpy (no statsmodels). The two regressions are distributed-lag models of the weekly pump-price
change on the last L weekly crude changes, with Newey-West standard errors because weekly price changes are
serially correlated.

Units: pump price in $/gal, crude in $/bbl. Reported pass-through is in cents per gallon per $1/barrel of crude move.
Full pass-through would be 100/42 = 2.38 cents (a barrel is 42 gallons).
"""
from __future__ import annotations

import math

import numpy as np

CENTS_PER_DOLLAR_BBL_FULL = 100.0 / 42.0
MIN_OBS = 80


def _ols_nw(X: np.ndarray, y: np.ndarray, q: int):
    """OLS with Newey-West (HAC) covariance, Bartlett weights, bandwidth q."""
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for j in range(1, q + 1):
        G = Xu[j:].T @ Xu[:-j]
        S += (1 - j / (q + 1)) * (G + G.T)
    V = xtx_inv @ S @ xtx_inv * (n / max(n - k, 1))   # small-sample scaling
    return beta, V, n


def _p_two_sided(t: float, dof: int) -> float:
    """Student-t p-value (more cautious than the normal curve with ~250 weekly observations)."""
    from scipy import stats   # scipy is already installed with scikit-learn
    return float(2 * stats.t.sf(abs(t), max(dof, 1)))


def _lagged(dC: np.ndarray, L: int):
    """Rows t = L..end. Columns: dC[t], dC[t-1], ... dC[t-L]."""
    n = len(dC)
    return np.column_stack([dC[L - k : n - k] for k in range(L + 1)])


def crude_asymmetry(gas: np.ndarray, crude: np.ndarray, L: int = 6) -> dict | None:
    """Cumulative pump-price response to a $1/bbl crude RISE versus a $1/bbl FALL, by weeks since the move.

    Model: dGas_t = a + sum_k up_k * max(dCrude_{t-k},0) + sum_k down_k * min(dCrude_{t-k},0) + e_t, k = 0..L.
    A positive coefficient means the pump price moves in the same direction as crude. The cumulative sums
    sum_{i<=k} up_i and sum_{i<=k} down_i are comparable (both in cents per gallon per $1/bbl).
    """
    gas, crude = np.asarray(gas, float), np.asarray(crude, float)
    dG, dC = np.diff(gas), np.diff(crude)
    n_eff = len(dG) - L
    if n_eff < MIN_OBS:
        return None
    y = dG[L:] * 100.0                                  # cents per gallon
    lag = _lagged(dC, L)
    up, dn = np.maximum(lag, 0.0), np.minimum(lag, 0.0)
    X = np.column_stack([np.ones(len(y)), up, dn])
    if (up.sum(axis=0) == 0).any() or (dn.sum(axis=0) == 0).any():
        return None
    beta, V, n = _ols_nw(X, y, q=L)
    b_up, b_dn = beta[1 : L + 2], beta[L + 2 :]
    out = {"lags": list(range(L + 1)), "rise": [], "fall": [], "rise_se": [], "fall_se": [], "diff": [], "diff_se": []}
    for k in range(L + 1):
        c_up = np.zeros(len(beta)); c_up[1 : k + 2] = 1
        c_dn = np.zeros(len(beta)); c_dn[L + 2 : L + 3 + k] = 1
        for name, c in (("rise", c_up), ("fall", c_dn), ("diff", c_up - c_dn)):
            out[name].append(float(c @ beta))
            out[name + "_se"].append(float(math.sqrt(max(c @ V @ c, 0.0))))
    dof = n - len(beta)
    out["diff_p"] = [_p_two_sided(d_ / s_ if s_ > 0 else 0.0, dof) for d_, s_ in zip(out["diff"], out["diff_se"])]
    d, se = out["diff"][-1], out["diff_se"][-1]
    t = d / se if se > 0 else 0.0
    p = out["diff_p"][-1]
    # speed: earliest horizon where the rise/fall gap is clearly non-zero. Seven horizons are checked, so the bar is
    # a Bonferroni-corrected 0.05/(L+1), not 0.05.
    bar = 0.05 / (L + 1)
    sig = [k for k in range(L + 1) if out["diff_p"][k] < bar]
    out["speed_lag"] = sig[0] if sig else None
    out["speed_dir"] = None if not sig else ("rise_faster" if out["diff"][sig[0]] > 0 else "fall_faster")
    out["speed_bar"] = bar
    out.update({"p_value": p, "t_stat": t, "n_weeks": int(n), "full_pass_through": CENTS_PER_DOLLAR_BBL_FULL})
    if p < 0.05 and d > 0:
        out["verdict"] = "rockets_feathers"
    elif p < 0.05 and d < 0:
        out["verdict"] = "reverse"
    else:
        out["verdict"] = "no_clear_difference"
    return out


def passthrough_speed(gas: np.ndarray, crude: np.ndarray, L: int = 6) -> dict | None:
    """Symmetric distributed lag: how much of a crude move reaches the pump, and how quickly."""
    gas, crude = np.asarray(gas, float), np.asarray(crude, float)
    dG, dC = np.diff(gas), np.diff(crude)
    n_eff = len(dG) - L
    if n_eff < MIN_OBS:
        return None
    y = dG[L:] * 100.0
    X = np.column_stack([np.ones(len(y)), _lagged(dC, L)])
    beta, V, n = _ols_nw(X, y, q=L)
    cum = np.cumsum(beta[1:])
    ones = np.zeros(len(beta)); ones[1:] = 1
    total, se = float(cum[-1]), float(math.sqrt(max(ones @ V @ ones, 0.0)))
    reliable = total > 0.25 * CENTS_PER_DOLLAR_BBL_FULL and total > 1.64 * se   # clearly positive, not noise
    share = (cum / total).tolist() if total > 0 else [float("nan")] * len(cum)
    first = lambda thr: next((k for k, s in enumerate(share) if s >= thr), None) if reliable else None
    return {
        "lags": list(range(L + 1)),
        "cumulative": [float(c) for c in cum],
        "share": share if reliable else None,
        "total": total,
        "total_se": se,
        "total_pct_of_full": total / CENTS_PER_DOLLAR_BBL_FULL * 100.0,
        "same_week_share": share[0] if reliable else None,
        "weeks_to_half": first(0.5),
        "weeks_to_80": first(0.8),
        "reliable": bool(reliable),
        "n_weeks": int(n),
    }


def flat_crude_shift(predict_change, base_crude: float, lo: int = -40, hi: int = 40):
    """Smallest-magnitude crude shift ($/bbl) at which predict_change(shift) crosses zero, or None.

    predict_change(shift) returns the model's predicted pump change ($/gal) if this week's crude moved by `shift`.
    """
    shifts = np.arange(lo, hi + 1, 1.0)
    vals = np.array([predict_change(float(s)) for s in shifts])
    hits = []
    for i in range(len(shifts) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0:
            hits.append(float(shifts[i]))
        elif a * b < 0:
            hits.append(float(shifts[i] + (0 - a) / (b - a)))
    if vals[-1] == 0:
        hits.append(float(shifts[-1]))
    if not hits:
        return None
    s = min(hits, key=abs)
    return {"shift": round(s, 1), "crude_then": round(base_crude + s, 2)}
