"""Explainable-AI layer: split every forecast into "what pushed the price up or down".

Both explainers satisfy an exact additive identity for the predicted weekly change:

    predicted_change = base + sum_j contribution_j

* Linear regression: contribution_j = coef_j * (x_j - mean_j). This is exactly the
  SHAP value for a linear model with independent features.
* Random forest: "tree-path" decomposition (Saabas / treeinterpreter). Walk each
  tree from root to leaf; every split changes the node's average prediction, and
  that change is credited to the feature used in the split. Average over trees.
  It is faster and dependency-free vs. shap.TreeExplainer, but credit can differ
  slightly from true Shapley values. Swap in `shap` for the final report if it is installed.
"""
from __future__ import annotations

import numpy as np

from features import FEATURE_GROUPS, FEATURE_LABELS


def linear_explain(model, scaler, X: np.ndarray):
    w = model.coef_ / scaler.scale_
    contrib = (X - scaler.mean_) * w
    return float(model.intercept_), contrib


def forest_explain(forest, X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)  # sklearn trees split on float32 thresholds
    n, p = X.shape
    contrib = np.zeros((n, p))
    base = 0.0
    for est in forest.estimators_:
        tree = est.tree_
        val = tree.value[:, 0, 0]
        base += val[0]
        paths = est.decision_path(X)
        for i in range(n):
            nodes = paths.indices[paths.indptr[i] : paths.indptr[i + 1]]  # root -> leaf
            for a, b in zip(nodes[:-1], nodes[1:]):
                contrib[i, tree.feature[a]] += val[b] - val[a]
    k = len(forest.estimators_)
    return base / k, contrib / k


def to_groups(contrib_row: dict[str, float]) -> dict[str, float]:
    return {g: float(sum(contrib_row[f] for f in fs)) for g, fs in FEATURE_GROUPS.items()}


def describe_feature(name: str, v: float) -> str:
    """One plain-English clause for a feature value."""
    if name.startswith("gas_chg"):
        when = {"gas_chg_1w": "this week", "gas_chg_2w": "last week", "gas_chg_3w": "two weeks ago"}[name]
        return f"the pump price {'rose' if v >= 0 else 'fell'} {abs(v) * 100:.1f}¢ {when}"
    if name.startswith("crude_chg"):
        when = {"1": "this week", "2": "last week", "3": "two weeks ago", "4": "three weeks ago"}[name[10]]
        return f"crude oil {'rose' if v >= 0 else 'fell'} ${abs(v) * 42:.2f}/barrel {when}"
    if name == "margin_gap_vs_52w":
        return f"the pump price sits {abs(v) * 100:.0f}¢ {'above' if v >= 0 else 'below'} its usual markup over crude"
    if name == "summer_season":
        return "the forecast week is in summer driving season" if v else "the forecast week is outside summer driving season"
    if name == "holiday_week":
        return "a holiday falls in the forecast week" if v else "no holiday falls in the forecast week"
    return "the typical seasonal pattern for that time of year"


def top_drivers(feat_contrib: dict[str, float], feat_value: dict[str, float], top: int = 3, min_cents: float = 0.2):
    """Largest group-level pushes, each with the single feature inside it that matters most."""
    groups = to_groups(feat_contrib)
    out = []
    for g, c in sorted(groups.items(), key=lambda kv: -abs(kv[1]))[:top]:
        if abs(c) * 100 < min_cents:
            continue
        best = max(FEATURE_GROUPS[g], key=lambda f: abs(feat_contrib[f]))
        out.append({"group": g, "cents": round(c * 100, 2), "because": describe_feature(best, feat_value[best])})
    return out


def sentence(price: float, chg: float, drivers: list[dict]) -> str:
    head = f"Forecast ${price + chg:.2f}, {'up' if chg >= 0 else 'down'} {abs(chg) * 100:.1f}¢ from this week's ${price:.2f}."
    if not drivers:
        return head + " No single factor stands out; the model sees a flat week."
    bits = [f"{d['group']} {'adds' if d['cents'] >= 0 else 'subtracts'} {abs(d['cents']):.1f}¢ ({d['because']})" for d in drivers]
    return head + " Main drivers: " + "; ".join(bits) + "."


def feature_table(feat_contrib: dict[str, float], feat_value: dict[str, float]):
    return [
        {"feature": f, "label": FEATURE_LABELS[f], "value": float(feat_value[f]), "cents": round(float(feat_contrib[f]) * 100, 3)}
        for f in feat_contrib
    ]
