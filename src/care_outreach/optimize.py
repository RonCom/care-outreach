"""Outreach plan for the deployment year: who gets which program under budget, staff-hour and capacity limits.

Plans compared at the same limits, scored with the simulated truth (expected acute events averted):
  risk_nurse   current practice: call the highest-risk members (champion risk model) until money or hours run out
  risk_nurse_digital  the same, then enrol the next-highest-risk members in the digital program with the money left
  risk_nba     same risk order, but each member gets the recommender's best program
  ilp_uplift   integer program on predicted benefit (multiple-choice knapsack, scipy / HiGHS):
                 maximize   sum_ia b_ia x_ia
                 subject to sum_ia cost_a x_ia <= budget,  sum_ia hours_a x_ia <= staff hours,
                            sum_i x_i,home_visit <= home-visit capacity,  sum_a x_ia <= 1 per member
  ilp_known    same program with the true effects (and model-estimated baseline risk): the ceiling for better
               effect estimates
"""
from __future__ import annotations

import json
import logging

import mlflow
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from . import data, simulate, uplift
from .config import Config

log = logging.getLogger(__name__)
PROGRAMS = simulate.ACTIONS[1:]


def limits(cfg: Config, n: int) -> dict:
    p = cfg["plan"]
    return {"budget": p["budget_per_member"] * n, "hours": p["staff_hours_per_member"] * n,
            "home": int(p["max_home_visit_share"] * n)}


def action_table(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    a = cfg["actions"]
    return (np.array([a[k]["cost"] for k in simulate.ACTIONS], float),
            np.array([a[k]["staff_hours"] for k in simulate.ACTIONS], float))


def plan_ilp(b: np.ndarray, cost: np.ndarray, hours: np.ndarray, lim: dict) -> np.ndarray:
    """b: benefit per member x action (column 0 = none). Returns the action index per member."""
    n = len(b)
    ii, aa = np.nonzero(b[:, 1:] > 0)          # only programs with positive predicted benefit are candidates
    aa = aa + 1
    m = len(ii)
    home = simulate.ACTIONS.index("home_visit")
    A = sparse.vstack([
        sparse.csr_matrix(cost[aa][None, :]),
        sparse.csr_matrix(hours[aa][None, :]),
        sparse.csr_matrix((aa == home).astype(float)[None, :]),
        sparse.csr_matrix((np.ones(m), (ii, np.arange(m))), shape=(n, m)),
    ])
    ub = np.r_[lim["budget"], lim["hours"], lim["home"], np.ones(n)]
    res = milp(-b[ii, aa], constraints=LinearConstraint(A, -np.inf, ub), integrality=np.ones(m),
               bounds=Bounds(0, 1), options={"mip_rel_gap": 1e-3, "time_limit": 180})
    if res.x is None:
        raise RuntimeError(f"outreach ILP failed: {res.message}")
    out = np.zeros(n, int)
    pick = res.x > 0.5
    out[ii[pick]] = aa[pick]
    return out


def plan_ranked(order: np.ndarray, choice: np.ndarray, cost, hours, lim) -> np.ndarray:
    """Walk down `order`; give each member their `choice` if it still fits the remaining limits."""
    out = np.zeros(len(choice), int)
    left = dict(lim)
    home = simulate.ACTIONS.index("home_visit")
    for i in order:
        a = choice[i]
        if a == 0 or cost[a] > left["budget"] or hours[a] > left["hours"] or (a == home and left["home"] < 1):
            continue
        out[i] = a
        left["budget"] -= cost[a]
        left["hours"] -= hours[a]
        left["home"] -= a == home
        if left["budget"] < cost[1:].min():
            break
    return out


def _nurse_then_digital(order, cost, hours, lim):
    first = plan_ranked(order, np.full(len(order), 1), cost, hours, lim)
    left = {"budget": lim["budget"] - cost[first].sum(), "hours": lim["hours"] - hours[first].sum(), "home": 0}
    rest = order[first[order] == 0]
    second = plan_ranked(rest, np.full(len(order), simulate.ACTIONS.index("digital")), cost, hours, left)
    return np.maximum(first, second)


def score(plan: np.ndarray, y0: np.ndarray, r: np.ndarray, cost: np.ndarray, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(y0)) if w is None else w
    averted = (w * simulate.expected_averted(y0, r, plan)).sum()
    spent = (w * cost[plan]).sum()
    return {"events_averted": float(averted), "spend": float(spent),
            "averted_per_10k": float(1e4 * averted / spent) if spent else 0.0,
            "members_reached": float((w * (plan > 0)).sum())}


def run(cfg: Config) -> dict:
    panel, _ = data.load(cfg)
    year = cfg["plan"]["year"]
    fit = uplift.run(cfg)
    model = fit["model"]
    d = panel[panel["year"] == year].merge(
        pd.read_parquet(cfg.path("data_dir") / "risk_scores.parquet")[["pid", "year", "risk"]], on=["pid", "year"])
    X = d[uplift.COLS].to_numpy(np.float32)
    y0, risk = d["label"].to_numpy(), d["risk"].to_numpy()
    r = simulate.rrr(d)
    cost, hours = action_table(cfg)
    lim = limits(cfg, len(d))
    b_hat = uplift.benefit(model, X)
    b_known = risk[:, None] * r
    order = np.argsort(-risk, kind="stable")
    plans = {
        "risk_nurse": plan_ranked(order, np.full(len(d), 1), cost, hours, lim),
        "risk_nurse_digital": _nurse_then_digital(order, cost, hours, lim),
        "risk_nba": plan_ranked(order, b_hat[:, 1:].argmax(1) + 1, cost, hours, lim),
        "ilp_uplift": plan_ilp(b_hat, cost, hours, lim),
        "ilp_known": plan_ilp(b_known, cost, hours, lim),
    }
    res = {"year": year, "members": len(d), "limits": lim,
           "untreated_events": int(y0.sum()), "plans": {}}
    for k, p in plans.items():
        res["plans"][k] = score(p, y0, r, cost) | {
            "mix": {simulate.ACTIONS[a]: int(n) for a, n in zip(*np.unique(p, return_counts=True)) if a > 0}}
    # member bootstrap for the gain over current practice
    rng = np.random.default_rng(cfg["risk"]["seed"])
    diffs = {k: [] for k in ("ilp_uplift", "risk_nba", "risk_nurse_digital")}
    for _ in range(1000):
        w = np.bincount(rng.integers(0, len(d), len(d)), minlength=len(d)).astype(float)
        base = score(plans["risk_nurse"], y0, r, cost, w)["events_averted"]
        for k, v in diffs.items():
            v.append(score(plans[k], y0, r, cost, w)["events_averted"] / base - 1)
    for k, v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[f"{k}_vs_risk_nurse_pct"] = {"est": res["plans"][k]["events_averted"]
                                        / res["plans"]["risk_nurse"]["events_averted"] - 1, "ci95": [lo, hi]}
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("outreach-plan")
    with mlflow.start_run(run_name=f"plan-{year}"):
        mlflow.log_params({"year": year} | {f"limit_{k}": v for k, v in lim.items()})
        mlflow.log_metrics({f"{k}_{m}": v for k, p in res["plans"].items() for m, v in p.items() if m != "mix"})
    (cfg.reports / "outreach_plan.json").write_text(json.dumps(res, indent=2, default=float))
    _chart(res, cfg)
    log.info("plan %s: %s", year, {k: round(v["events_averted"], 1) for k, v in res["plans"].items()})
    return res


def _chart(res: dict, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    from .charts import COLORS, save, style
    style()
    labels = {"risk_nurse": "Nurse calls to\nhighest risk", "risk_nba": "Highest risk,\nrecommended program",
              "risk_nurse_digital": "Nurse calls, then digital,\nby risk", "ilp_uplift": "Optimized plan\n(estimated effects)",
              "ilp_known": "Optimized plan\n(true effects, ceiling)"}
    keys = list(labels)
    vals = [res["plans"][k]["events_averted"] for k in keys]
    spend = [res["plans"][k]["spend"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    cols = [COLORS["static"], COLORS["static"], COLORS["static"], COLORS["adaptive"], COLORS["grid_dark"]]
    bars = ax.bar(range(len(keys)), vals, color=cols)
    for b, v, sp in zip(bars, vals, spend):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}\n${sp / 1000:.0f}K", ha="center", fontsize=8)
    ax.set_xticks(range(len(keys)), [labels[k] for k in keys], fontsize=8)
    ax.set_ylabel("Expected acute events averted")
    ax.set_title(f"Outreach plans, {res['year']}: same budget and staff-hour limits")
    ax.set_ylim(0, max(vals) * 1.2)
    save(fig, cfg, "outreach_plans.png")
