import numpy as np
import pandas as pd

from care_outreach import optimize, simulate


def members(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(40, 90, n), "chf": rng.integers(0, 2, n), "copd": rng.integers(0, 2, n),
        "dementia": rng.integers(0, 2, n), "ckd": rng.integers(0, 2, n), "n_meds": rng.integers(0, 12, n),
        "ed_1y": rng.integers(0, 3, n), "ip_1y": rng.integers(0, 2, n), "acute_2y": rng.integers(0, 3, n),
        "diabetes": rng.integers(0, 2, n), "obesity": rng.integers(0, 2, n), "label": rng.integers(0, 2, n)})


def test_rrr_bounds_and_none():
    r = simulate.rrr(members())
    assert r.shape[1] == len(simulate.ACTIONS)
    assert (r[:, 0] == 0).all() and (r >= 0).all() and (r <= 0.6).all()


def test_outcome_never_creates_events():
    d = members()
    y0 = d["label"].to_numpy()
    y = simulate.outcome(y0, np.full(len(d), 0.5), np.random.default_rng(1))
    assert (y <= y0).all()


def test_ilp_respects_limits_and_beats_ranked():
    rng = np.random.default_rng(2)
    n = 300
    b = np.column_stack([np.zeros(n), rng.random((n, 4)) * 0.05])
    cost = np.array([0, 60, 150, 450, 25.0])
    hours = np.array([0, 1, 2, 4, 0.0])
    lim = {"budget": 3000, "hours": 20, "home": 2}
    plan = optimize.plan_ilp(b, cost, hours, lim)
    assert cost[plan].sum() <= lim["budget"] + 1e-6
    assert hours[plan].sum() <= lim["hours"] + 1e-6
    assert (plan == 3).sum() <= lim["home"]
    ranked = optimize.plan_ranked(np.argsort(-b.max(1)), b[:, 1:].argmax(1) + 1, cost, hours, lim)
    val = lambda p: b[np.arange(n), p].sum()
    assert val(plan) >= val(ranked) - 1e-9


def test_steering_lp_respects_capacity():
    from care_outreach import steer
    rng = np.random.default_rng(3)
    v = rng.random((40, 6)) - 0.3
    cap = np.array([2, 0, 5, 1, 3, 1.0])
    ch = steer.plan_lp(v, cap)
    for j in range(6):
        assert (ch == j).sum() <= cap[j]
    assert all(v[i, ch[i]] > 0 for i in range(40) if ch[i] >= 0)
