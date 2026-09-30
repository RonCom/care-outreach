"""Semi-synthetic outreach response.

Synthea gives each member's untreated outcome (Y0: did an ED visit or inpatient stay happen that year). No public
dataset records how members respond to outreach, so program effects are simulated on top of it, the approach used
by semi-synthetic causal benchmarks (e.g. IHDP): an action averts an event that would have happened with a
member-specific probability, its relative risk reduction (RRR). Effects are heterogeneous by design, so the members
at highest risk are not always the ones who benefit most:

  nurse_call  helps members with heart failure / COPD and recent acute use; less for under-55s
  med_review  helps polypharmacy (5+, 8+ active medications) and CKD
  home_visit  helps the oldest members, heart failure / dementia, recent inpatient stays; expensive
  digital     cheap; works through engagement, which falls with age; helps diabetes / obesity

Because the effects are known, any policy can be scored exactly: expected events averted = sum_i Y0_i * RRR(a_i, x_i).
Models never see RRR or Y0 of the year they plan for; they learn from a randomized pilot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIONS = ["none", "nurse_call", "med_review", "home_visit", "digital"]


def rrr(df: pd.DataFrame, digital_boost: float = 1.0) -> np.ndarray:
    """True relative risk reduction, shape (rows, actions). `digital_boost` scales the digital program's effect
    (used by the bandit's mid-stream program change)."""
    f = {c: df[c].to_numpy(float) for c in ("age", "chf", "copd", "dementia", "ckd", "n_meds", "ed_1y", "ip_1y",
                                            "acute_2y", "diabetes", "obesity")}
    recent_acute = ((f["ed_1y"] + f["ip_1y"] + f["acute_2y"]) > 0).astype(float)
    heart_lung = np.maximum(f["chf"], f["copd"])
    nurse = 0.08 + 0.12 * heart_lung + 0.06 * recent_acute - 0.05 * (f["age"] < 55)
    med = 0.02 + 0.16 * (f["n_meds"] >= 8) + 0.08 * ((f["n_meds"] >= 5) & (f["n_meds"] < 8)) + 0.06 * f["ckd"]
    home = 0.04 + 0.20 * (f["age"] >= 75) + 0.10 * np.maximum(f["chf"], f["dementia"]) + 0.06 * (f["ip_1y"] > 0)
    engage = np.select([f["age"] < 65, f["age"] < 75], [0.6, 0.35], 0.1)
    digital = digital_boost * engage * (0.08 + 0.12 * np.maximum(f["diabetes"], f["obesity"]))
    out = np.column_stack([np.zeros(len(df)), nurse, med, home, digital])
    return np.clip(out, 0.0, 0.6)


def outcome(y0: np.ndarray, rrr_taken: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Observed outcome under the action taken: an event that would have happened is averted with prob RRR."""
    return (y0 * (rng.random(len(y0)) >= rrr_taken)).astype(int)


def expected_averted(y0: np.ndarray, r: np.ndarray, action_idx: np.ndarray) -> np.ndarray:
    return y0 * r[np.arange(len(y0)), action_idx]


def pilot(panel: pd.DataFrame, year: int, share: float, seed: int) -> pd.DataFrame:
    """Randomized pilot: a random share of the year's members get a uniformly random action."""
    rng = np.random.default_rng(seed)
    d = panel[panel["year"] == year]
    d = d.loc[rng.random(len(d)) < share].copy()
    a = rng.integers(0, len(ACTIONS), len(d))
    r = rrr(d)
    d["action"] = a
    d["y_obs"] = outcome(d["label"].to_numpy(), r[np.arange(len(d)), a], rng)
    return d
