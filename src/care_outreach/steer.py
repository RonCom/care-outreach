"""Provider-network steering: recommend a high-value physical therapist to members who need one.

Real providers, synthetic members:
  providers  every North Carolina physical therapist in the CMS Physician & Other Practitioners file (2022-2024):
             location (ZIP centroid), Medicare patients, and risk-adjusted standardized payment per patient (the
             episode cost), plus the Medicare FWA project's outlier score. Providers in the top 5% of that score are
             kept out of the steering network (a payment-integrity guardrail).
  members    Synthea members (home coordinates) with a new condition that typically needs physical therapy
             (fracture, sprain, ligament / tendon / meniscus injury, whiplash, knee or hip osteoarthritis...).

Without steering a member picks a provider by a gravity model (larger, nearer practices are likelier). The plan can
recommend one provider; the member accepts with a probability that falls with the extra distance. If they decline
they fall back to their usual choice. Expected savings of recommending j to member i:
    accept_ij x (expected default cost_i - cost_j)
Acceptance is simulated (no public data records it). Its true form has things the plan's starting model lacks:
larger practices are more accepted (easier scheduling), and members far from any provider mind distance less.

Each provider can take a limited number of steered members a year (1.5 x the year's episodes in total, apportioned by
Medicare volume; most providers get 0 or 1 slot, large practices several) and slots are paced through the year
(by the end of month k at most ceil(slots x k / 12) are used), so the
recommendation problem is a capacitated assignment, solved as a linear program (HiGHS; bipartite b-matching, so the
LP optimum is integral):
    maximize   sum_ij accept_ij (default_i - cost_j) x_ij
    subject to sum_j x_ij <= 1 per member,  sum_i x_ij <= remaining capacity_j,  0 <= x <= 1

Monthly cohorts of members arrive over 2023-2025 (each year uses the previous year's provider costs). Policies:
  none      no steering
  cheapest  recommend the cheapest in-network provider within the radius, first come first served (no acceptance
            model, no capacity planning beyond first come)
  lp_prior  the LP with the plan's starting acceptance model (distance only), never updated
  lp_learn  the LP with a Bayesian logistic acceptance model updated monthly from accept / decline outcomes and
            acting on a posterior draw per member-provider pair (Thompson sampling)
  oracle    the LP with the true acceptance probabilities
"""
from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from .config import Config

log = logging.getLogger(__name__)

PT_TRIGGERS = ["fracture of", "fracture subluxation", "sprain", "whiplash", "injury of neck", "injury of knee",
               "rotator cuff", "collateral ligament", "cruciate ligament", "meniscus", "patellar tendon",
               "osteoarthritis of knee", "osteoarthritis of hip", "chronic pain", "fracture of bone"]
PT_EXCLUDE = ["dental", "mandible", "rib", "skull"]
EARTH_KM = 6371.0


# ---- data ---------------------------------------------------------------------------------------------------

def extract_providers(fwa_dir: Path, out: Path, state: str = "NC") -> None:
    """From a local MedicareFWA checkout: this state's physical therapists (raw CMS file) and their FWA scores."""
    cols = ["Rndrng_NPI", "Rndrng_Prvdr_Zip5", "Rndrng_Prvdr_RUCA", "Rndrng_Prvdr_State_Abrvtn", "Tot_Benes",
            "Tot_Mdcr_Stdzd_Amt", "Bene_Avg_Risk_Scre"]
    rows = []
    for y in (2022, 2023, 2024):
        for f in glob.glob(str(fwa_dir / f"data/raw/physician_provider/year={y}/physical_therapist*.parquet")):
            d = pd.read_parquet(f, columns=cols)
            rows.append(d[d["Rndrng_Prvdr_State_Abrvtn"] == state].assign(year=y))
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_parquet(out / "nc_pt_providers.parquet", index=False)
    s = pd.read_parquet(fwa_dir / "data/out/provider_scores_labelled.parquet",
                        columns=["year", "npi", "specialty", "state", "risk_pct", "pymt_per_bene_risk_adj",
                                 "srvcs_per_bene"])
    s = s[(s["state"] == state) & (s["year"] >= 2022) & s["specialty"].str.startswith("Physical")]
    s.to_parquet(out / "nc_pt_scores.parquet", index=False)


def zip_coords(zips) -> pd.DataFrame:
    import zipcodes
    rows = []
    for z in pd.unique(zips):
        m = zipcodes.matching(str(z))
        if m:
            rows.append({"zip": z, "lat": float(m[0]["lat"]), "lon": float(m[0]["long"])})
    return pd.DataFrame(rows)


def providers(cfg: Config) -> pd.DataFrame:
    ext = cfg.path("steer_data")
    p = pd.read_parquet(ext / "nc_pt_providers.parquet")
    s = pd.read_parquet(ext / "nc_pt_scores.parquet")
    p = p.assign(npi=p["Rndrng_NPI"].astype(str), zip=p["Rndrng_Prvdr_Zip5"].astype(str),
                 benes=pd.to_numeric(p["Tot_Benes"]).astype(float),
                 ruca=pd.to_numeric(p["Rndrng_Prvdr_RUCA"], errors="coerce"))
    s = s.assign(npi=s["npi"].astype(str))
    d = p.merge(s[["npi", "year", "risk_pct", "pymt_per_bene_risk_adj"]], on=["npi", "year"], how="inner")
    d = d.merge(zip_coords(d["zip"]), on="zip", how="inner").rename(columns={"pymt_per_bene_risk_adj": "cost"})
    return d[["npi", "year", "zip", "lat", "lon", "benes", "ruca", "cost", "risk_pct"]].dropna(subset=["cost"])


def episodes(cfg: Config, years) -> pd.DataFrame:
    src = cfg.path("synthea_csv")
    trig = " OR ".join(f"lower(DESCRIPTION) LIKE '%{t}%'" for t in PT_TRIGGERS)
    excl = " OR ".join(f"lower(DESCRIPTION) LIKE '%{t}%'" for t in PT_EXCLUDE)
    q = f"""
        WITH c AS (
          SELECT PATIENT AS pid, START::DATE AS start FROM read_csv_auto('{src}/conditions.csv', header=true,
                 all_varchar=true, quote='"')
          WHERE ({trig}) AND NOT ({excl}) AND year(START::DATE) BETWEEN {min(years)} AND {max(years)})
        , e AS (SELECT pid, min(start) AS start FROM c GROUP BY pid, year(start))
        SELECT e.pid, e.start, p.LAT::DOUBLE AS lat, p.LON::DOUBLE AS lon,
               date_diff('year', p.BIRTHDATE::DATE, e.start) AS age
        FROM e JOIN read_csv_auto('{src}/patients.csv', header=true, all_varchar=true, quote='"') p ON p.Id = e.pid
    """
    e = duckdb.sql(q).df()
    e["start"] = pd.to_datetime(e["start"])
    e = e[e["age"] >= cfg["min_age"]].copy()
    e["year"] = e["start"].dt.year
    e["month"] = (e["year"] - min(years)) * 12 + e["start"].dt.month - 1
    return e.sort_values("start").reset_index(drop=True)


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


# ---- member choice (simulation) -----------------------------------------------------------------------------

def default_choice(dist: np.ndarray, benes: np.ndarray, decay_km: float) -> np.ndarray:
    """Gravity model: P(member picks j) proportional to volume x exp(-distance / decay)."""
    w = benes[None, :] * np.exp(-dist / decay_km)
    return w / w.sum(1, keepdims=True)


def pair_features(extra_km, size_z, remote) -> np.ndarray:
    """Features of a member-provider pair seen by the learning policy (all observable to a real plan)."""
    return np.stack([np.ones_like(extra_km), extra_km / 10, size_z, remote, remote * extra_km / 10], axis=-1)


TRUE_W = np.array([1.0, -2.0, 0.5, 0.0, 1.2])     # remote members mind distance less; bigger practices accepted more
PRIOR_W = np.array([1.0, -1.2, 0.0, 0.0, 0.0])    # the plan's starting belief: distance only


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---- planning -----------------------------------------------------------------------------------------------

def plan_lp(value: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """value: members x providers (<= 0 means not allowed). Returns chosen provider index per member, or -1."""
    ii, jj = np.nonzero(value > 0)
    n, m = value.shape
    if len(ii) == 0:
        return np.full(n, -1)
    k = len(ii)
    A = sparse.vstack([sparse.csr_matrix((np.ones(k), (ii, np.arange(k))), shape=(n, k)),
                       sparse.csr_matrix((np.ones(k), (jj, np.arange(k))), shape=(m, k))])
    res = linprog(-value[ii, jj], A_ub=A, b_ub=np.r_[np.ones(n), cap], bounds=(0, 1), method="highs")
    out = np.full(n, -1)
    pick = res.x > 0.5
    out[ii[pick]] = jj[pick]
    return out


class BayesLogit:
    def __init__(self, m0: np.ndarray, prior_var: float):
        self.m, self.H = m0.astype(float).copy(), np.eye(len(m0)) / prior_var

    def update(self, X, y, iters=25):
        m0, H0, w = self.m.copy(), self.H.copy(), self.m.copy()
        for _ in range(iters):
            p = sigmoid(X @ w)
            H = H0 + (X * (p * (1 - p))[:, None]).T @ X
            step = np.linalg.solve(H, X.T @ (y - p) - H0 @ (w - m0))
            w = w + step
            if np.abs(step).max() < 1e-6:
                break
        p = sigmoid(X @ w)
        self.m, self.H = w, H0 + (X * (p * (1 - p))[:, None]).T @ X

    def sample(self, rng, shape) -> np.ndarray:
        cov = np.linalg.inv(self.H)
        L = np.linalg.cholesky((cov + cov.T) / 2)
        return self.m + rng.standard_normal(shape + (len(self.m),)) @ L.T


def run_seed(cfg: Config, prov: pd.DataFrame, eps: pd.DataFrame, seed: int) -> pd.DataFrame:
    c = cfg["steer"]
    rng = np.random.default_rng(seed)
    learner = BayesLogit(PRIOR_W, c["prior_var"])
    rows = []
    cap_left = {}
    for month in sorted(eps["month"].unique()):
        g = eps[eps["month"] == month]
        year = int(g["year"].iloc[0])
        pr = prov[prov["year"] == year - 1].reset_index(drop=True)      # last year's costs are what the plan knows
        if (year, "init") not in cap_left:
            annual = eps[eps["year"] == year].shape[0]
            quota = c["capacity_ratio"] * annual * pr["benes"].to_numpy() / pr["benes"].sum()
            base = np.floor(quota)                              # largest-remainder apportionment of the slots
            extra_slots = round(c["capacity_ratio"] * annual - base.sum())
            base[np.argsort(-(quota - base), kind="stable")[:extra_slots]] += 1
            cap_left[(year, "init")] = base
            for p in ("cheapest", "lp_prior", "lp_learn", "oracle"):
                cap_left[(year, p)] = np.zeros_like(base)          # slots used so far this year
        # pacing: by the end of month k of the year, at most ceil(annual slots x k / 12) may be used
        paced = np.ceil(cap_left[(year, "init")] * (int(month) % 12 + 1) / 12)
        avail = {p: paced - cap_left[(year, p)] for p in ("cheapest", "lp_prior", "lp_learn", "oracle")}
        dist = haversine(g["lat"].to_numpy()[:, None], g["lon"].to_numpy()[:, None],
                         pr["lat"].to_numpy()[None, :], pr["lon"].to_numpy()[None, :])
        cost = pr["cost"].to_numpy()
        benes = pr["benes"].to_numpy()
        pdef = default_choice(dist, benes, c["decay_km"])
        default_cost = pdef @ cost
        nearest = dist.min(1)
        extra = np.clip(dist - nearest[:, None], 0, None)
        size_z = (np.log(benes) - np.log(benes).mean()) / np.log(benes).std()
        remote = (nearest > c["remote_km"]).astype(float)
        F = pair_features(extra, np.broadcast_to(size_z, dist.shape), np.broadcast_to(remote[:, None], dist.shape))
        a_true = sigmoid(F @ TRUE_W)
        saving = default_cost[:, None] - cost[None, :]
        allowed = (dist <= c["radius_km"]) & (pr["risk_pct"].to_numpy() < c["fwa_cutoff"])[None, :] & (saving > 0)

        choice = {}
        # cheapest in radius, first come first served
        order_cost = np.where(allowed, cost[None, :], np.inf)
        ch = np.full(len(g), -1)
        cl = avail["cheapest"]
        for i in range(len(g)):
            for j in np.argsort(order_cost[i]):
                if not np.isfinite(order_cost[i, j]):
                    break
                if cl[j] >= 1:
                    ch[i] = j
                    cl[j] -= 1
                    cap_left[(year, "cheapest")][j] += 1
                    break
        choice["cheapest"] = ch
        for p, acc in (("lp_prior", sigmoid(F @ PRIOR_W)),
                       ("lp_learn", sigmoid(np.einsum("ijk,ik->ij", F, learner.sample(rng, (len(g),))))),
                       ("oracle", a_true)):
            v = np.where(allowed, acc * saving, 0)
            ch = plan_lp(v, avail[p])
            np.add.at(cap_left[(year, p)], ch[ch >= 0], 1)
            choice[p] = ch
        choice["none"] = np.full(len(g), -1)

        idx = np.arange(len(g))
        for p, ch in choice.items():
            rec = ch >= 0
            exp_saving = np.where(rec, a_true[idx, np.maximum(ch, 0)] * saving[idx, np.maximum(ch, 0)], 0.0)
            accepted = rec & (rng.random(len(g)) < a_true[idx, np.maximum(ch, 0)])
            rows.append({"seed": seed, "month": int(month), "year": year, "policy": p, "members": len(g),
                         "recommended": int(rec.sum()), "accepted": int(accepted.sum()),
                         "expected_saving": float(exp_saving.sum()), "default_cost": float(default_cost.sum()),
                         "extra_km_recommended": float(extra[idx[rec], ch[rec]].mean()) if rec.any() else 0.0})
            if p == "lp_learn" and rec.any():
                learner.update(F[idx[rec], ch[rec]], accepted[rec].astype(float))
    return pd.DataFrame(rows)


def run(cfg: Config) -> dict:
    c = cfg["steer"]
    prov = providers(cfg)
    eps = episodes(cfg, c["years"])
    t = pd.concat([run_seed(cfg, prov, eps, s) for s in c["seeds"]], ignore_index=True)
    t.to_csv(cfg.reports / "steer_monthly.csv", index=False)
    tot = t.groupby(["seed", "policy"])[["expected_saving", "recommended", "accepted", "default_cost"]].sum()
    mean = tot.groupby("policy").mean()
    by_seed = tot["expected_saving"].unstack()
    later = t["year"] > min(c["years"])
    by_seed_later = t[later].groupby(["seed", "policy"])["expected_saving"].sum().unstack()
    pr_cur = prov[prov["year"] == max(c["years"]) - 1]
    res = {
        "providers_by_year": prov.groupby("year").size().to_dict(),
        "excluded_by_fwa_score": int((pr_cur["risk_pct"] >= c["fwa_cutoff"]).sum()),
        "episodes": len(eps), "episodes_by_year": eps.groupby("year").size().to_dict(),
        "cost_per_patient_quartiles": [float(x) for x in pr_cur["cost"].quantile([0.25, 0.5, 0.75])],
        "totals_mean": {p: {k: float(v) for k, v in r.items()} for p, r in mean.iterrows()},
        "saving_per_episode": {p: float(mean.loc[p, "expected_saving"] / len(eps)) for p in mean.index},
        "saving_pct_of_default_cost": {p: float(mean.loc[p, "expected_saving"] / mean.loc[p, "default_cost"])
                                       for p in mean.index},
        "lp_learn_vs_lp_prior_pct_by_seed": (by_seed["lp_learn"] / by_seed["lp_prior"] - 1).round(4).to_dict(),
        "lp_learn_vs_lp_prior_pct_after_year1_by_seed":
            (by_seed_later["lp_learn"] / by_seed_later["lp_prior"] - 1).round(4).to_dict(),
        "lp_prior_vs_cheapest_pct_by_seed": (by_seed["lp_prior"] / by_seed["cheapest"] - 1).round(4).to_dict(),
        "share_of_oracle": {p: float(mean.loc[p, "expected_saving"] / mean.loc["oracle", "expected_saving"])
                            for p in mean.index},
    }
    (cfg.reports / "steering.json").write_text(json.dumps(res, indent=2, default=float))
    _chart(t, cfg)
    log.info("steering: %s", {p: round(v, 0) for p, v in res["saving_per_episode"].items()})
    return res


def _chart(t: pd.DataFrame, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    from .charts import COLORS, save, style
    style()
    m = t.groupby(["policy", "month"])[["expected_saving", "members"]].sum().reset_index()
    m["per"] = m["expected_saving"] / m["members"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lab = {"cheapest": ("Cheapest nearby, first come", COLORS["hgb"], "--"),
           "lp_prior": ("LP, fixed acceptance model", COLORS["static"], "-"),
           "lp_learn": ("LP, acceptance learned monthly", COLORS["adaptive"], "-"),
           "oracle": ("LP, true acceptance", COLORS["oracle"], ":")}
    for p, (l, col, ls) in lab.items():
        s = m[m.policy == p].sort_values("month")
        n = t["seed"].nunique()
        axes[0].plot(s["month"] + 1, s["expected_saving"].cumsum() / n / 1000, color=col, ls=ls, label=l)
    axes[0].set_title("Cumulative expected saving")
    axes[0].set_ylabel("$ thousands (standardized Medicare)")
    axes[0].set_xlabel("Month (Jan 2023 = 1)")
    axes[0].legend(frameon=False, fontsize=8)
    tot = t.groupby(["policy", "seed"])["expected_saving"].sum().groupby("policy").mean()
    keys = list(lab)
    axes[1].bar(range(len(keys)), [tot[k] / 1000 for k in keys], color=[lab[k][1] for k in keys])
    axes[1].set_xticks(range(len(keys)), ["Cheapest\nnearby", "LP, fixed\nmodel", "LP,\nlearned", "LP, true\n(ceiling)"],
                       fontsize=8)
    axes[1].set_title("Total expected saving, 2023-2025")
    axes[1].set_ylabel("$ thousands")
    save(fig, cfg, "steering.png")
