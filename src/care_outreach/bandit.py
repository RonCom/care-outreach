"""Online learning of the outreach policy: a contextual bandit (one-step reinforcement learning).

Members arrive in monthly cohorts over 2023-2025 (36 months). Each month the policy picks a program per member,
observes whether an acute event followed, and updates. Reward = -(acute event) - cost_weight x program cost, so the
policy trades events averted against money.

Policies (all warm-started from the same randomized pilot):
  static      the pilot-trained recommender, frozen (argmax of predicted benefit - cost_weight x cost)
  ts          linear Thompson sampling, one Bayesian linear model per program
  ts_discount the same with exponential forgetting (gamma per month), so old evidence fades
  oracle      the true program effects with the model-estimated baseline risk (the benchmark for regret)

Protocol: the exploration scale was chosen from {1, 0.5, 0.25} on months 1-12 only; months 13-36 are the
evaluation window. Midway (`shift_month`) the digital program gets a new app version and its effect rises 2.5x. A frozen policy
cannot notice; a learning policy can. Value is scored with the simulated truth: expected events averted minus
cost_weight x spend.
"""
from __future__ import annotations

import json
import logging

import mlflow
import numpy as np
import pandas as pd

from . import data, simulate, uplift
from .config import Config

log = logging.getLogger(__name__)
A = len(simulate.ACTIONS)


class LinTS:
    def __init__(self, dim: int, prior_var: float, noise_var: float, gamma: float = 1.0, seed: int = 0,
                 scale: float = 1.0):
        self.P = np.stack([np.eye(dim) / prior_var] * A)
        self.b = np.zeros((A, dim))
        self.P0 = np.eye(dim) / prior_var
        self.noise, self.gamma, self.scale = noise_var, gamma, scale   # scale < 1 narrows exploration
        self.rng = np.random.default_rng(seed)

    def update(self, phi: np.ndarray, a: np.ndarray, r: np.ndarray) -> None:
        for k in range(A):
            m = a == k
            if m.any():
                self.P[k] += phi[m].T @ phi[m] / self.noise
                self.b[k] += phi[m].T @ r[m] / self.noise

    def decay(self) -> None:
        if self.gamma < 1:
            self.P = self.gamma * self.P + (1 - self.gamma) * self.P0
            self.b *= self.gamma

    def sample_rewards(self, phi: np.ndarray) -> np.ndarray:
        """One posterior draw per member and program (not one per batch, which would move whole cohorts at once)."""
        out = np.empty((len(phi), A))
        for k in range(A):
            cov = np.linalg.inv(self.P[k])
            L = np.linalg.cholesky((cov + cov.T) / 2)
            theta = (cov @ self.b[k])[None, :] + self.scale * self.rng.standard_normal((len(phi), len(cov))) @ L.T
            out[:, k] = np.einsum("ij,ij->i", phi, theta)
        return out


def run(cfg: Config) -> dict:
    c = cfg["bandit"]
    lam = c["cost_weight"]
    panel, _ = data.load(cfg)
    costs = np.array([cfg["actions"][k]["cost"] for k in simulate.ACTIONS], float)
    fit = uplift.run(cfg)
    static_model, pilot = fit["model"], fit["pilot"]
    scores = pd.read_parquet(cfg.path("data_dir") / "risk_scores.parquet")[["pid", "year", "risk"]]
    rng = np.random.default_rng(c["seed"])

    # context: intercept, baseline risk, and baseline risk x standardized features. Program effects act on the
    # risk of an event, so interactions with risk carry the signal; scaling is fitted on the pilot.
    mu = pilot[data.FEATURES].mean().to_numpy()
    sd = pilot[data.FEATURES].std().to_numpy() + 1e-6

    def phi(df):
        p0 = df["risk"].to_numpy()[:, None]
        return np.column_stack([np.ones(len(df)), p0, p0 * (df[data.FEATURES].to_numpy() - mu) / sd])

    # monthly cohorts
    d = panel[panel["year"].isin(c["years"])].merge(scores, on=["pid", "year"]).reset_index(drop=True)
    d["month"] = (d["year"] - min(c["years"])) * 12 + rng.integers(0, 12, len(d))
    months = sorted(d["month"].unique())

    k = len(mu) + 2
    policies = {"ts": LinTS(k, c["prior_var"], c["noise_var"], 1.0, c["seed"], c["explore_scale"]),
                "ts_discount": LinTS(k, c["prior_var"], c["noise_var"], c["discount"], c["seed"], c["explore_scale"])}
    pa, pr = pilot["action"].to_numpy(), -pilot["y_obs"].to_numpy(float)
    for p in policies.values():
        p.update(phi(pilot), pa, pr)

    names = ["static", "ts", "ts_discount", "oracle"]
    rows = []
    for m in months:
        g = d[d["month"] == m]
        boost = 2.5 if m >= c["shift_month"] else 1.0
        r_true = simulate.rrr(g, digital_boost=boost)
        y0 = g["label"].to_numpy()
        f = phi(g)
        choice = {
            "static": uplift.best_action(uplift.benefit(static_model, g[uplift.COLS].to_numpy(np.float32)), costs, lam),
            "oracle": np.argmax(g["risk"].to_numpy()[:, None] * r_true - lam * costs[None, :], axis=1),
        }
        for k, p in policies.items():
            choice[k] = np.argmax(p.sample_rewards(f) - lam * costs[None, :], axis=1)
        for k in names:
            a = choice[k]
            value = simulate.expected_averted(y0, r_true, a) - lam * costs[a]
            rows.append({"month": int(m), "policy": k, "members": len(g), "value": float(value.sum()),
                         "averted": float(simulate.expected_averted(y0, r_true, a).sum()),
                         "spend": float(costs[a].sum()), "digital_share": float(np.mean(a == 4)),
                         "treated_share": float(np.mean(a > 0))})
            if k in policies:
                y_obs = simulate.outcome(y0, r_true[np.arange(len(g)), a], rng)
                policies[k].update(f, a, -y_obs.astype(float))
        for p in policies.values():
            p.decay()

    t = pd.DataFrame(rows)
    t["cum_value"] = t.groupby("policy")["value"].cumsum()
    oracle = t[t.policy == "oracle"].set_index("month")["cum_value"]
    t["cum_regret"] = t["month"].map(oracle) - t["cum_value"]
    t.to_csv(cfg.reports / "bandit_monthly.csv", index=False)
    pre, post = t["month"] < c["shift_month"], t["month"] >= c["shift_month"]
    summ = {}
    for k in names:
        s = t[t.policy == k]
        summ[k] = {"total_value": float(s["value"].sum()), "events_averted": float(s["averted"].sum()),
                   "spend": float(s["spend"].sum()), "final_regret": float(s["cum_regret"].iloc[-1]),
                   "value_months_1_12": float(s.loc[s["month"] < 12, "value"].sum()),
                   "value_months_13_36": float(s.loc[s["month"] >= 12, "value"].sum()),
                   "averted_months_13_36": float(s.loc[s["month"] >= 12, "averted"].sum()),
                   "spend_months_13_36": float(s.loc[s["month"] >= 12, "spend"].sum()),
                   "value_before_shift": float(s.loc[pre[s.index], "value"].sum()),
                   "value_after_shift": float(s.loc[post[s.index], "value"].sum()),
                   "digital_share_before": float(s.loc[pre[s.index], "digital_share"].mean()),
                   "digital_share_after": float(s.loc[post[s.index], "digital_share"].mean())}
    res = {"months": len(months), "member_months": len(d), "cost_weight": lam, "shift_month": c["shift_month"],
           "policies": summ}
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("outreach-bandit")
    with mlflow.start_run(run_name="bandit"):
        mlflow.log_params({k: c[k] for k in ("cost_weight", "prior_var", "noise_var", "shift_month")})
        mlflow.log_metrics({f"{k}_{m}": v for k, s in summ.items() for m, v in s.items()})
        for _, r in t[t.policy != "oracle"].iterrows():
            mlflow.log_metric(f"cum_regret_{r.policy}", r.cum_regret, step=int(r.month))
    (cfg.reports / "bandit.json").write_text(json.dumps(res, indent=2))
    _chart(t, c["shift_month"], cfg)
    log.info("bandit: %s", {k: round(v["total_value"], 1) for k, v in summ.items()})
    return res


def _chart(t: pd.DataFrame, shift: int, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    from .charts import COLORS, save, style
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = {"static": "Frozen recommender", "ts": "Thompson sampling",
              "ts_discount": "Thompson sampling, discounted"}
    for k, lab in labels.items():
        s = t[t.policy == k]
        axes[0].plot(s["month"] + 1, s["cum_regret"], color=COLORS[k], label=lab)
        axes[1].plot(s["month"] + 1, 100 * s["digital_share"], color=COLORS[k], label=lab)
    s = t[t.policy == "oracle"]
    axes[1].plot(s["month"] + 1, 100 * s["digital_share"], color=COLORS["oracle"], ls="--", label="Oracle")
    for ax in axes:
        ax.axvline(shift + 1, color=COLORS["grid_dark"], lw=1)
        ax.set_xlabel("Month (Jan 2023 = 1)")
    axes[0].set_title("Cumulative regret vs oracle")
    axes[0].set_ylabel("Events-equivalent")
    axes[1].set_title("Members sent to the digital program")
    axes[1].set_ylabel("% of monthly cohort")
    axes[0].legend(frameon=False, fontsize=8)
    lo = axes[1].get_ylim()[0]
    axes[1].text(shift + 1.5, lo + 2, "new app version", fontsize=8, color=COLORS["ink2"])
    save(fig, cfg, "bandit_regret.png")
