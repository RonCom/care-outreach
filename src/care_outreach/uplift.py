"""Next-best-action recommender: which outreach program helps each member most.

Trained on the randomized pilot (uniform random assignment), so treatment and control are comparable.
Three meta-learners:
  S-learner   one gradient-boosting model of P(event | member, action) with the action as a feature
  T-learner   one gradient-boosting model per action
  DR-learner  doubly robust: cross-fitted S-learner outcome predictions plus an inverse-propensity correction give
              a pseudo-outcome per member and program, regressed on member features (Kennedy 2023)
  anchored    risk-anchored (anchored_constant: the same with one relative effect per program): programs act on the risk of an event, so E[event | x, a] = p0(x) * exp(g(x) + h_a(x)),
              with p0 from the deep-learning risk model (trained on years of history) as a fixed offset and
              log-linear g, h_a fitted on the pilot (Poisson GLM). Benefit = p0 e^g (1 - e^h_a). Borrows strength
              from the large risk model instead of learning baseline risk from the small pilot.
All learners also see the risk model's score as a feature (last column of X).
Predicted benefit of program a = P(event | none) - P(event | a). Model choice uses only pilot data: 5-fold
cross-validated, inverse-propensity-weighted (IPW) events per member under each learner's budgeted outreach plan,
which is unbiased because assignment was random. The known simulated effects are used afterwards, for evaluation only.
"""
from __future__ import annotations

import json
import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from . import data, simulate
from .config import Config

log = logging.getLogger(__name__)
A = len(simulate.ACTIONS)


def _hgb(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=seed)


class SLearner:
    def fit(self, X, a, y, seed=0):
        self.m = _hgb(seed).fit(np.column_stack([X, np.eye(A)[a]]), y)
        return self

    def event_prob(self, X):
        return np.column_stack([self.m.predict_proba(np.column_stack([X, np.tile(np.eye(A)[k], (len(X), 1))]))[:, 1]
                                for k in range(A)])


class TLearner:
    def fit(self, X, a, y, seed=0):
        self.m = [_hgb(seed).fit(X[a == k], y[a == k]) for k in range(A)]
        return self

    def event_prob(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.m])


class AnchoredLearner:
    interactions = True        # program effect varies with member features; False = one relative effect per program

    def fit(self, X, a, y, seed=0):
        p0 = np.clip(X[:, -1], 1e-3, 1)
        z = X[:, :-1]
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-6
        zs = (z - self.mu) / self.sd
        T = np.eye(A)[a][:, 1:]                                         # program indicators
        design = np.column_stack([zs] + [T[:, [k]] * self._h(zs) for k in range(A - 1)])
        # Poisson GLM with offset log p0  ==  target y / p0 with weight p0
        self.m = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(design, y / p0, sample_weight=p0)
        return self

    def benefit(self, X):
        p0 = np.clip(X[:, -1], 1e-3, 1)
        zs = (X[:, :-1] - self.mu) / self.sd
        d, coef = zs.shape[1], self.m.coef_
        base = p0 * np.exp(self.m.intercept_ + zs @ coef[:d])
        h = self._h(zs)
        w = h.shape[1]
        out = [np.zeros(len(X))]
        for k in range(A - 1):
            out.append(base * (1 - np.exp(h @ coef[d + k * w: d + (k + 1) * w])))
        return np.column_stack(out)

    def _h(self, zs):
        return np.column_stack([np.ones(len(zs)), zs]) if self.interactions else np.ones((len(zs), 1))


class AnchoredConstant(AnchoredLearner):
    interactions = False


class DRLearner:
    def fit(self, X, a, y, seed=0):
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, 2, len(y))
        mu = np.zeros((len(y), A))
        for f in (0, 1):                                  # cross-fitting: outcome model never sees its own rows
            mu[fold == f] = SLearner().fit(X[fold != f], a[fold != f], y[fold != f], seed).event_prob(X[fold == f])
        e = 1.0 / A                                       # known propensity (uniform randomization)
        corr = lambda k: (a == k) / e * (y - mu[:, k])
        self.m = [None] + [
            HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80,
                                          l2_regularization=1.0, random_state=seed)
            .fit(X, (mu[:, 0] + corr(0)) - (mu[:, k] + corr(k))) for k in range(1, A)]
        return self

    def benefit(self, X):
        return np.column_stack([np.zeros(len(X))] + [m.predict(X) for m in self.m[1:]])


LEARNERS = {"s_learner": SLearner, "t_learner": TLearner, "dr_learner": DRLearner, "anchored": AnchoredLearner,
            "anchored_constant": AnchoredConstant}
COLS = data.FEATURES + ["risk"]


def benefit(model, X) -> np.ndarray:
    """Predicted events averted per member and action (column 0 = none = 0)."""
    if hasattr(model, "benefit"):
        return model.benefit(X)
    p = model.event_prob(X)
    return p[:, [0]] - p


def best_action(b: np.ndarray, costs: np.ndarray, lam: float) -> np.ndarray:
    return np.argmax(b - lam * costs[None, :], axis=1)


def ipw_value(y: np.ndarray, a: np.ndarray, chosen: np.ndarray) -> float:
    """Events per member if everyone got `chosen`, from uniformly randomized data (propensity 1/A)."""
    return float(np.mean((a == chosen) * y * A))


def risk_scores(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("data_dir") / "risk_scores.parquet")[["pid", "year", "risk"]]


def fit_pilot(cfg: Config, panel: pd.DataFrame):
    pc = cfg["pilot"]
    p = simulate.pilot(panel, pc["year"], pc["share"], pc["seed"]).merge(risk_scores(cfg), on=["pid", "year"])
    X, a, y = p[COLS].to_numpy(np.float32), p["action"].to_numpy(), p["y_obs"].to_numpy()
    from .optimize import action_table, limits, plan_ilp  # local import: optimize imports this module
    cost, hours = action_table(cfg)
    folds = np.random.default_rng(pc["seed"]).integers(0, 5, len(p))
    sel = {}
    for name, cls in LEARNERS.items():
        plan_ev, argmax_ev = [], []
        for f in range(5):                                  # 5-fold cross-validation on the pilot
            tr, te = folds != f, folds == f
            m = cls().fit(X[tr], a[tr], y[tr], seed=pc["seed"])
            b = benefit(m, X[te])
            plan_ev.append(ipw_value(y[te], a[te], plan_ilp(b, cost, hours, limits(cfg, te.sum()))))
            argmax_ev.append(ipw_value(y[te], a[te], b[:, 1:].argmax(1) + 1))
        sel[name] = {"ipw_events_per_member_budgeted_plan": float(np.mean(plan_ev)),
                     "ipw_events_per_member_best_program_for_all": float(np.mean(argmax_ev))}
    sel["no_outreach"] = {"ipw_events_per_member_budgeted_plan": ipw_value(y, a, np.zeros(len(y), int))}
    best = min(LEARNERS, key=lambda k: sel[k]["ipw_events_per_member_budgeted_plan"])
    model = LEARNERS[best]().fit(X, a, y, seed=pc["seed"])      # refit the chosen learner on the whole pilot
    return model, best, sel, p


def evaluate_truth(model, df: pd.DataFrame) -> dict:
    """Compare recommendations with the simulated truth on the deployment year (evaluation only).
    Expected benefit is Y0 x RRR, so for any one member the best action is simply the one with the largest RRR."""
    X = df[COLS].to_numpy(np.float32)
    rec = benefit(model, X)[:, 1:].argmax(1) + 1          # best program if every member got one
    r = simulate.rrr(df)
    best = r[:, 1:].argmax(1) + 1
    y0 = df["label"].to_numpy()
    idx = np.arange(len(df))
    return {"top_action_agreement": float(np.mean(rec == best)),
            "share_of_oracle_benefit": float((y0 * r[idx, rec]).sum() / (y0 * r[idx, best]).sum()),
            "share_nurse_call_for_all": float((y0 * r[:, 1]).sum() / (y0 * r[idx, best]).sum())}


def run(cfg: Config) -> dict:
    panel, _ = data.load(cfg)
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("next-best-action")
    with mlflow.start_run(run_name="uplift"):
        model, best, sel, p = fit_pilot(cfg, panel)
        test = panel[panel["year"] == cfg["plan"]["year"]].merge(risk_scores(cfg), on=["pid", "year"])
        ev = evaluate_truth(model, test)
        mlflow.log_params({"learner": best, "pilot_year": cfg["pilot"]["year"], "pilot_members": len(p)})
        mlflow.log_metrics({f"{k}_{m}": v for k, d in sel.items() for m, v in d.items()} | ev)
    res = {"pilot_members": len(p), "pilot_event_rate": float(p["y_obs"].mean()), "selection": sel,
           "chosen": best, "deployment_year_check": ev}
    (cfg.reports / "uplift.json").write_text(json.dumps(res, indent=2))
    log.info("uplift: chose %s; %s", best, ev)
    return res | {"model": model, "pilot": p}
