"""Self-updating risk model: Evidently drift monitoring, MLflow champion/challenger retraining.

Scenario: from Jan 2023 a new claims clearinghouse drops the place-of-service detail on 70% of facility claims, so
emergency visits and inpatient stays arrive as generic ambulatory encounters (a feed change of the kind that happens
in practice). Outcome labels come from adjudicated claims and are unaffected, but the model's strongest inputs
degrade. Two models then score each year 2022-2025:

  static    the champion trained on 2015-2020, never updated
  adaptive  each year, before scoring:
              1. Evidently data-drift report: this year's features vs the current model's training data
              2. performance check on the most recent labeled year (AUC vs the model's validation AUC)
              3. trigger if a key utilization feature drifts, the drifted-feature share reaches drift_share, or AUC
                 fell by auc_drop. The first year a trigger fires is recorded as a change point.
              4. retrain a challenger (the deep model is fine-tuned from the champion's weights) on labeled data
                 since the change point (older data no longer describes the
                 inputs); if no labeled year exists since then, keep the champion and wait. 70/30 split of the
                 latest labeled year for early stopping and comparison. The challenger is logged to MLflow and
                 registered, and the `champion` alias moves only if it beats the current champion on the holdout.
"""
from __future__ import annotations

import json
import logging

import mlflow
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import roc_auc_score

from . import data, risk
from .config import Config

log = logging.getLogger(__name__)
MONITORED = [f for f in data.FEATURES if f not in ("male",)]


def feed_change(panel: pd.DataFrame, seq: np.ndarray, from_year: int = 2023, share: float = 0.7, seed: int = 3):
    """From `from_year`, record `share` of ED visits and inpatient stays as ambulatory encounters."""
    rng = np.random.default_rng(seed)
    p, s = panel.copy(), seq.copy()
    hit = (p["year"] - 1 >= from_year).to_numpy()        # prior (feature) year affected
    for col in ("ed_1y", "ip_1y"):
        moved = rng.binomial(p.loc[hit, col].to_numpy(), share)
        p.loc[hit, col] -= moved
        p.loc[hit, "amb_1y"] += moved
    hit2 = (p["year"] - 2 >= from_year).to_numpy()       # acute_2y counts ED + inpatient two years back
    p.loc[hit2, "acute_2y"] -= rng.binomial(p.loc[hit2, "acute_2y"].to_numpy(), share)
    # sequence months: a row covers Jan(Y-2)..Dec(Y-1); month j falls in calendar year Y-2 + j // 12
    amb = data.CHANNELS.index("amb")
    cal = (p["year"].to_numpy()[:, None] - 2) + np.arange(24)[None, :] // 12
    mask = cal >= from_year
    for ch in ("ed", "ip"):
        k = data.CHANNELS.index(ch)
        m = rng.binomial(s[:, :, k].astype(int), share) * mask
        s[:, :, k] -= m
        s[:, :, amb] += m
    return p, s


def drift_share(ref: pd.DataFrame, cur: pd.DataFrame, html=None) -> tuple[float, list[str]]:
    snap = Report([DataDriftPreset()]).run(current_data=cur[MONITORED], reference_data=ref[MONITORED])
    if html:
        snap.save_html(str(html))
    d = snap.dict()["metrics"]
    share = next(m["value"]["share"] for m in d if m["metric_name"].startswith("DriftedColumnsCount"))
    drifted = []
    for m in d:
        cfg = m["config"]
        if cfg.get("type", "").endswith("ValueDrift"):
            method, thr, v = cfg.get("method", ""), cfg.get("threshold"), m["value"]
            is_drift = v < thr if "p_value" in method else v >= thr
            if is_drift:
                drifted.append(cfg["column"])
    return float(share), drifted


def run(cfg: Config) -> dict:
    mc, rc = cfg["monitor"], cfg["risk"]
    panel0, seq0 = data.load(cfg)
    panel, seq = feed_change(panel0, seq0)
    family = json.loads((cfg.reports / "risk_models.json").read_text())["champion"]
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("acute-risk-monitoring")
    client = mlflow.MlflowClient()
    name = "acute-risk-adaptive"

    def auc(model, mask):
        X, S = panel.loc[mask, data.FEATURES].to_numpy(np.float32), seq[mask]
        return roc_auc_score(panel.loc[mask, "label"], model.predict(X, S))

    with mlflow.start_run(run_name=f"initial-{family}"):
        static = risk.fit_family(family, cfg, panel, seq, mc["reference_years"], rc["valid_years"])
        valid_auc = auc(static, panel["year"].isin(rc["valid_years"]).to_numpy())
        mlflow.log_metric("valid_auc", valid_auc)
        info = risk.log_model(static, family, panel[data.FEATURES].to_numpy(np.float32)[:5])
    v = mlflow.register_model(info.model_uri, name)
    client.set_registered_model_alias(name, "champion", v.version)

    champ, champ_ref, champ_valid = static, panel["year"].isin(mc["reference_years"]), valid_auc
    rows, rng, change_point = [], np.random.default_rng(rc["seed"]), None
    for y in mc["score_years"]:
        cur = (panel["year"] == y).to_numpy()
        last = (panel["year"] == y - 1).to_numpy()
        html = cfg.reports / f"drift_{y}.html"
        dshare, drifted = drift_share(panel[champ_ref], panel[cur], html)
        auc_last = auc(champ, last)
        key = sorted(set(drifted) & set(mc["key_features"]))
        trigger = bool(key) or dshare >= mc["drift_share"] or (champ_valid - auc_last) >= mc["auc_drop"]
        if trigger and change_point is None:
            change_point = y
        promoted, retrained = False, False
        since = (panel["year"] >= (change_point or 0)).to_numpy()
        if trigger and (since & last).any():
            retrained = True
            hold = last & (rng.random(len(panel)) < 0.3)
            train = (panel["year"] < y).to_numpy() & since & ~hold
            with mlflow.start_run(run_name=f"challenger-{y}"):
                X, S, yy = panel.loc[train, data.FEATURES].to_numpy(np.float32), seq[train], panel.loc[train, "label"]
                Xh, Sh, yh = panel.loc[hold, data.FEATURES].to_numpy(np.float32), seq[hold], panel.loc[hold, "label"]
                if isinstance(champ, risk.Deep):     # deep model: fine-tune the champion on post-change data
                    chal = risk.Deep(cfg).fine_tune(champ, X, S, yy.to_numpy(), Xh, Sh, yh.to_numpy())
                else:
                    chal = risk.FAMILIES[family](cfg).fit(X, S, yy.to_numpy(), Xh, Sh, yh.to_numpy())
                a_chal, a_champ = roc_auc_score(yh, chal.predict(Xh, Sh)), roc_auc_score(yh, champ.predict(Xh, Sh))
                mlflow.log_params({"score_year": y, "trigger_drift_share": round(dshare, 3), "train_from": change_point,
                                   "drifted": ",".join(drifted[:10])})
                mlflow.log_metrics({"holdout_auc_challenger": a_chal, "holdout_auc_champion": a_champ})
                info = risk.log_model(chal, family, X[:5])
            v = mlflow.register_model(info.model_uri, name)
            client.set_model_version_tag(name, v.version, "trained_through", str(y - 1))
            if a_chal > a_champ:
                client.set_registered_model_alias(name, "champion", v.version)
                champ, champ_ref, champ_valid, promoted = chal, pd.Series(train, index=panel.index), a_chal, True
        diff = None
        if champ is not static:                           # paired bootstrap: adaptive minus static AUC this year
            X, S, yy = panel.loc[cur, data.FEATURES].to_numpy(np.float32), seq[cur], panel.loc[cur, "label"].to_numpy()
            pa, ps = champ.predict(X, S), static.predict(X, S)
            b = np.random.default_rng(0)
            draws = []
            for _ in range(1000):
                i = b.integers(0, len(yy), len(yy))
                draws.append(roc_auc_score(yy[i], pa[i]) - roc_auc_score(yy[i], ps[i]))
            diff = {"est": roc_auc_score(yy, pa) - roc_auc_score(yy, ps),
                    "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}
        rows.append({"year": y, "adaptive_minus_static_auc": diff, "drift_share": dshare, "drifted": drifted, "key_features_drifted": key,
                     "auc_last_labeled_year": auc_last, "triggered": trigger, "change_point": change_point,
                     "retrained": retrained, "promoted": promoted,
                     "auc_static": auc(static, cur), "auc_adaptive": auc(champ, cur),
                     "auc_static_no_feed_change": roc_auc_score(
                         panel0.loc[cur, "label"],
                         static.predict(panel0.loc[cur, data.FEATURES].to_numpy(np.float32), seq0[cur]))})
        log.info("monitor %s: drift %.2f key %s retrain %s promote %s | AUC static %.3f adaptive %.3f", y, dshare, key,
                 retrained,
                 promoted, rows[-1]["auc_static"], rows[-1]["auc_adaptive"])
    res = {"family": family, "scenario": "from 2023, 70% of ED visits and inpatient stays arrive as ambulatory",
           "years": rows}
    (cfg.reports / "monitoring.json").write_text(json.dumps(res, indent=2, default=float))
    _chart(rows, cfg)
    return res


def _chart(rows, cfg):
    import matplotlib.pyplot as plt

    from .charts import COLORS, save, style
    style()
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(t.year, t.auc_static_no_feed_change, color=COLORS["grid_dark"], marker="o", ls="--",
            label="Static, if the feed had not changed")
    ax.plot(t.year, t.auc_static, color=COLORS["static"], marker="o", label="Static model")
    ax.plot(t.year, t.auc_adaptive, color=COLORS["adaptive"], marker="o", label="Drift-triggered retraining")
    for _, r in t[t.promoted].iterrows():
        ax.annotate("retrained", (r.year, r.auc_adaptive), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=COLORS["adaptive"])
    ax.set_xticks(t.year)
    ax.set_title("Risk model AUC after a claims-feed change")
    ax.set_ylabel("AUC (acute event in year)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, cfg, "monitoring_auc.png")
