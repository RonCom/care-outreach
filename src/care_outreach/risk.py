"""Acute-event risk model: predict any ED visit or inpatient stay in the outcome year.

Three model families are benchmarked on an out-of-time split (train 2015-20, validate 2021, test 2023):
  logistic   - standardized tabular features, L2 logistic regression
  hgb        - histogram gradient boosting on the same features
  seqnet     - deep learning: a GRU over 24 months of utilization (ED, inpatient, urgent, ambulatory, cost)
               fused with an MLP over the tabular features (PyTorch), early-stopped on validation AUC
Every run is tracked in MLflow; the best model on validation AUC is registered as `acute-risk` with alias
`champion`, and its test-year scores feed the outreach plan.
"""
from __future__ import annotations

import json
import logging

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import data
from .config import Config

log = logging.getLogger(__name__)
MODEL_NAME = "acute-risk"


def metrics(y: np.ndarray, p: np.ndarray, prefix: str) -> dict:
    top = p >= np.quantile(p, 0.9)
    return {f"{prefix}_auc": roc_auc_score(y, p), f"{prefix}_ap": average_precision_score(y, p),
            f"{prefix}_brier": brier_score_loss(y, p), f"{prefix}_top10_lift": y[top].mean() / y.mean()}


# ---- model families: fit(X, seq, y, Xv, seqv, yv, cfg) -> object with predict(X, seq) -----------------------

class Tabular:
    def __init__(self, est):
        self.est = est

    def fit(self, X, seq, y, *_):
        self.est.fit(X, y)
        return self

    def predict(self, X, seq=None):
        return self.est.predict_proba(X)[:, 1]


def logistic(cfg: Config) -> Tabular:
    return Tabular(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)))


def hgb(cfg: Config) -> Tabular:
    return Tabular(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, early_stopping=True,
                                                  random_state=cfg["risk"]["seed"]))


class SeqNet(nn.Module):
    def __init__(self, n_tab: int, n_ch: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.tab = nn.Sequential(nn.Linear(n_tab, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, tab, seq):
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], self.tab(tab)], dim=1)).squeeze(1)


class Scored(nn.Module):
    """The trained net plus its input standardization, so the logged model takes raw features."""

    def __init__(self, deep: Deep):
        super().__init__()
        self.net = deep.net
        for k in ("mu", "sd", "smu", "ssd"):
            self.register_buffer(k, torch.tensor(getattr(deep, k), dtype=torch.float32))

    def forward(self, tab, seq):
        t = (tab - self.mu) / self.sd
        s = (torch.log1p(torch.clamp(seq, min=0)) - self.smu) / self.ssd
        return torch.sigmoid(self.net(t, s))


class Deep:
    """GRU + MLP with standardization fitted on the training rows; early stopping on validation AUC."""

    def __init__(self, cfg: Config):
        self.c = cfg["risk"]

    def _prep(self, X, seq):
        t = torch.tensor((np.asarray(X, np.float32) - self.mu) / self.sd)
        s = torch.tensor((np.log1p(np.clip(seq, 0, None)) - self.smu) / self.ssd, dtype=torch.float32)
        return t, s

    def fit(self, X, seq, y, Xv, seqv, yv):
        torch.manual_seed(self.c["seed"])
        X = np.asarray(X, np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        ls = np.log1p(np.clip(seq, 0, None))
        self.smu, self.ssd = ls.mean((0, 1)), ls.std((0, 1)) + 1e-6
        self.net = SeqNet(X.shape[1], seq.shape[2], self.c["hidden"])
        return self._train(X, seq, y, Xv, seqv, yv, self.c["lr"], self.c["epochs"], patience_max=6)

    def fine_tune(self, base: Deep, X, seq, y, Xv, seqv, yv):
        """Start from a trained model's weights and input scaling; continue training on new data at a lower rate."""
        import copy
        self.mu, self.sd, self.smu, self.ssd = base.mu, base.sd, base.smu, base.ssd
        self.net = copy.deepcopy(base.net)
        return self._train(np.asarray(X, np.float32), seq, y, Xv, seqv, yv, self.c["lr"] * 0.3, 15, patience_max=3)

    def _train(self, X, seq, y, Xv, seqv, yv, lr, epochs, patience_max):
        t, s = self._prep(X, seq)
        yt = torch.tensor(np.asarray(y, np.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        best = roc_auc_score(yv, self.predict(Xv, seqv)) if hasattr(self, "epochs_run") else -1.0
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        patience = 0
        g = torch.Generator().manual_seed(self.c["seed"])
        for epoch in range(epochs):
            self.net.train()
            perm = torch.randperm(len(yt), generator=g)
            for i in range(0, len(perm), self.c["batch_size"]):
                b = perm[i:i + self.c["batch_size"]]
                opt.zero_grad()
                lossf(self.net(t[b], s[b]), yt[b]).backward()
                opt.step()
            auc = roc_auc_score(yv, self.predict(Xv, seqv))
            if auc > best + 1e-4:
                best, patience = auc, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience += 1
                if patience >= patience_max:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_run, self.best_valid_auc = epoch + 1, best
        return self

    @torch.no_grad()
    def predict(self, X, seq):
        self.net.eval()
        t, s = self._prep(X, seq)
        return torch.sigmoid(self.net(t, s)).numpy()


FAMILIES = {"logistic": logistic, "hgb": hgb, "seqnet": Deep}


def split(panel: pd.DataFrame, seq: np.ndarray, years) -> tuple:
    m = panel["year"].isin(years).to_numpy()
    return panel.loc[m, data.FEATURES].to_numpy(np.float32), seq[m], panel.loc[m, "label"].to_numpy(), m


def fit_family(name: str, cfg: Config, panel, seq, train_years, valid_years):
    X, S, y, _ = split(panel, seq, train_years)
    Xv, Sv, yv, _ = split(panel, seq, valid_years)
    return FAMILIES[name](cfg).fit(X, S, y, Xv, Sv, yv)


def log_model(model, name: str, X_example, seq_example=None):
    """Log the model artifact: sklearn models natively, the torch net with mlflow.pytorch."""
    if isinstance(model, Tabular):
        # the HGB tree predictor is a type skops asks to confirm; this model was trained here, in this run
        trusted = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
        return mlflow.sklearn.log_model(model.est, name="model", input_example=X_example[:5],
                                        skops_trusted_types=trusted)
    module = Scored(model).eval()
    return mlflow.pytorch.log_model(module, name="model", serialization_format="pickle")


def run(cfg: Config) -> dict:
    panel, seq = data.load(cfg)
    r = cfg["risk"]
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("acute-risk")
    Xt, St, yt, _ = split(panel, seq, r["test_years"])
    Xv, Sv, yv, _ = split(panel, seq, r["valid_years"])
    results, models = {}, {}
    for name in FAMILIES:
        with mlflow.start_run(run_name=name) as run:
            model = fit_family(name, cfg, panel, seq, r["train_years"], r["valid_years"])
            m = metrics(yv, model.predict(Xv, Sv), "valid") | metrics(yt, model.predict(Xt, St), "test")
            mlflow.log_params({"family": name, "train_years": r["train_years"], "valid_years": r["valid_years"],
                               "n_features": len(data.FEATURES)} | ({"epochs_run": model.epochs_run}
                                                                   if name == "seqnet" else {}))
            mlflow.log_metrics(m)
            info = log_model(model, name, Xt, St)
            results[name] = m | {"run_id": run.info.run_id, "model_uri": info.model_uri}
            models[name] = model
            log.info("%s: valid AUC %.3f  test AUC %.3f", name, m["valid_auc"], m["test_auc"])
    best = max(results, key=lambda k: results[k]["valid_auc"])
    client = mlflow.MlflowClient()
    mv = mlflow.register_model(results[best]["model_uri"], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "family", best)
    log.info("registered %s v%s (%s) as champion", MODEL_NAME, mv.version, best)

    # scores for every year from each model; the champion's feed the outreach plan and the bandit's oracle
    Xa = panel[data.FEATURES].to_numpy(np.float32)
    scores = panel[["pid", "year", "label"]].copy()
    for name, model in models.items():
        scores[f"risk_{name}"] = model.predict(Xa, seq)
    scores["risk"] = scores[f"risk_{best}"]
    scores.to_parquet(cfg.path("data_dir") / "risk_scores.parquet", index=False)
    # paired bootstrap on the test year: champion AUC minus each other family's
    rng = np.random.default_rng(r["seed"])
    ts = scores[scores["year"].isin(r["test_years"])]
    yb = ts["label"].to_numpy()
    diffs = {k: [] for k in FAMILIES if k != best}
    for _ in range(1000):
        i = rng.integers(0, len(ts), len(ts))
        a_best = roc_auc_score(yb[i], ts[f"risk_{best}"].to_numpy()[i])
        for k, v in diffs.items():
            v.append(a_best - roc_auc_score(yb[i], ts[f"risk_{k}"].to_numpy()[i]))
    boot = {f"{best}_minus_{k}_test_auc": {"est": results[best]["test_auc"] - results[k]["test_auc"],
                                           "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])]}
            for k, v in diffs.items()}
    out = {"champion": best, "test_auc_differences": boot, "models": {k: {m: v for m, v in d.items() if m != "model_uri"} for k, d in results.items()}}
    (cfg.reports / "risk_models.json").write_text(json.dumps(out, indent=2))
    return out
