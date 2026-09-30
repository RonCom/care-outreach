"""CLI:  care-outreach [build|risk|plan|bandit|monitor|steer|all]   and   care-outreach providers --fwa-dir PATH"""
from __future__ import annotations

import argparse
import logging
import os
import warnings


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="care-outreach")
    ap.add_argument("step", choices=["build", "risk", "plan", "bandit", "monitor", "providers", "steer", "all"])
    ap.add_argument("--fwa-dir", help="providers: path to a MedicareFWA checkout with data/raw and data/out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    from . import bandit, data, monitor, optimize, risk, steer
    from .config import Config
    cfg = Config.load()
    steps = ["build", "risk", "plan", "bandit", "monitor", "steer"] if a.step == "all" else [a.step]
    if a.step == "providers":
        from pathlib import Path
        steer.extract_providers(Path(a.fwa_dir), cfg.path("steer_data"))
        return
    for s in steps:
        {"build": data.build, "risk": risk.run, "plan": optimize.run, "bandit": bandit.run,
         "monitor": monitor.run, "steer": steer.run}[s](cfg)


if __name__ == "__main__":
    main()
