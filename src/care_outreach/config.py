from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def __getitem__(self, key):
        return self.raw[key]

    def path(self, key: str) -> Path:
        p = Path(self.raw[key])
        return p if p.is_absolute() else ROOT / p

    @property
    def reports(self) -> Path:
        d = ROOT / "reports"
        (d / "figures").mkdir(parents=True, exist_ok=True)
        return d

    @property
    def mlflow_uri(self) -> str:
        return f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
