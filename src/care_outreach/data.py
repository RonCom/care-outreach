"""Build the member-year panel from Synthea CSVs.

One row per member and outcome year Y (member alive on Jan 1 of Y, aged >= min_age, with any encounter in the two
prior years). Features describe the member on Jan 1 of Y; the label is any emergency visit or inpatient stay in Y.

Outputs (data/):
  panel.parquet   member-year features + label
  seq.npy         monthly utilization for the 24 months before Y, shape (rows, 24, 5), aligned with panel rows
"""
from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

# Chronic conditions: SNOMED descriptions in Synthea matched by substring (case-insensitive).
CHRONIC = {
    "diabetes": ["diabetes mellitus", "prediabetes"],
    "hypertension": ["hypertension"],
    "ihd": ["ischemic heart disease", "myocardial infarction", "coronary"],
    "chf": ["heart failure"],
    "copd": ["chronic obstructive", "emphysema", "chronic bronchitis"],
    "ckd": ["chronic kidney disease", "end-stage renal"],
    "dementia": ["alzheimer", "dementia"],
    "cancer": ["malignant neoplasm", "carcinoma"],
    "afib": ["atrial fibrillation"],
    "asthma": ["asthma"],
    "obesity": ["body mass index 30+", "obesity"],
    "hyperlipidemia": ["hyperlipidemia", "hypertriglyceridemia"],
}
CHANNELS = ["amb", "urgent", "ed", "ip", "log_cost"]
CLASS_GROUP = """CASE WHEN ENCOUNTERCLASS IN ('emergency') THEN 'ed'
                      WHEN ENCOUNTERCLASS IN ('inpatient') THEN 'ip'
                      WHEN ENCOUNTERCLASS IN ('urgentcare') THEN 'urgent'
                      WHEN ENCOUNTERCLASS IN ('snf', 'hospice') THEN 'post_acute'
                      ELSE 'amb' END"""


def build(cfg: Config) -> pd.DataFrame:
    src = cfg.path("synthea_csv")
    years = cfg["years"]
    y0, y1 = min(years) - 2, max(years)
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW enc AS SELECT PATIENT AS pid, START::TIMESTAMP AS ts, {CLASS_GROUP} AS grp,
               TOTAL_CLAIM_COST::DOUBLE AS cost
        FROM read_csv_auto('{src}/encounters.csv', header=true, all_varchar=true, quote='"');
        CREATE VIEW pat AS SELECT Id AS pid, BIRTHDATE::DATE AS birth, TRY_CAST(DEATHDATE AS DATE) AS death,
               GENDER AS gender, TRY_CAST(INCOME AS DOUBLE) AS income
        FROM read_csv_auto('{src}/patients.csv', header=true, all_varchar=true, quote='"');
        CREATE VIEW cond AS SELECT PATIENT AS pid, START::DATE AS start, TRY_CAST(STOP AS DATE) AS stop,
               lower(DESCRIPTION) AS descr
        FROM read_csv_auto('{src}/conditions.csv', header=true, all_varchar=true, quote='"');
        CREATE VIEW med AS SELECT PATIENT AS pid, START::TIMESTAMP AS start, TRY_CAST(STOP AS TIMESTAMP) AS stop,
               CODE AS code
        FROM read_csv_auto('{src}/medications.csv', header=true, all_varchar=true, quote='"');
        CREATE TABLE yrs AS SELECT unnest({list(years)}) AS y;
    """)
    # eligible member-years
    con.execute(f"""
        CREATE TABLE my AS
        SELECT p.pid, y.y AS year, date_diff('year', p.birth, make_date(y.y, 1, 1)) AS age,
               (p.gender = 'M')::INT AS male, ln(1 + coalesce(p.income, 0)) AS log_income
        FROM pat p, yrs y
        WHERE date_diff('year', p.birth, make_date(y.y, 1, 1)) >= {cfg['min_age']}
          AND (p.death IS NULL OR p.death > make_date(y.y, 1, 1))
          AND EXISTS (SELECT 1 FROM enc e WHERE e.pid = p.pid
                      AND e.ts >= make_date(y.y - 2, 1, 1) AND e.ts < make_date(y.y, 1, 1))
    """)
    # prior-year utilization, outcome-year label
    prior = con.execute("""
        SELECT my.pid, my.year,
          count(*) FILTER (WHERE e.grp = 'amb'    AND year(e.ts) = my.year - 1) AS amb_1y,
          count(*) FILTER (WHERE e.grp = 'urgent' AND year(e.ts) = my.year - 1) AS urgent_1y,
          count(*) FILTER (WHERE e.grp = 'ed'     AND year(e.ts) = my.year - 1) AS ed_1y,
          count(*) FILTER (WHERE e.grp = 'ip'     AND year(e.ts) = my.year - 1) AS ip_1y,
          count(*) FILTER (WHERE e.grp = 'post_acute' AND year(e.ts) = my.year - 1) AS post_acute_1y,
          count(*) FILTER (WHERE e.grp IN ('ed', 'ip') AND year(e.ts) = my.year - 2) AS acute_2y,
          ln(1 + coalesce(sum(e.cost) FILTER (WHERE year(e.ts) = my.year - 1), 0)) AS log_cost_1y,
          count(*) FILTER (WHERE e.grp IN ('ed', 'ip') AND year(e.ts) = my.year) AS acute_events,
          coalesce(sum(e.cost) FILTER (WHERE year(e.ts) = my.year), 0) AS cost_y
        FROM my LEFT JOIN enc e ON e.pid = my.pid AND year(e.ts) BETWEEN my.year - 2 AND my.year
        GROUP BY ALL
    """).df()
    flags = ",\n".join(
        f"max(({' OR '.join(f'c.descr LIKE {chr(39)}%{k}%{chr(39)}' for k in keys)})::INT) AS {name}"
        for name, keys in CHRONIC.items())
    chron = con.execute(f"""
        SELECT my.pid, my.year, {flags}
        FROM my JOIN cond c ON c.pid = my.pid AND c.start < make_date(my.year, 1, 1)
             AND (c.stop IS NULL OR c.stop >= make_date(my.year, 1, 1))
        GROUP BY ALL
    """).df()
    meds = con.execute("""
        SELECT my.pid, my.year, count(DISTINCT m.code) AS n_meds
        FROM my JOIN med m ON m.pid = my.pid AND m.start < make_date(my.year, 1, 1)
             AND (m.stop IS NULL OR m.stop >= make_date(my.year, 1, 1))
        GROUP BY ALL
    """).df()
    panel = (con.execute("SELECT * FROM my").df()
             .merge(prior, on=["pid", "year"], how="left")
             .merge(chron, on=["pid", "year"], how="left")
             .merge(meds, on=["pid", "year"], how="left"))
    for c in list(CHRONIC) + ["n_meds"]:
        panel[c] = panel[c].fillna(0).astype(int)
    panel["n_chronic"] = panel[list(CHRONIC)].sum(axis=1)
    panel["label"] = (panel["acute_events"] > 0).astype(int)
    panel = panel.sort_values(["year", "pid"]).reset_index(drop=True)

    # monthly sequence for the 24 months before Y
    monthly = con.execute(f"""
        SELECT pid, (year(ts) - {y0}) * 12 + month(ts) - 1 AS m,
          count(*) FILTER (WHERE grp = 'amb') AS amb, count(*) FILTER (WHERE grp = 'urgent') AS urgent,
          count(*) FILTER (WHERE grp = 'ed') AS ed, count(*) FILTER (WHERE grp = 'ip') AS ip,
          ln(1 + sum(cost)) AS log_cost
        FROM enc WHERE year(ts) BETWEEN {y0} AND {y1} GROUP BY ALL
    """).df()
    con.close()
    pids = pd.Index(panel["pid"].unique())
    n_months = (y1 - y0 + 1) * 12
    dense = np.zeros((len(pids), n_months, len(CHANNELS)), np.float32)
    dense[pids.get_indexer(monthly["pid"]), monthly["m"].to_numpy()] = monthly[CHANNELS].to_numpy(np.float32)
    pi = pids.get_indexer(panel["pid"])
    start = (panel["year"].to_numpy() - 2 - y0) * 12
    seq = np.stack([dense[p, s:s + 24] for p, s in zip(pi, start)])

    out = cfg.path("data_dir")
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "panel.parquet", index=False)
    np.save(out / "seq.npy", seq)
    log.info("panel: %d member-years, %d members, acute rate %.3f", len(panel), panel["pid"].nunique(),
             panel["label"].mean())
    log.info("rows by year:\n%s", panel.groupby("year")["label"].agg(["size", "mean"]).to_string())
    return panel


def load(cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    d = cfg.path("data_dir")
    return pd.read_parquet(d / "panel.parquet"), np.load(d / "seq.npy")


FEATURES = (["age", "male", "log_income", "amb_1y", "urgent_1y", "ed_1y", "ip_1y", "post_acute_1y", "acute_2y",
             "log_cost_1y", "n_meds", "n_chronic"] + list(CHRONIC))
