"""Shared benchmark helpers: well loading, forecast origins and scoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (7, 30, 90)
ORIGIN_STEP = 7


def load_split(data: Path) -> dict:
    return json.loads((Path(data) / "split.json").read_text())


def load_well(data: Path, well_id: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data) / "wells" / f"{well_id}.parquet").set_index("date")


def model_level(df: pd.DataFrame) -> np.ndarray:
    """Level series given to forecasters: raw observations minus gross outliers (bounds from the training
    period), gaps left as NaN. It uses no information after the day it describes."""
    return df["level_in"].to_numpy(float)


def origins(df: pd.DataFrame, start: str, end: str, horizon: int, step: int = ORIGIN_STEP) -> np.ndarray:
    """Positions of the first forecast day, every `step` days, with the whole horizon inside [start, end]
    and an input observation on the day before the origin."""
    idx = df.index
    lo, hi = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end), side="right")
    obs = df["level_in"].notna().to_numpy()
    pos = np.arange(max(lo, 1), hi - horizon + 1, step)
    return pos[obs[pos - 1]]


def last_valid(level: np.ndarray, pos: int) -> float:
    v = level[:pos]
    ok = np.isfinite(v)
    return float(v[ok][-1]) if ok.any() else np.nan


def score_table(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: well_id, origin, lead, pred, pers, obs, ok. Returns per-well RMSE and skill."""
    r = rows[rows.ok].copy()
    r["e2"] = (r.pred - r.obs) ** 2
    r["p2"] = (r.pers - r.obs) ** 2
    g = r.groupby("well_id").agg(rmse=("e2", "mean"), rmse_pers=("p2", "mean"), n=("e2", "size"))
    g["rmse"] = np.sqrt(g.rmse)
    g["rmse_pers"] = np.sqrt(g.rmse_pers)
    g["skill"] = 1 - g.rmse / g.rmse_pers
    return g.reset_index()
