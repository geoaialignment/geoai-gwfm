"""Read groundwater-level and rainfall tables supplied by a user."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

DATE_NAMES = ("date", "datetime", "time", "day", "ymd", "날짜", "일자", "일시", "관측일")
LEVEL_NAMES = ("level", "water_level", "waterlevel", "gwl", "head", "elevation", "수위", "지하수위", "관측수위")
RAIN_NAMES = ("rain", "rainfall", "precip", "precipitation", "prcp", "강수", "강수량", "강우", "강우량")


def _pick(cols: list[str], names: tuple[str, ...]) -> str | None:
    low = {c: str(c).strip().lower() for c in cols}
    for n in names:
        for c, l in low.items():
            if l == n:
                return c
    for n in names:
        for c, l in low.items():
            if n in l:
                return c
    return None


def read_table(path: str | Path | None = None, text: str | None = None) -> pd.DataFrame:
    if (path is None) == (text is None):
        raise ValueError("Give exactly one of a file path or CSV text.")
    if path is not None:
        p = Path(path).expanduser()
        if p.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(p)
        return pd.read_csv(p, sep=None, engine="python", encoding_errors="replace")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def daily_series(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a daily frame with columns level (m, NaN for gaps) and rain (mm, 0 for missing)."""
    cols = list(df.columns)
    dcol, lcol, rcol = _pick(cols, DATE_NAMES), _pick(cols, LEVEL_NAMES), _pick(cols, RAIN_NAMES)
    if dcol is None or lcol is None:
        raise ValueError(f"Could not find date and water-level columns in {cols}. "
                         "Name them for example 'date' and 'level' (rainfall optional, e.g. 'rain').")
    raw = df[dcol]
    if pd.api.types.is_numeric_dtype(raw):  # e.g. 20240101: read as YYYYMMDD, never as a timestamp in ns
        dates = pd.to_datetime(raw.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        dates = pd.to_datetime(raw, errors="coerce")
    d = pd.DataFrame({"date": dates,
                      "level": pd.to_numeric(df[lcol], errors="coerce")})
    if rcol is not None:
        d["rain"] = pd.to_numeric(df[rcol], errors="coerce")
    d = d.dropna(subset=["date"]).set_index("date").sort_index()
    agg = {"level": "mean", **({"rain": "sum"} if rcol else {})}
    d = d.resample("D").agg(agg)
    if rcol is None:
        d["rain"] = np.nan
    info = dict(date_column=dcol, level_column=lcol, rain_column=rcol,
                first_date=str(d.index[0].date()), last_date=str(d.index[-1].date()), n_days=int(len(d)),
                n_level_missing=int(d["level"].isna().sum()), last_level=float(d["level"].dropna().iloc[-1]),
                has_rain=rcol is not None)
    # the series ends at the last observed level
    last = d["level"].last_valid_index()
    return d.loc[:last], info
