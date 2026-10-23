"""Build the benchmark well files from daily GIMS water levels and KMA weather.

Inputs are two long-format CSV files:
  levels  : station_id, date, water_level
  weather : station_id, date, temp, rain, wind, hpa, humid
and, optionally, a station table with columns 측정망 (station name), 정제주소 (address), 경도, 위도.

Output: one parquet file per well (daily index) and wells.csv with metadata.

Cleaning keeps the raw level and flags three kinds of bad values:
  * flat runs of 10 or more days at exactly the same value (stuck logger),
  * isolated spikes (Hampel filter, 7-day window, 6 robust standard deviations), except rises
    within three days of at least 10 mm of rain, which are kept as recharge responses,
  * gross outliers outside the median +/- 10 IQR of the record up to 2022.
These flags decide only which observed days are scored (eval_ok); they may use the whole record.
Model inputs (level_in) are the raw observations with gross outliers removed, using bounds from data
up to 2021-12-31 (the end of training), so no input depends on later data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"
WEATHER = ["rain", "temp", "wind", "hpa", "humid"]

PROVINCE = {
    "경상북도": "Gyeongbuk", "경북": "Gyeongbuk", "경상남도": "Gyeongnam", "경남": "Gyeongnam",
    "대구광역시": "Daegu", "부산광역시": "Busan", "울산광역시": "Ulsan", "강원도": "Gangwon",
    "강원특별자치도": "Gangwon", "전라북도": "Jeonbuk", "전북특별자치도": "Jeonbuk",
}


def aquifer_of(name: str) -> str:
    if "충적" in name:
        return "alluvial"
    if "암반" in name:
        return "bedrock"
    return "unknown"


def flat_runs(x: np.ndarray, min_days: int = 10) -> np.ndarray:
    flags = np.zeros(len(x), bool)
    i, n = 0, len(x)
    while i < n:
        if not np.isfinite(x[i]):
            i += 1
            continue
        j = i + 1
        while j < n and np.isfinite(x[j]) and x[j] == x[i]:
            j += 1
        if j - i >= min_days:
            flags[i:j] = True
        i = j
    return flags


def hampel(x: pd.Series, half: int = 3, k: float = 6.0) -> np.ndarray:
    med = x.rolling(2 * half + 1, center=True, min_periods=half + 1).median()
    mad = (x - med).abs().rolling(2 * half + 1, center=True, min_periods=half + 1).median()
    scale = 1.4826 * mad
    floor = 1.4826 * np.nanmedian(np.abs(x.diff()))  # avoid flagging tiny wiggles when the window is flat
    scale = np.maximum(scale, floor if np.isfinite(floor) and floor > 0 else 1e-3)
    return ((x - med).abs() > k * scale).to_numpy()


def clean_well(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    lev = df["level_raw"]
    ref = lev[lev.index <= VAL_END].dropna()
    q1, q3 = np.percentile(ref, [25, 75]) if len(ref) else (np.nan, np.nan)
    med = float(np.median(ref)) if len(ref) else np.nan
    iqr = max(q3 - q1, 1e-3)
    gross = ((lev < med - 10 * iqr) | (lev > med + 10 * iqr)).to_numpy()
    flat = flat_runs(lev.to_numpy())
    rain3 = df["rain"].rolling(3, min_periods=1).max()
    rising = lev > lev.rolling(7, center=True, min_periods=4).median()
    spike = hampel(lev) & ~(rising & (rain3 >= 10)).to_numpy()
    bad = (gross | flat | spike) & lev.notna().to_numpy()
    clean = lev.mask(bad)
    out = df.copy()
    out["eval_ok"] = clean.notna()
    tr = lev[lev.index <= TRAIN_END].dropna()
    if len(tr) >= 30:
        t1, t3 = np.percentile(tr, [25, 75])
        tmed, tiqr = float(np.median(tr)), max(t3 - t1, 1e-3)
        out["level_in"] = lev.mask((lev < tmed - 10 * tiqr) | (lev > tmed + 10 * tiqr))
    else:
        out["level_in"] = lev
    out["level"] = out["level_in"].interpolate(limit_area="inside").ffill().bfill()
    rep = dict(n_obs=int(lev.notna().sum()), n_gross=int((gross & lev.notna()).sum()),
               n_flat=int((flat & lev.notna().to_numpy()).sum()), n_spike=int((spike & lev.notna().to_numpy()).sum()),
               n_flagged=int(bad.sum()))
    return out, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--stations", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "wells").mkdir(parents=True, exist_ok=True)

    L = pd.read_csv(a.levels, parse_dates=["date"])
    W = pd.read_csv(a.weather, parse_dates=["date"])
    st = None
    if a.stations:
        st = pd.read_csv(a.stations).drop_duplicates("측정망").set_index("측정망")

    rows = []
    for k, (name, g) in enumerate(sorted(L.groupby("station_id"), key=lambda t: t[0])):
        g = g.set_index("date").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        df = pd.DataFrame(index=idx)
        df.index.name = "date"
        df["level_raw"] = g["water_level"].reindex(idx).astype(float)
        w = W[W.station_id == name].set_index("date").sort_index()
        for c in WEATHER:
            df[c] = w[c].reindex(idx).astype(float) if c in w else np.nan
        df["rain"] = df["rain"].fillna(0.0).clip(lower=0)
        for c in WEATHER[1:]:
            df[c] = df[c].interpolate(limit_direction="both")
        df, rep = clean_well(df)
        well_id = f"YN{k + 1:03d}"
        df.reset_index().to_parquet(out / "wells" / f"{well_id}.parquet", index=False)

        base = name.split("_")[0]
        prov, lon, lat = "unknown", np.nan, np.nan
        if st is not None and base in st.index:
            r = st.loc[base]
            prov = PROVINCE.get(str(r["정제주소"]).split()[0], str(r["정제주소"]).split()[0])
            lon, lat = r["경도"], r["위도"]
        ok = df["eval_ok"]
        rows.append(dict(
            well_id=well_id, station_name=name, aquifer=aquifer_of(name), province=prov, lon=lon, lat=lat,
            date_start=str(idx[0].date()), date_end=str(idx[-1].date()),
            n_valid_train=int(ok[: TRAIN_END].sum()), n_valid_val=int(ok["2022-01-01":VAL_END].sum()),
            n_valid_pre=int(ok[:VAL_END].sum()),
            n_valid_test=int(ok[TEST_START:TEST_END].sum()),
            weather_missing=float(w["rain"].reindex(idx).isna().mean()) if "rain" in w else 1.0,
            level_sd=float(df["level"].std()), **rep))
    wells = pd.DataFrame(rows)
    wells.to_csv(out / "wells_all.csv", index=False)
    print(wells[["n_obs", "n_flagged", "n_gross", "n_flat", "n_spike"]].sum().to_dict())
    print(wells.groupby(["province", "aquifer"]).size())


if __name__ == "__main__":
    main()
