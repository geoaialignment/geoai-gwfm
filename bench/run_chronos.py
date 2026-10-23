"""Forecast the test wells with Chronos-2 (zero-shot or with an adapter) and save every forecast.

Modes: A = past rain + observed future rain (perfect rainfall forecast),
       B = past rain only (no rainfall forecast).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwfm.forecast import ForecastRequest, load_pipeline, predict  # noqa: E402

from common import HORIZONS, load_split, load_well, model_level, origins  # noqa: E402


def run(pipe, data: Path, wells: list[str], horizon: int, mode: str, start: str, end: str) -> pd.DataFrame:
    reqs, meta = [], []
    for w in wells:
        df = load_well(data, w)
        lev = model_level(df)
        rain = df["rain"].to_numpy(float)
        obs = df["level_raw"].to_numpy(float)
        ok = df["eval_ok"].to_numpy()
        for o in origins(df, start, end, horizon):
            reqs.append(ForecastRequest(level=lev[:o], rain=None if mode == "U" else rain[:o],
                                        future_rain=rain[o:o + horizon] if mode == "A" else None))
            meta.append((w, df.index[o], lev[o - 1], obs[o:o + horizon], ok[o:o + horizon]))
    q = predict(pipe, reqs, horizon)
    frames = []
    lead = np.arange(1, horizon + 1)
    for (w, t0, pers, ob, okk), qq in zip(meta, q):
        frames.append(pd.DataFrame(dict(well_id=w, origin=t0, lead=lead, q10=qq[0], pred=qq[1], q90=qq[2],
                                        pers=pers, obs=ob, ok=okk)))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--modes", default="A,B")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--wells", default="test", choices=["test", "train"])
    ap.add_argument("--period", default="test", choices=["test", "val"])
    a = ap.parse_args()
    sp = load_split(a.data)
    wells = sp[f"{a.wells}_wells"]
    start, end = sp[a.period]
    pipe = load_pipeline(a.adapter)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for h in map(int, a.horizons.split(",")):
        for m in a.modes.split(","):
            t = time.time()
            df = run(pipe, Path(a.data), wells, h, m, start, end)
            f = out / f"{a.name}_{m}_H{h}_{a.wells}_{a.period}.parquet"
            df.to_parquet(f, index=False)
            print(f"{f.name}: {df.origin.nunique()} origins, {len(df)} rows, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
