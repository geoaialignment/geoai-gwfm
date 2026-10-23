"""Command line: gwfm forecast --csv well.csv --horizon 30 [--future-rain rain.csv] [--zeroshot]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__
from .io import daily_series, read_table
from .model import forecast_well, resolve_adapter
from .plot import plot_forecast


def main():
    ap = argparse.ArgumentParser(prog="gwfm", description="GeoAI-GWFM-KR groundwater-level forecasting")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="forecast one well")
    f.add_argument("--csv", required=True, help="table with date, level and (optional) rain columns")
    f.add_argument("--horizon", type=int, default=30)
    f.add_argument("--future-rain", default=None, help="CSV with one rainfall value (mm) per forecast day")
    f.add_argument("--zeroshot", action="store_true", help="use Chronos-2 without the adapter")
    f.add_argument("--adapter", default=None, help="local adapter folder (default: download or cache)")
    f.add_argument("--out", default="gwfm_out")
    sub.add_parser("download", help="download the adapter into the local cache")
    a = ap.parse_args()

    if a.cmd == "download":
        print(resolve_adapter())
        return
    series, info = daily_series(read_table(a.csv))
    fr = None
    if a.future_rain:
        t = pd.read_csv(a.future_rain)
        fr = pd.to_numeric(t[t.columns[-1]], errors="coerce").fillna(0).to_numpy()
    model = "zeroshot" if a.zeroshot else "adapter"
    fc = forecast_well(series, a.horizon, fr, model, a.adapter)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fc.table.to_csv(out / f"forecast_{a.horizon}d.csv", index=False)
    plot_forecast(series, {model: fc.table}, out / f"forecast_{a.horizon}d.png",
                  future_rain={"future rain": list(fr)} if fr is not None else None)
    print(f"{info['last_date']} last level {info['last_level']:.3f} m; {fc.mode}; context {fc.n_context_days} days")
    print(fc.table.iloc[[0, len(fc.table) // 2, -1]].to_string(index=False))
    print(f"-> {out / f'forecast_{a.horizon}d.csv'}")


if __name__ == "__main__":
    main()
