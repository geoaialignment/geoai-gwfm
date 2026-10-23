"""Scenario consistency on the test wells: does the forecast respond to rainfall with the right sign?

At origins every 28 days in 2023-2024, each test well is forecast 30 days ahead with the observed future
rainfall scaled by 0, 1 and 2, and with 100 mm added on day 3. Reported per forecaster:
  * monotonic share: median level at day 30 ordered x0 <= x1 <= x2 (tolerance 1 mm),
  * storm response: peak rise of the median over the base case, the day of the peak, and the share of the
    peak still present at day 30.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwfm.forecast import ForecastRequest, load_pipeline, predict  # noqa: E402

from common import load_split, load_well, model_level, origins  # noqa: E402

H = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    sp = load_split(a.data)
    pipe = load_pipeline(a.adapter)
    cases = {"x0": lambda r: r * 0, "x1": lambda r: r, "x2": lambda r: r * 2,
             "storm": lambda r: r + np.eye(1, H, 2).ravel() * 100.0}
    reqs, meta = [], []
    for w in sp["test_wells"]:
        df = load_well(a.data, w)
        lev, rain = model_level(df), df["rain"].to_numpy(float)
        for o in origins(df, *sp["test"], H, step=28):
            for c, f in cases.items():
                reqs.append(ForecastRequest(lev[:o], rain[:o], f(rain[o:o + H])))
                meta.append((w, o, c))
    q = predict(pipe, reqs, H)[:, 1]
    df = pd.DataFrame(meta, columns=["well", "origin", "case"])
    med = {k: q[np.asarray(v)] for k, v in df.groupby("case").groups.items()}
    d01, d12 = med["x1"][:, -1] - med["x0"][:, -1], med["x2"][:, -1] - med["x1"][:, -1]
    rise = med["storm"] - med["x1"]
    peak = rise.max(1)
    res = dict(n_forecasts=int(len(d01)),
               monotonic_share=float(((d01 >= -1e-3) & (d12 >= -1e-3)).mean()),
               wrong_sign_share_x0_x1=float((d01 < -1e-3).mean()),
               storm_peak_rise_median_m=float(np.median(peak)),
               storm_positive_share=float((peak > 1e-3).mean()),
               storm_peak_day_median=float(np.median(rise.argmax(1) + 1)),
               storm_retained_at_day30_median=float(np.nanmedian(rise[:, -1] / np.where(peak > 1e-3, peak, np.nan))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scenario_{a.name}.json").write_text(json.dumps(res, indent=1))
    print(a.name, json.dumps(res))


if __name__ == "__main__":
    main()
