"""Observed level rise after heavy-rain days in the test wells' records before 2023 (reference for the
scenario check). Events: days with at least 80 mm, at least 15 days apart; rise = maximum level in the next
15 days minus the level on the day before."""
import json
import sys
from pathlib import Path

import numpy as np

from common import load_split, load_well

data, out = Path(sys.argv[1]), Path(sys.argv[2])
rises, days = [], []
for w in load_split(data)["test_wells"]:
    d = load_well(data, w).loc[:"2022-12-31"]
    lev, r = d.level_raw.where(d.eval_ok).to_numpy(), d.rain.to_numpy()
    last = -99
    for e in np.flatnonzero(r >= 80):
        if e - last < 15 or e < 1 or e + 15 >= len(lev):
            continue
        last = e
        win = lev[e:e + 15]
        if not np.isfinite(lev[e - 1]) or np.isfinite(win).sum() < 10:
            continue
        rises.append(np.nanmax(win) - lev[e - 1])
        days.append(np.nanargmax(win) + 1)
res = dict(n_events=len(rises), median_rise_m=float(np.median(rises)),
           q25_rise_m=float(np.percentile(rises, 25)), q75_rise_m=float(np.percentile(rises, 75)),
           median_peak_day=float(np.median(days)))
(out / "observed_storm_response.json").write_text(json.dumps(res, indent=1))
print(res)
