"""Write a synthetic three-year daily well record (examples/example_well.csv) for trying the tools.

Rainfall follows a monsoon-like seasonal cycle. The level follows the groundwater fluctuation equation
dh = g_r * R - g_d * (h - h_ref) with noise. The series is synthetic and is not an observed well.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
doy = dates.dayofyear.to_numpy()
p_wet = 0.15 + 0.35 * np.exp(-0.5 * ((doy - 200) / 30) ** 2)
rain = np.where(rng.random(len(dates)) < p_wet, rng.gamma(0.8, 18 + 25 * np.exp(-0.5 * ((doy - 200) / 30) ** 2)), 0.0)
h = np.empty(len(dates))
h[0] = 52.0
for t in range(1, len(dates)):
    h[t] = h[t - 1] + 0.004 * rain[t] - 0.015 * (h[t - 1] - 51.2) + rng.normal(0, 0.004)
df = pd.DataFrame({"date": dates.date, "level": h.round(3), "rain": rain.round(1)})
df.loc[rng.choice(len(df), 20, replace=False), "level"] = np.nan  # a few missing days
out = Path(__file__).with_name("example_well.csv")
df.to_csv(out, index=False)
print(out, len(df))
