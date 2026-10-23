"""Select eligible wells and hold out test wells (fixed before any model is run).

Eligibility: at least 2 years of valid levels before 2023 and at least 1 year in 2023-2024.
Periods: training up to 2021-12-31, validation 2022, test 2023-2024.
Test wells: all eligible wells at 20% of the stations (stations stratified by province, seed 20261022),
so no test well shares a station with a training well.
Wells whose station could not be geocoded take the province of other wells from the same city.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261022
TEST_FRACTION = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    d = Path(a.data)
    w = pd.read_csv(d / "wells_all.csv")

    w["province"] = w["province"].where(~w["province"].isin(["unknown", "nan"]) & w["province"].notna(), None)
    city = w["station_name"].str[:2]
    city_prov = w.dropna(subset=["province"]).groupby(city)["province"].agg(lambda s: s.mode().iloc[0])
    w["province"] = w["province"].fillna(city.map(city_prov)).fillna("unknown")

    w["eligible"] = (w.n_valid_pre >= 2 * 365) & (w.n_valid_test >= 365)
    e = w[w.eligible].copy()
    rng = np.random.default_rng(SEED)
    e["station"] = e["station_name"].str.split("_").str[0]
    st = e.groupby("station")["province"].first().reset_index().sort_values("station")
    test_st = []
    for _, g in st.groupby("province"):
        n_test = int(round(TEST_FRACTION * len(g)))
        if n_test:
            test_st += list(rng.choice(g.station.to_numpy(), n_test, replace=False))
    e["role"] = np.where(e.station.isin(test_st), "test", "train")
    w["role"] = "excluded"
    w.loc[e.index, "role"] = e["role"]
    w.to_csv(d / "wells.csv", index=False)
    split = {"seed": SEED, "train_end": "2021-12-31", "val": ["2022-01-01", "2022-12-31"],
             "test": ["2023-01-01", "2024-12-31"],
             "train_wells": sorted(w.loc[w.role == "train", "well_id"]),
             "test_wells": sorted(w.loc[w.role == "test", "well_id"])}
    (d / "split.json").write_text(json.dumps(split, indent=1))
    print(w.role.value_counts().to_dict())
    print(pd.crosstab([w.province, w.aquifer], w.role))


if __name__ == "__main__":
    main()
