"""Coverage of the 10-90% forecast interval and skill by aquifer type, test wells."""
import json
import sys
from pathlib import Path

import pandas as pd

from common import score_table

data, R = Path(sys.argv[1]), Path(sys.argv[2])
wells = pd.read_csv(data / "wells.csv").set_index("well_id")
res = {}
for h in (7, 30, 90):
    for name in ("zeroshot", "gwfmkr_s0"):
        for m in "AB":
            d = pd.read_parquet(R / "preds" / f"{name}_{m}_H{h}_test_test.parquet")
            d = d[d.ok]
            cov = float(((d.obs >= d.q10) & (d.obs <= d.q90)).mean())
            s = score_table(d.assign(ok=True)).set_index("well_id")
            s["aquifer"] = wells.loc[s.index, "aquifer"]
            res[f"{name}_{m}_H{h}"] = dict(coverage_10_90=round(cov, 3),
                                           median_skill_by_aquifer=s.groupby("aquifer").skill.median().round(3).to_dict())
(R / "coverage.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, v)
