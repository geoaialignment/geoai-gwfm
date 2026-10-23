"""Validation-period (2022, training wells) skill for each candidate adapter setting."""
import sys
from pathlib import Path

import pandas as pd

from common import score_table

R = Path(sys.argv[1])
names = sorted({f.name.split("_A_H")[0] for f in R.glob("*_A_H7_train_val.parquet")})
rows = []
for n in names:
    row = {"name": n}
    for h in (7, 30, 90):
        for m in "AB":
            f = R / f"{n}_{m}_H{h}_train_val.parquet"
            if f.exists():
                row[f"{m}{h}"] = round(float(score_table(pd.read_parquet(f)).skill.median()), 3)
    rows.append(row)
t = pd.DataFrame(rows).set_index("name")
t["mean"] = t.mean(1).round(3)
print(t.to_string())
