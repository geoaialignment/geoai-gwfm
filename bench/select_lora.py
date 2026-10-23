"""Select the LoRA setting on the validation year (training wells, 2022), with the rule fixed in PLAN.md:
highest mean of the median skill over (future rain, past rain only) x (7, 30, 90 days); it must beat Chronos-2
without the adapter on the same score."""
import json
import sys
from pathlib import Path

import pandas as pd

from common import score_table

R = Path(sys.argv[1])
scores = {}
for f in sorted(R.glob("*_A_H7_train_val.parquet")):
    n = f.name.split("_A_H7")[0]
    v = [score_table(pd.read_parquet(R / f"{n}_{m}_H{h}_train_val.parquet")).skill.median()
         for m in "AB" for h in (7, 30, 90)]
    scores[n] = float(sum(v) / len(v))
best = max((k for k in scores if k != "zeroshot"), key=scores.get)
sel = dict(scores=scores, selected=best, beats_zeroshot=scores[best] > scores["zeroshot"])
(R / "selected.json").write_text(json.dumps(sel, indent=1))
print(json.dumps(sel))
