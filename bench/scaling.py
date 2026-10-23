"""Median test skill (future rain) against the number of training wells in the adapter (all seed 0)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import score_table

P = Path(sys.argv[1]) / "preds"
n_all = len(json.loads((Path(sys.argv[2]) / "split.json").read_text())["train_wells"])
out = {}
for h in (7, 30, 90):
    row = {"0": float(score_table(pd.read_parquet(P / f"zeroshot_A_H{h}_test_test.parquet")).skill.median())}
    for n in (10, 40, 100):
        v = [float(score_table(pd.read_parquet(f)).skill.median()) for f in sorted(P.glob(f"scale_n{n}_s*_A_H{h}_test_test.parquet"))]
        row[str(n)] = float(np.mean(v))
    row[str(n_all)] = float(score_table(pd.read_parquet(P / f"gwfmkr_s0_A_H{h}_test_test.parquet")).skill.median())
    out[f"H{h}"] = row
(P.parent / "scaling.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
