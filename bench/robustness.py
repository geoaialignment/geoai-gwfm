"""Robustness checks on the test stations (results/robustness.json):
  * oracle per-well LSTM: each well's LSTM setting chosen with hindsight on the test period (the most
    favourable choice for the baseline), compared with GeoAI-GWFM-KR;
  * last-day scoring: RMSE on day H only instead of days 1..H.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze import CONFIGS, per_well_rnn
from common import score_table

R, D = Path(sys.argv[1]), Path(sys.argv[2])
W = json.loads((D / "split.json").read_text())["test_wells"]


def seedmean(frames, lastonly=False):
    s = [score_table(f[f.lead == f.lead.max()] if lastonly else f).set_index("well_id") for f in frames]
    out = s[0][["rmse_pers"]].copy()
    out["rmse"] = np.mean([x.rmse for x in s], 0)
    return out


def rnn(w, tag, h, lastonly=False):
    return seedmean([pd.read_parquet(R / "rnn" / w / f"{tag}_H{h}_s{s}.parquet") for s in (7, 42, 99)], lastonly).iloc[0].rmse


res = {}
for h in (7, 30, 90):
    preds = [pd.read_parquet(R / "preds" / f"gwfmkr_s{s}_A_H{h}_test_test.parquet") for s in range(3)]
    A, Al = seedmean(preds), seedmean(preds, True)
    Zl = seedmean([pd.read_parquet(R / "preds" / f"zeroshot_A_H{h}_test_test.parquet")], True)
    orc = pd.Series([min(rnn(w, f"lstm_plain_h{a}_w{b}", h) for a, b in CONFIGS) for w in W], index=W)
    d = A.loc[W, "rmse"] - orc
    res[f"oracle_H{h}"] = dict(adapter_better=int((d < 0).sum()), n=len(W), p=float(wilcoxon(d).pvalue),
                               oracle_median_skill=float(np.median(1 - orc / A.loc[W, "rmse_pers"])))
    ch = dict(zip(*per_well_rnn(R / "rnn", W, h, "lstm", "plain", [(a, b, None) for a, b in CONFIGS])[["well_id", "choice"]].T.values))
    Ll = pd.Series([rnn(w, "lstm_plain_h{}_w{}".format(*eval(ch[w])[:2]), h, True) for w in W], index=W)
    Gl = pd.Series([rnn(w, "gru_plain_h64_w120", h, True) for w in W], index=W)
    for name, a, b in (("lastday_adapter_vs_lstm", Al.rmse, Ll), ("lastday_adapter_vs_gru", Al.rmse, Gl),
                       ("lastday_zeroshot_vs_lstm", Zl.rmse, Ll)):
        dd = a.loc[W] - b
        res[f"{name}_H{h}"] = dict(first_better=int((dd < 0).sum()), n=len(W), p=float(wilcoxon(dd).pvalue))
(R / "robustness.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
