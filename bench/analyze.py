"""Benchmark summary on the test stations: per-well RMSE and skill against persistence for every forecaster.

Rules fixed in PLAN.md before the results were read:
  * every trained model: RMSE per well = mean over three training seeds;
  * per-well LSTM: (hidden, window) chosen per well and horizon from {64, 128} x {120, 365} by the lowest
    seed-mean validation loss (2022); GRU uses hidden 64, window 120;
  * LSTM + equation: hidden 64, window 120, weight chosen per well and horizon from {0.1, 1, 10};
    its effect is judged against the LSTM with the same hidden size and window;
  * regional LSTM: chosen per horizon from six settings, hidden {64, 128} x window {120, 365} at learning rate
    1e-3 and hidden {64, 128} x window 120 at 1e-4, by the lowest seed-mean validation loss;
  * comparisons are paired by well (Wilcoxon signed-rank test on RMSE), and repeated on station means of the
    paired RMSE differences, because wells at one station are not independent;
  * every forecaster must be scored on the same (origin, lead) days at every well; the script stops otherwise;
  * record groups: long = record starts in 2012 or earlier, short = record starts in 2019 or later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import HORIZONS, score_table

SEEDS = (7, 42, 99)
CONFIGS = [(64, 120), (64, 365), (128, 120), (128, 365)]
LAMS = (0.1, 1.0, 10.0)
REG_CONFIGS = [(64, 120, 1e-3), (128, 120, 1e-3), (64, 365, 1e-3), (128, 365, 1e-3), (64, 120, 1e-4), (128, 120, 1e-4)]
A, Z = "GeoAI-GWFM-KR, future rain", "Chronos-2 alone, future rain"
AB, ZB = "GeoAI-GWFM-KR, past rain only", "Chronos-2 alone, past rain only"
PW, PW0, GRU, EQ, REG = "Per-well LSTM", "Per-well LSTM (64, 120)", "Per-well GRU", "Per-well LSTM + equation", "Regional LSTM"


def mean_scores(frames: list[pd.DataFrame]) -> pd.DataFrame:
    s = [score_table(f).set_index("well_id") for f in frames]
    out = s[0][["rmse_pers", "n"]].copy()
    out["rmse"] = np.mean([x.rmse for x in s], axis=0)
    out["skill"] = 1 - out.rmse / out.rmse_pers
    return out.reset_index()


def per_well_rnn(rnn: Path, wells, h: int, model: str, variant: str, choices: list[tuple]) -> pd.DataFrame:
    """choices: list of (hidden, window, lam). Picks per well the choice with the lowest seed-mean val loss."""
    rows = []
    for w in wells:
        best = None
        for hd, win, lam in choices:
            tags = [f"{model}_{variant}" + (f"_lam{lam:g}" if lam is not None else "") + f"_h{hd}_w{win}_H{h}_s{s}"
                    for s in SEEDS]
            if not all((rnn / w / f"{t}.json").exists() for t in tags):
                continue
            js = [json.loads((rnn / w / f"{t}.json").read_text()) for t in tags]
            v = float(np.mean([j["best_val"] for j in js]))
            if best is None or v < best[0]:
                best = (v, tags, js, (hd, win, lam))
        if best is None:
            continue
        sc = mean_scores([pd.read_parquet(rnn / w / f"{t}.parquet") for t in best[1]]).iloc[0]
        rows.append(dict(well_id=w, rmse=sc.rmse, rmse_pers=sc.rmse_pers, skill=sc.skill, n=sc.n,
                         g1=all(j["g1"] for j in best[2]), choice=str(best[3])))
    return pd.DataFrame(rows)


def regional(reg: Path, h: int) -> tuple[pd.DataFrame | None, dict]:
    best = None
    for hd, win, lr in REG_CONFIGS:
        dirs = [reg / f"lstm_h{hd}_w{win}_lr{lr:g}_H{h}_s{s}" for s in SEEDS]
        if not all((d / "run.json").exists() for d in dirs):
            continue
        v = float(np.mean([json.loads((d / "run.json").read_text())["best_val"] for d in dirs]))
        if best is None or v < best[0]:
            best = (v, dirs, (hd, win, lr))
    if best is None:
        return None, {}
    frames = [pd.concat([pd.read_parquet(f) for f in sorted(d.glob("YN*.parquet"))]) for d in best[1]]
    return mean_scores(frames), dict(hidden=best[2][0], window=best[2][1], lr=best[2][2], val=round(best[0], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    R = Path(a.results)
    wells = pd.read_csv(Path(a.data) / "wells.csv").set_index("well_id")
    station = wells.station_name.str.split("_").str[0]
    test = wells[wells.role == "test"]
    y0 = test.date_start.str[:4].astype(int)
    groups = {"all": test.index, "long record": test.index[y0 <= 2012], "short record": test.index[y0 >= 2019]}

    summary, perwell = {}, []
    for h in HORIZONS:
        models: dict[str, pd.DataFrame] = {}
        meta = {}
        for label, mode in ((A, "A"), (AB, "B")):
            fs = [R / "preds" / f"gwfmkr_s{s}_{mode}_H{h}_test_test.parquet" for s in (0, 1, 2)]
            if all(f.exists() for f in fs):
                models[label] = mean_scores([pd.read_parquet(f) for f in fs])
        for label, mode in ((Z, "A"), (ZB, "B")):
            f = R / "preds" / f"zeroshot_{mode}_H{h}_test_test.parquet"
            if f.exists():
                models[label] = score_table(pd.read_parquet(f))
        tw = list(test.index)
        for label, model, var, ch in ((PW, "lstm", "plain", [(hd, win, None) for hd, win in CONFIGS]),
                                      (PW0, "lstm", "plain", [(64, 120, None)]),
                                      (GRU, "gru", "plain", [(64, 120, None)]),
                                      (EQ, "lstm", "ode", [(64, 120, l) for l in LAMS])):
            d = per_well_rnn(R / "rnn", tw, h, model, var, ch)
            if len(d):
                models[label] = d
                meta[label] = dict(g1_pass=int(d.g1.sum()), choices=d.choice.value_counts().to_dict())
        reg, rmeta = regional(R / "rnn_regional", h)
        if reg is not None:
            models[REG] = reg
            meta[REG] = rmeta

        # identical scoring days for every forecaster
        ref = models[Z].set_index("well_id")[["n", "rmse_pers"]]
        for label, d in models.items():
            d2 = d.set_index("well_id").loc[ref.index]
            assert (d2.n == ref.n).all() and np.allclose(d2.rmse_pers, ref.rmse_pers), f"scoring days differ: {label} H{h}"
        hs = {"meta": meta, "models": {}}
        for label, s in models.items():
            s = s.set_index("well_id")
            hs["models"][label] = {}
            for g, idx in groups.items():
                ss = s.loc[s.index.intersection(idx)]
                hs["models"][label][g] = dict(n=int(len(ss)), median_skill=round(float(ss.skill.median()), 3),
                                              mean_rmse=round(float(ss.rmse.mean()), 3),
                                              beats_persistence=int((ss.skill > 0).sum()))
            perwell.append(s.reset_index()[["well_id", "rmse", "rmse_pers", "skill"]].assign(horizon=h, model=label))
        pr = {}
        for m1, m2 in [(A, Z), (AB, ZB), (A, PW), (A, REG), (A, GRU), (A, EQ), (Z, PW), (Z, GRU), (Z, REG), (REG, PW), (EQ, PW0)]:
            if m1 not in models or m2 not in models:
                continue
            a1, a2 = models[m1].set_index("well_id"), models[m2].set_index("well_id")
            for g, idx in groups.items():
                c = a1.index.intersection(a2.index).intersection(idx)
                d = a1.loc[c, "rmse"] - a2.loc[c, "rmse"]
                ds = d.groupby(station.loc[c]).mean()
                pr.setdefault(f"{m1} | {m2}", {})[g] = dict(
                    n=int(len(c)), first_better=int((d < 0).sum()),
                    median_skill_diff=round(float((a1.loc[c, "skill"] - a2.loc[c, "skill"]).median()), 3),
                    p=float(wilcoxon(d).pvalue) if len(c) > 5 and (d != 0).any() else None,
                    n_stations=int(len(ds)), stations_first_better=int((ds < 0).sum()),
                    p_station=float(wilcoxon(ds).pvalue) if len(ds) > 5 and (ds != 0).any() else None)
        hs["pairs"] = pr
        summary[f"H{h}"] = hs

    (R / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    pd.concat(perwell).to_csv(R / "perwell_scores.csv", index=False)
    lines = ["# Test-station benchmark"]
    for h, hs in summary.items():
        lines.append(f"\n## {h}\n\n| Model | n | Median skill | Mean RMSE (m) | Beats persistence |\n|---|---|---|---|---|")
        for label, row in hs["models"].items():
            al = row["all"]
            lines.append(f"| {label} | {al['n']} | {al['median_skill']} | {al['mean_rmse']} | {al['beats_persistence']} |")
        lines.append(f"\nSelections: {json.dumps(hs['meta'], ensure_ascii=False)}\n\nPaired (first better / n, median skill diff, p):")
        for k, v in hs["pairs"].items():
            lines.append(f"- {k}: " + "; ".join(f"{g} {x['first_better']}/{x['n']}, {x['median_skill_diff']:+.3f}, p={x['p']:.2g}"
                                                for g, x in v.items() if x["p"] is not None))
    (R / "summary.md").write_text("\n".join(lines) + "\n")
    print((R / "summary.md").read_text())


if __name__ == "__main__":
    main()
