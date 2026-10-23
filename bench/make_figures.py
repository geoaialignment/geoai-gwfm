"""Release figures from the benchmark results (Times New Roman, 300 dpi).

Fig. 1  Skill against persistence on the 44 test wells, by forecaster and horizon.
Fig. 2  (a) Per-well RMSE with and without the adapter; (b) test skill against the number of training wells.
Fig. 3  Forecasts at one test well, chosen as the well with the median gain of the adapter over the per-well
        LSTM (30-day, future rain).
Fig. 4  Skill by record length: wells with long records and wells installed in 2019 or later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_well, score_table

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.labelweight": "bold",
                     "axes.labelsize": 11, "savefig.dpi": 300})
BLUE, ORANGE, GREY, GREEN, RED = "#1f5fa8", "#d68910", "#7f8c8d", "#2e8b57", "#c0392b"
ORDER = [("Regional LSTM", "Regional LSTM", "#b5bfc0"),
         ("Per-well LSTM", "Per-well LSTM", GREY),
         ("Per-well GRU", "Per-well GRU", GREY),
         ("Per-well LSTM + equation", "Per-well LSTM\n+ physics loss", GREY),
         ("Chronos-2 alone, future rain", "Chronos-2\nzero-shot", ORANGE),
         ("GeoAI-GWFM-KR, future rain", "GeoAI-GWFM-KR", BLUE)]


def fig1(pw: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
    for ax, h in zip(axes, (7, 30, 90)):
        d = pw[pw.horizon == h]
        data = [d[d.model == m].skill.clip(lower=-1).to_numpy() for m, _, _ in ORDER]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
        for patch, (_, _, c) in zip(bp["boxes"], ORDER):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for i, x in enumerate(data):
            ax.scatter(np.full(len(x), i + 1) + np.random.default_rng(i).uniform(-0.18, 0.18, len(x)), x, s=6,
                       color="black", alpha=0.45, zorder=3)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(1, len(ORDER) + 1), [o[1] for o in ORDER], rotation=45, ha="right", fontsize=8.5)
        ax.set_title(f"{h}-day forecast", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Skill against persistence")
    fig.tight_layout()
    fig.savefig(out / "Figure1_skill.png")
    plt.close(fig)


def fig2(pw: pd.DataFrame, preds: Path, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.9))
    d = pw[pw.horizon == 30].pivot(index="well_id", columns="model", values="rmse")
    x, y = d["Chronos-2 alone, future rain"], d["GeoAI-GWFM-KR, future rain"]
    a.scatter(x, y, s=16, color=BLUE, alpha=0.8)
    lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2
    a.plot([lo, hi], [lo, hi], color="black", lw=0.8)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("RMSE, Chronos-2 zero-shot (m)")
    a.set_ylabel("RMSE, GeoAI-GWFM-KR (m), seed mean")
    a.set_title(f"(a) 30-day, future rain: adapter better at {(y < x).sum()} of {len(x)} wells", fontsize=9.5)
    a.grid(alpha=0.3, which="both")
    sc = json.loads((preds.parent / "scaling.json").read_text())["H30"]
    n_ = [int(k) for k in sc]
    b.plot(n_, list(sc.values()), "o-", color=BLUE)
    b.set_xlabel("Training wells in the adapter (0 = zero-shot)")
    b.set_ylabel("Median skill, 30-day forecast")
    b.set_title("(b) Test skill and number of training wells", fontsize=9.5)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "Figure2_adapter.png")
    plt.close(fig)


def fig3(pw: pd.DataFrame, data: Path, preds: Path, rnn: Path, out: Path) -> str:
    d = pw[pw.horizon == 30].pivot(index="well_id", columns="model", values="rmse")
    gain = d["Per-well LSTM"] - d["GeoAI-GWFM-KR, future rain"]
    w = gain.sub(gain.median()).abs().idxmin()
    df = load_well(data, w).loc["2022-10-01":"2024-12-31"]
    g = pd.read_parquet(preds / "gwfmkr_s0_A_H30_test_test.parquet").query("well_id == @w")
    z = pd.read_parquet(preds / "zeroshot_A_H30_test_test.parquet").query("well_id == @w")
    from analyze import CONFIGS, per_well_rnn
    hd, win, _ = eval(per_well_rnn(rnn, [w], 30, "lstm", "plain", [(a, b, None) for a, b in CONFIGS]).choice.iloc[0])
    lstm = pd.concat([pd.read_parquet(f) for f in sorted((rnn / w).glob(f"lstm_plain_h{hd}_w{win}_H30_s*.parquet"))])
    lstm = lstm.groupby(["origin", "lead"], as_index=False).pred.mean()
    origins = sorted(g.origin.unique())[::4]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(df.index, df.level_raw.where(df.eval_ok), color="black", lw=1.0, label="Observed")
    ax.set_title(f"Test well {w}: 30-day forecasts from every fourth origin, rainfall of the forecast days given", fontsize=10)
    for i, o in enumerate(origins):
        for frame, c, lab in ((lstm, GREY, "Per-well LSTM"), (z, ORANGE, "Chronos-2 zero-shot"), (g, BLUE, "GeoAI-GWFM-KR")):
            f = frame[frame.origin == o].sort_values("lead")
            t = pd.Timestamp(o) + pd.to_timedelta(f.lead - 1, unit="D")
            ax.plot(t, f.pred, color=c, lw=1.3, label=lab if i == 0 else None)
    ax.set_ylabel("Groundwater level (m)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(df.index, df.rain, color="#7fb3d5", width=1.0, alpha=0.6)
    ax2.set_ylim(0, df.rain.max() * 3)
    ax2.invert_yaxis()
    ax2.set_ylabel("Rainfall (mm/day)")
    ax.axvline(pd.Timestamp("2023-01-01"), color="black", lw=0.6, ls=":")
    ax.legend(loc="lower left", frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "Figure3_example.png")
    plt.close(fig)
    return w


def fig4(pw: pd.DataFrame, wells: pd.DataFrame, out: Path):
    y0 = wells.set_index("well_id").date_start.str[:4].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    sel = [ORDER[0], ORDER[1], ORDER[4], ORDER[5]]
    for ax, (name, mask) in zip(axes, (("Records from 2012 or earlier", y0 <= 2012), ("Wells installed 2019 or later", y0 >= 2019))):
        ids = y0[mask].index.intersection(pw.well_id.unique())
        d = pw[(pw.horizon == 30) & pw.well_id.isin(ids)]
        data = [d[d.model == m].skill.clip(lower=-1).to_numpy() for m, _, _ in sel]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
        for patch, (_, _, c) in zip(bp["boxes"], sel):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(1, len(sel) + 1), [o[1] for o in sel], rotation=45, ha="right", fontsize=8.5)
        ax.set_title(f"{name} (n = {len(ids)})", fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Skill, 30-day forecast")
    fig.tight_layout()
    fig.savefig(out / "Figure4_record_length.png")
    plt.close(fig)


def hero(data: Path, preds: Path, w: str, out: Path, theme: str):
    """Wide banner for the release page: observed level, GeoAI-GWFM-KR 30-day forecasts and rainfall."""
    ink, band, water, rain, grid = (("#12343B", "#1F5FA8", "#1F5FA8", "#8FB8DE", "#C9D3D2") if theme == "light"
                                    else ("#E3ECEA", "#6FA8E8", "#6FA8E8", "#3E6E95", "#2C474B"))
    df = load_well(data, w).loc["2023-01-01":"2024-12-31"]
    g = pd.read_parquet(preds / "gwfmkr_s0_A_H30_test_test.parquet").query("well_id == @w")
    with plt.rc_context({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": grid,
                         "xtick.color": ink, "ytick.color": ink, "axes.labelcolor": ink}):
        fig, ax = plt.subplots(figsize=(12, 3.6), dpi=200)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        for o in sorted(g.origin.unique())[::4]:
            f = g[g.origin == o].sort_values("lead")
            t = pd.Timestamp(o) + pd.to_timedelta(f.lead - 1, unit="D")
            ax.fill_between(t, f.q10, f.q90, color=band, alpha=0.18, lw=0)
            ax.plot(t, f.pred, color=water, lw=2.0)
        ax.plot(df.index, df.level_raw.where(df.eval_ok), color=ink, lw=1.0)
        ax.set_ylabel("Level (m)")
        ax.grid(axis="y", color=grid, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax2 = ax.twinx()
        ax2.bar(df.index, df.rain, color=rain, width=1.0)
        ax2.set_ylim(0, df.rain.max() * 4.5)
        ax2.invert_yaxis()
        ax2.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"hero_{theme}.png", transparent=True)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pw = pd.read_csv(R / "perwell_scores.csv")
    wells = pd.read_csv(Path(a.data) / "wells.csv")
    fig1(pw, out)
    fig2(pw, R / "preds", out)
    w = fig3(pw, Path(a.data), R / "preds", R / "rnn", out)
    fig4(pw, wells, out)
    for theme in ("light", "dark"):
        hero(Path(a.data), R / "preds", w, out, theme)
    (out / "figure_notes.json").write_text(json.dumps({"figure3_well": w}, indent=1))
    print("figures ->", out, "example well", w)


if __name__ == "__main__":
    main()
