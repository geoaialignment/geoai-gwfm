"""Forecast figure: recent observed levels, forecast median with the 10-90% band, rainfall bars."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_forecast(series: pd.DataFrame, forecasts: dict[str, pd.DataFrame], path: str | Path,
                  future_rain: dict[str, list] | None = None, history_days: int = 180, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = series.iloc[-history_days:]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    ax.plot(hist.index, hist["level"], color="black", lw=1.2, label="Observed")
    colors = ["#1f5fa8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910"]
    for i, (name, t) in enumerate(forecasts.items()):
        d = pd.to_datetime(t["date"])
        c = colors[i % len(colors)]
        ax.plot(d, t["median"], color=c, lw=1.6, label=f"{name} (median)")
        ax.fill_between(d, t["q10"], t["q90"], color=c, alpha=0.18, lw=0)
    ax.set_ylabel("Groundwater level (m)", fontweight="bold")
    ax.grid(alpha=0.3)
    if hist["rain"].notna().any():
        ax2 = ax.twinx()
        ax2.bar(hist.index, hist["rain"], color="#7fb3d5", width=1.0, alpha=0.6)
        for i, (name, r) in enumerate((future_rain or {}).items()):
            d = pd.to_datetime(list(forecasts.values())[0]["date"])
            ax2.bar(d, r, color=colors[i % len(colors)], width=1.0, alpha=0.35)
        ax2.set_ylabel("Rainfall (mm/day)", fontweight="bold")
        ax2.set_ylim(top=max(1.0, float(hist["rain"].max())) * 3)
        ax2.invert_yaxis()
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
