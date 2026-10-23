"""GeoAI-GWFM-KR forecasting agent as a local MCP server (stdio).

Add to an MCP client, for example Claude Desktop or Claude Code:
    {"mcpServers": {"gwfm": {"command": "gwfm-mcp"}}}
Then ask in plain language, e.g. "Forecast the next 30 days for wells/my_well.csv".
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import ADAPTER_NAME, MODEL_NAME, __version__
from .io import daily_series, read_table
from .model import forecast_well
from .plot import plot_forecast

INSTRUCTIONS = (
    "Forecasts daily groundwater levels from a well's past levels, with daily rainfall as an optional covariate. "
    "Inputs are a CSV/Excel file path or CSV text with a date column, a water-level column (m) and optionally a "
    "rainfall column (mm/day). Future rainfall for the forecast days is optional: a forecast or a scenario. "
    "Without it the model forecasts from past data only."
)
mcp = MCPServer(name="gwfm", title=f"{MODEL_NAME} groundwater forecaster", version=__version__,
                instructions=INSTRUCTIONS)

Source = Annotated[str | None, Field(description="Path to a CSV or Excel file on this computer")]
Text = Annotated[str | None, Field(description="CSV text with a header row, used instead of a file path")]
Horizon = Annotated[int, Field(ge=1, le=365, description="Number of days to forecast after the last observation")]
Model = Annotated[Literal["adapter", "zeroshot"], Field(
    description="'adapter' = GeoAI-GWFM-KR (Chronos-2 with the Korean groundwater adapter); "
                "'zeroshot' = Chronos-2 alone")]
OutDir = Annotated[str | None, Field(description="Folder for the forecast CSV and figure; a temporary folder if omitted")]


class Scenario(BaseModel):
    name: str = Field(description="Short label, e.g. 'forecast', 'wet', 'dry'")
    rain: list[float] = Field(description="Daily rainfall (mm) for each forecast day; length must equal horizon_days")


def _load(path, text):
    series, info = daily_series(read_table(path, text))
    return series, info


def _outdir(out_dir):
    d = Path(out_dir).expanduser() if out_dir else Path(tempfile.mkdtemp(prefix="gwfm_"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _summary(t, last_level):
    end = t.iloc[-1]
    return dict(last_observed_level=round(last_level, 3), end_date=end["date"],
                end_median=float(end["median"]), end_q10=float(end["q10"]), end_q90=float(end["q90"]),
                change_at_end=round(float(end["median"]) - last_level, 3),
                max_median=float(t["median"].max()), min_median=float(t["median"].min()))


@mcp.tool(title="Inspect a well series", annotations=ToolAnnotations(
    title="Inspect a well series", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def inspect_series(csv_path: Source = None, csv_text: Text = None) -> str:
    """Read a water-level table and report the detected columns, date range, gaps and last level.
    Use this before forecasting when the file layout is unknown. Does not forecast."""
    try:
        _, info = _load(csv_path, csv_text)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    return json.dumps(info, ensure_ascii=False, indent=1)


@mcp.tool(title="Forecast groundwater level", annotations=ToolAnnotations(
    title="Forecast groundwater level", readOnlyHint=False, destructiveHint=False, idempotentHint=True,
    openWorldHint=False))
def forecast_groundwater(horizon_days: Horizon, csv_path: Source = None, csv_text: Text = None,
                         future_rain: Annotated[list[float] | None, Field(
                             description="Daily rainfall (mm) for each forecast day from a weather forecast or a "
                                         "scenario; omit when unknown")] = None,
                         model: Model = "adapter", out_dir: OutDir = None) -> str:
    """Forecast daily groundwater levels for the next horizon_days after the last observation, with the
    10%, 50% and 90% quantiles. Writes the forecast CSV and a PNG figure and returns their paths with a
    summary. For several rainfall cases at once use compare_scenarios."""
    try:
        series, info = _load(csv_path, csv_text)
        fc = forecast_well(series, horizon_days, future_rain, model)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    d = _outdir(out_dir)
    stem = f"forecast_{horizon_days}d"
    csv_out = d / f"{stem}.csv"
    fc.table.to_csv(csv_out, index=False)
    fig = plot_forecast(series, {model: fc.table}, d / f"{stem}.png",
                        future_rain={"future rain": future_rain} if future_rain is not None else None,
                        title=f"{MODEL_NAME if model == 'adapter' else 'Chronos-2'} forecast, {fc.mode}")
    res = dict(model=ADAPTER_NAME if model == "adapter" else "amazon/chronos-2 (zero-shot)", mode=fc.mode,
               context_days=fc.n_context_days, summary=_summary(fc.table, info["last_level"]),
               forecast_csv=str(csv_out), figure=str(fig))
    if fc.mode != "future rain given":
        res["note"] = ("No rainfall was supplied for the forecast days. GeoAI-GWFM-KR is most accurate with a rainfall "
                       "forecast or scenario for those days (future_rain). Without it the gain over Chronos-2 alone is "
                       "smaller: in the benchmark it was significant at 30 and 90 days and not detected at 7 days.")
    if horizon_days <= 31:
        res["forecast"] = fc.table.to_dict(orient="records")
    else:
        res["forecast_every_7_days"] = fc.table.iloc[6::7].to_dict(orient="records")
    return json.dumps(res, ensure_ascii=False, indent=1)


@mcp.tool(title="Compare rainfall scenarios", annotations=ToolAnnotations(
    title="Compare rainfall scenarios", readOnlyHint=False, destructiveHint=False, idempotentHint=True,
    openWorldHint=False))
def compare_scenarios(horizon_days: Horizon, scenarios: list[Scenario], csv_path: Source = None,
                      csv_text: Text = None, include_no_rain_case: bool = True, model: Model = "adapter",
                      out_dir: OutDir = None) -> str:
    """Forecast the same well under several future-rainfall scenarios and compare the levels at the end of
    the horizon. Adds a case with zero future rainfall unless include_no_rain_case is false. Writes one CSV
    and one figure."""
    try:
        series, info = _load(csv_path, csv_text)
        cases = {s.name: s.rain for s in scenarios}
        if include_no_rain_case:
            cases.setdefault("no rain", [0.0] * horizon_days)
        tables = {n: forecast_well(series, horizon_days, r, model).table for n, r in cases.items()}
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    d = _outdir(out_dir)
    import pandas as pd

    long = pd.concat([t.assign(scenario=n) for n, t in tables.items()])
    csv_out = d / f"scenarios_{horizon_days}d.csv"
    long.to_csv(csv_out, index=False)
    fig = plot_forecast(series, tables, d / f"scenarios_{horizon_days}d.png", future_rain=cases,
                        title=f"Rainfall scenarios, {horizon_days} days")
    res = dict(model=ADAPTER_NAME if model == "adapter" else "amazon/chronos-2 (zero-shot)",
               scenarios={n: dict(total_rain_mm=round(float(sum(cases[n])), 1), **_summary(t, info["last_level"]))
                          for n, t in tables.items()},
               forecast_csv=str(csv_out), figure=str(fig))
    return json.dumps(res, ensure_ascii=False, indent=1)


@mcp.tool(title="Model information", annotations=ToolAnnotations(
    title="Model information", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def model_info() -> str:
    """Describe the model, its training data, tested accuracy, intended use and license."""
    from .card import CARD

    return CARD


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
