# GeoAI-GWFM-KR

**Groundwater-level forecasting for Korean monitoring wells, adapted from Amazon Chronos-2.**
GeoAI Alignment, Inc. · [geoaialign.ai](https://geoaialign.ai) · v1.0 (2026)

GeoAI-GWFM-KR forecasts daily groundwater levels from a well's past levels, with daily rainfall as a covariate.
It runs on the Chronos-2 time-series foundation model (Amazon, Apache-2.0) and adds a LoRA adapter that
GeoAI Alignment trained on Korean National Groundwater Monitoring Network wells. No training is needed for a
new well: give the model the well's record and, if available, a rainfall forecast or scenario.

This repository contains:

| Folder | Content |
|---|---|
| `src/gwfm/` | Python package: forecasting API, command line tool, and a forecasting agent (MCP server) |
| `bench/` | The Yeongnam groundwater benchmark: data preparation, station split, forecasters, per-well and regional LSTM baselines, scoring |
| `examples/` | A synthetic example well and a script that makes it |
| `tests/` | End-to-end test of the agent |

The adapter weights (about 5 MB) are downloaded from [geoaialign.ai](https://geoaialign.ai). The Chronos-2 base
weights are downloaded from the official [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) repository.

## Results

Benchmark on 46 wells at 30 monitoring stations held out before training (no station shared with
the 188 training wells), weekly forecasts for 2023-2024, rainfall of the forecast days given. All forecasters get the
same input variables (past levels, past and forecast-day rainfall) and training data up to 2021, choose their settings on
2022 and are averaged over three seeds; the candidate settings per model are listed in the technical note. Skill =
1 - RMSE / RMSE of persistence over days 1 to H (median over wells).

| Forecaster | 7 days | 30 days | 90 days |
|---|---|---|---|
| **GeoAI-GWFM-KR** | **0.41** | **0.48** | **0.52** |
| Chronos-2 alone (zero-shot) | 0.21 | 0.26 | 0.31 |
| Per-well LSTM (trained on each test well) | 0.37 | 0.30 | 0.33 |
| Per-well GRU (trained on each test well) | 0.41 | 0.33 | 0.37 |
| Per-well LSTM with a fluctuation-equation loss | 0.39 | 0.29 | 0.34 |
| Regional LSTM (same training wells as the adapter) | 0.25 | 0.20 | 0.24 |

GeoAI-GWFM-KR was more accurate than the per-well LSTM at 34, 39 and 40 of 46 wells at 7, 30
and 90 days. Chronos-2 alone, not trained on the groundwater records of this benchmark, showed no significant difference
from the per-well LSTM at 30 and 90 days (p = 0.084 and p = 0.49) and was less accurate at 7 days. Details: technical
notes at [geoaialign.ai/models/gwfm-kr](https://geoaialign.ai/models/gwfm-kr/) and [Zenodo](https://doi.org/10.5281/zenodo.23584923); the English
note is published as a preprint in the [ESS Open Archive](https://doi.org/10.22541/essoar.15011386/v1).

## Install

```bash
pip install "geoai-gwfm[agent] @ git+https://github.com/geoaialignment/geoai-gwfm"
gwfm download            # adapter into ~/.cache/gwfm (or set GWFM_ADAPTER to a local folder)
```

Python 3.10 or later. Runs on CPU, Apple silicon (MPS) or CUDA; a 90-day forecast of one well takes well under
a second on a laptop GPU.

## Forecast from the command line

The input table needs a date column, a water-level column (m) and a daily rainfall column (mm). Column names such as `date`, `level`, `rain` (or 날짜, 수위, 강수량) are detected.

```bash
gwfm forecast --csv examples/example_well.csv --horizon 30
gwfm forecast --csv my_well.csv --horizon 30 --future-rain rain_forecast.csv   # one value per forecast day
```

Outputs: a CSV with the 10%, 50% and 90% quantiles for each day, and a figure.

## Forecast from Python

```python
from gwfm.io import read_table, daily_series
from gwfm.model import forecast_well

series, info = daily_series(read_table("my_well.csv"))
fc = forecast_well(series, horizon=30, future_rain=None)   # or a list of 30 daily rainfall values
print(fc.table)
```

## Forecasting agent

`gwfm-mcp` is a local [Model Context Protocol](https://modelcontextprotocol.io) server. Add it to an MCP client
such as Claude Desktop or Claude Code:

```json
{"mcpServers": {"gwfm": {"command": "gwfm-mcp"}}}
```

If the adapter is kept in a folder of your choice, pass it to the server:
`{"mcpServers": {"gwfm": {"command": "gwfm-mcp", "env": {"GWFM_ADAPTER": "/path/to/GeoAI-GWFM-KR-v1.0"}}}}`

Then ask in plain language, for example: *"Here is wells/A01.csv with past levels and rainfall. Forecast the
next 30 days, and compare a dry month with the KMA rainfall outlook."* The agent has four tools:

| Tool | What it does |
|---|---|
| `inspect_series` | Reads the table, detects columns, reports dates, gaps and the last level |
| `forecast_groundwater` | Forecasts the next N days (with or without future rainfall), writes CSV and figure |
| `compare_scenarios` | Forecasts under several rainfall scenarios and compares the end-of-horizon levels |
| `model_info` | Model card: training data, tested accuracy, intended use, license |

When future rainfall is not supplied the forecast uses past data only. The tools never substitute observed
future rainfall.

## Benchmark

`bench/` reproduces the evaluation in the technical note from GIMS water levels and KMA weather:

```bash
python bench/prepare.py --levels levels.csv --weather weather.csv --stations stations.csv --out data
python bench/split.py --data data
python bench/run_chronos.py --data data --out results/preds --name zeroshot
python bench/train_lora.py --data data --out artifacts/final/seed0 --end 2021-12-31 --seed 0
python bench/run_chronos.py --data data --out results/preds --name gwfmkr_s0 --adapter artifacts/final/seed0/trainer/adapter
python bench/baselines/rnn.py --data data --out results/rnn --well YN008 --horizon 30 --hidden 128 --window 365
python bench/baselines/rnn_regional.py --data data --out results/rnn_regional --horizon 30 --lr 0.0001
python bench/analyze.py --data data --results results
```

`experiments/run_foundation_model.sh` runs the full foundation-model side (zero-shot, selection, three seeds, sensitivity check,
training-size curve).

The well list with test/training roles is in `bench/wells.csv`.

## License and citation

- Code in this repository: [PolyForm Noncommercial License 1.0.0](LICENSE). Redistribution must keep the
  license and the `Required Notice` line.
- Adapter weights, benchmark split and results, and the technical notes: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  Attribution is required; commercial use is not permitted.
- Chronos-2 remains under its own Apache-2.0 license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- For commercial licensing, contact GeoAI Alignment through [geoaialign.ai](https://geoaialign.ai).

Please cite GeoAI-GWFM-KR when you use it or its forecasts in reports or publications (see [CITATION.cff](CITATION.cff)):

> Park, E., Park, J., & Kim, T. (2026). GeoAI-GWFM-KR v1.0: groundwater-level forecasting for Korean monitoring wells, adapted from Chronos-2. GeoAI Alignment, Inc. https://doi.org/10.5281/zenodo.23584923

The technical note (preprint):

> Park, E., Park, J., & Kim, T. (2026). Forecasting groundwater levels at unseen wells without per-well training: GeoAI-GWFM-KR, a time-series foundation model adapted to the Korean monitoring network. ESS Open Archive. https://doi.org/10.22541/essoar.15011386/v1
