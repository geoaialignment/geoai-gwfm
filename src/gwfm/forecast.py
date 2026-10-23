"""Groundwater-level forecasting with Chronos-2 and the GeoAI-GWFM-KR adapter.

The past water level is the target and daily rainfall is a covariate. Future rainfall is optional:
pass a forecast or a scenario when one exists, otherwise the model forecasts from past data alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

BASE_MODEL = "amazon/chronos-2"
CONTEXT_DAYS = 1095
QUANTILES = (0.1, 0.5, 0.9)


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(adapter: str | Path | None = None, device: str | None = None):
    """Load Chronos-2, optionally with a LoRA adapter (a local folder or a Hugging Face repo id).

    The base weights are always fetched from the official amazon/chronos-2 repository.
    """
    from chronos import Chronos2Pipeline

    device = device or default_device()
    if adapter is None:
        return Chronos2Pipeline.from_pretrained(BASE_MODEL, device_map=device)
    from peft import PeftModel

    base = Chronos2Pipeline.from_pretrained(BASE_MODEL, device_map=device)
    model = PeftModel.from_pretrained(base.model, str(adapter)).merge_and_unload()
    return Chronos2Pipeline(model=model)


@dataclass
class ForecastRequest:
    level: np.ndarray  # past daily levels (m), NaN allowed for gaps
    rain: np.ndarray | None = None  # past daily rainfall (mm), same length as level
    future_rain: np.ndarray | None = None  # rainfall for the forecast days (mm), length = horizon


def _input(req: ForecastRequest, horizon: int, context_days: int) -> dict:
    lev = np.asarray(req.level, dtype=float)[-context_days:]
    item: dict = {"target": lev}
    if req.rain is not None:
        rain = np.nan_to_num(np.asarray(req.rain, dtype=float)[-context_days:], nan=0.0)
        if len(rain) != len(lev):
            raise ValueError("rain must have the same length as level")
        item["past_covariates"] = {"rain": rain}
        if req.future_rain is not None:
            fr = np.nan_to_num(np.asarray(req.future_rain, dtype=float), nan=0.0)
            if len(fr) != horizon:
                raise ValueError(f"future_rain must have {horizon} values")
            item["future_covariates"] = {"rain": fr}
    return item


def predict(pipe, requests: Sequence[ForecastRequest], horizon: int, context_days: int = CONTEXT_DAYS,
            quantiles: Sequence[float] = QUANTILES, batch_size: int = 256) -> np.ndarray:
    """Return forecasts of shape (n_requests, len(quantiles), horizon) in metres.

    Requests with and without future rainfall can be mixed; they are run as separate batches.
    """
    out = np.empty((len(requests), len(quantiles), horizon))
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(requests):
        key = (r.rain is not None, r.future_rain is not None)
        groups.setdefault(key, []).append(i)
    qidx = [pipe.quantiles.index(q) for q in quantiles]
    for idx in groups.values():
        inputs = [_input(requests[i], horizon, context_days) for i in idx]
        preds = pipe.predict(inputs, prediction_length=horizon, batch_size=batch_size, context_length=context_days)
        for j, i in enumerate(idx):
            out[i] = preds[j][0, qidx].float().cpu().numpy()
    return out
