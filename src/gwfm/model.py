"""Locate the GeoAI-GWFM-KR adapter and build forecasts for a user's well."""
from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import ADAPTER_MIRROR_URL, ADAPTER_NAME, ADAPTER_SHA256, ADAPTER_URL
from .forecast import CONTEXT_DAYS, ForecastRequest, load_pipeline, predict

CACHE = Path(os.environ.get("GWFM_CACHE", Path.home() / ".cache" / "gwfm"))
_PIPES: dict[str, object] = {}


def resolve_adapter(adapter: str | None = None) -> Path:
    """Adapter folder: an explicit path, $GWFM_ADAPTER, the local cache, or a download from geoaialign.ai."""
    for cand in (adapter, os.environ.get("GWFM_ADAPTER")):
        if cand:
            p = Path(cand).expanduser()
            if (p / "adapter_config.json").exists():
                return p
            raise FileNotFoundError(f"No adapter_config.json in {p}")
    target = CACHE / ADAPTER_NAME
    if (target / "adapter_config.json").exists():
        return target
    CACHE.mkdir(parents=True, exist_ok=True)
    zpath = CACHE / f"{ADAPTER_NAME}.zip"
    urls = [os.environ["GWFM_ADAPTER_URL"]] if os.environ.get("GWFM_ADAPTER_URL") else [ADAPTER_URL, ADAPTER_MIRROR_URL]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, zpath)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"Could not download the adapter from {urls}; download it manually and set GWFM_ADAPTER")
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if ADAPTER_SHA256 and digest != ADAPTER_SHA256:
        zpath.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}")
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise RuntimeError(f"Unsafe path in archive: {name}")
        z.extractall(CACHE)
    zpath.unlink()
    return target


def get_pipeline(model: str = "adapter", adapter: str | None = None):
    """model = 'adapter' (GeoAI-GWFM-KR) or 'zeroshot' (Chronos-2 without the adapter)."""
    key = f"{model}:{adapter}"
    if key not in _PIPES:
        _PIPES[key] = load_pipeline(resolve_adapter(adapter) if model == "adapter" else None)
    return _PIPES[key]


@dataclass
class Forecast:
    table: pd.DataFrame  # date, q10, median, q90 (m)
    mode: str  # 'future rain given', 'past rain only' or 'level only'
    n_context_days: int


def forecast_well(series: pd.DataFrame, horizon: int, future_rain=None, model: str = "adapter",
                  adapter: str | None = None) -> Forecast:
    """series: daily frame with level (NaN for gaps) and rain (NaN when no rainfall was supplied)."""
    pipe = get_pipeline(model, adapter)
    hist = series.iloc[-CONTEXT_DAYS:]
    has_rain = hist["rain"].notna().any()
    if future_rain is not None:
        if not has_rain:
            raise ValueError("Future rainfall needs past rainfall in the input table as well.")
        future_rain = np.asarray(future_rain, dtype=float)
        if len(future_rain) != horizon:
            raise ValueError(f"future_rain has {len(future_rain)} values; the horizon is {horizon} days.")
    req = ForecastRequest(level=hist["level"].to_numpy(float),
                          rain=hist["rain"].fillna(0).to_numpy(float) if has_rain else None,
                          future_rain=future_rain)
    q = predict(pipe, [req], horizon)[0]
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    table = pd.DataFrame({"date": dates.date.astype(str), "q10": q[0], "median": q[1], "q90": q[2]})
    mode = "future rain given" if future_rain is not None else ("past rain only" if has_rain else "level only")
    return Forecast(table=table.round(3), mode=mode, n_context_days=int(len(hist)))
