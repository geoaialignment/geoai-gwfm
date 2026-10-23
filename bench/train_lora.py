"""Train the GeoAI-GWFM-KR LoRA adapter on the training wells.

Each training well enters twice: once with rainfall known for the forecast days and once with
rainfall as a past-only covariate, so one adapter serves forecasts with and without future rainfall.
Only data up to --end are used; test wells are never seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwfm.forecast import BASE_MODEL, CONTEXT_DAYS, default_device  # noqa: E402

from common import load_split, load_well, model_level  # noqa: E402

LORA_TARGETS = ["self_attention.q", "self_attention.k", "self_attention.v", "self_attention.o",
                "output_patch_embedding.output_layer"]


def series_dicts(data: Path, wells: list[str], end: str) -> list[dict]:
    out = []
    for w in wells:
        df = load_well(data, w).loc[:end]
        lev = model_level(df)
        keep = np.flatnonzero(np.isfinite(lev))
        if len(keep) < 365:
            continue
        df, lev = df.iloc[keep[0]:], lev[keep[0]:]
        out.append({"target": lev.astype(np.float32), "past_covariates": {"rain": df["rain"].to_numpy(np.float32)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--n-wells", type=int, default=0, help="random subset of training wells (0 = all)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts
    from peft import LoraConfig

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    wells = load_split(a.data)["train_wells"]
    if a.n_wells:
        wells = sorted(np.random.default_rng(a.subset_seed).choice(wells, a.n_wells, replace=False).tolist())
    dicts = series_dicts(Path(a.data), wells, a.end)
    inputs = (from_list_of_dicts(dicts, a.horizon, known_covariates_names=["rain"])
              + from_list_of_dicts(dicts, a.horizon))

    pipe = Chronos2Pipeline.from_pretrained(BASE_MODEL, device_map=default_device())
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=LORA_TARGETS)
    out = Path(a.out)
    t = time.time()
    ft = pipe.fit(inputs, prediction_length=a.horizon, finetune_mode="lora", lora_config=lora,
                  context_length=CONTEXT_DAYS, learning_rate=a.lr, num_steps=a.steps, batch_size=a.batch,
                  output_dir=out / "trainer", finetuned_ckpt_name="adapter", min_past=180,
                  remove_printer_callback=True, seed=a.seed, logging_steps=100)
    n_tr, n_all = ft.model.get_nb_trainable_parameters()
    meta = dict(vars(a), wells=wells, n_series=len(dicts), trainable_params=int(n_tr), total_params=int(n_all),
                seconds=round(time.time() - t), base_model=BASE_MODEL, lora_targets=LORA_TARGETS)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"{len(dicts)} series, {n_tr} trainable of {n_all}, {meta['seconds']}s -> {out / 'trainer' / 'adapter'}")


if __name__ == "__main__":
    main()
