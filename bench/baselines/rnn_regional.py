"""Regional recursive LSTM: one network trained on all training wells, evaluated on the test stations.

Same information as GeoAI-GWFM-KR: the training wells up to 2021 (validation on 2022), the observed levels
with gaps (filled causally, with an observed/filled channel, as in rnn.py), and past and forecast-day
rainfall. Features are standardized per well with that well's statistics up to 2021, the network predicts
the daily change in units of each well's daily-change standard deviation, and losses use observed days only.
One run = one configuration (hidden size, window) x one horizon x one seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, load_well, origins  # noqa: E402
from rnn import anchored, features, trim, window  # noqa: E402


class RegionalNet(nn.Module):
    def __init__(self, n_past: int, n_exo: int, hidden: int):
        super().__init__()
        self.rnn = nn.LSTM(n_past, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_exo, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, P, E, ha, mu, sd, dsd):
        out, state = self.rnn(P)
        h, lev, preds = out[:, -1], ha, []
        one = torch.ones_like(ha)[:, None]
        for k in range(E.size(1)):
            d = self.head(torch.cat([h, E[:, k]], -1)).squeeze(-1) * dsd
            lev = lev + d
            preds.append(lev)
            x = torch.cat([((lev - mu) / sd)[:, None], (d / dsd)[:, None], one, E[:, k]], -1)[:, None]
            out, state = self.rnn(x, state)
            h = out[:, -1]
        return torch.stack(preds, 1)


def well_block(data, w, sp, W, K):
    df = trim(load_well(data, w))
    f = features(df, sp["train_end"])
    idx = df.index
    i_tr = idx.searchsorted(pd.Timestamp(sp["train_end"]), side="right")
    i_v0, i_v1 = idx.searchsorted(pd.Timestamp(sp["val"][0])), idx.searchsorted(pd.Timestamp(sp["val"][1]), side="right")
    return dict(df=df, y=f["y_ff"], y_obs=f["y_obs"], F_past=f["F_past"], F_exo=f["F_exo"], norm=f["norm"],
                o_tr=anchored(f["y_obs"], 0, i_tr, W, K), o_va=anchored(f["y_obs"], i_v0, i_v1, W, K))


def make_batch(blocks, pairs, W, K):
    P, E, ha, mu, sd, dsd, T = [], [], [], [], [], [], []
    for b, o in pairs:
        B = blocks[b]
        P.append(window(B["F_past"], o, W))
        E.append(B["F_exo"][o:o + K])
        ha.append(B["y"][o - 1])
        T.append(B["y_obs"][o:o + K])
        n = B["norm"]
        mu.append(n["lev_mu"]); sd.append(n["lev_sd"]); dsd.append(n["dlev_sd"])
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return f(np.stack(P)), f(np.stack(E)), f(ha), f(mu), f(sd), f(dsd), f(np.stack(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples-per-epoch", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    K, W = a.horizon, a.window
    tag = f"lstm_h{a.hidden}_w{W}_lr{a.lr:g}_H{K}_s{a.seed}"
    out = Path(a.out) / tag
    if (out / "run.json").exists():
        return
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    sp = load_split(a.data)
    blocks = [well_block(a.data, w, sp, W, K) for w in sp["train_wells"]]
    blocks = [b for b in blocks if len(b["o_tr"]) > 0]
    tr_pairs = np.array([(i, o) for i, b in enumerate(blocks) for o in b["o_tr"]])
    va_pairs = np.array([(i, o) for i, b in enumerate(blocks) for o in b["o_va"]])
    va_pairs = va_pairs[rng.choice(len(va_pairs), min(8000, len(va_pairs)), replace=False)]
    lead = torch.sqrt(torch.arange(1, K + 1, dtype=torch.float32))
    net = RegionalNet(blocks[0]["F_past"].shape[1], blocks[0]["F_exo"].shape[1], a.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-5)

    def loss_of(batch):
        P, E, ha, mu, sd, dsd, T = batch
        pred = net(P, E, ha, mu, sd, dsd)
        m = torch.isfinite(T)
        err = ((pred - torch.nan_to_num(T)) / dsd[:, None]) / lead
        return (err[m] ** 2).mean()

    Vb = [make_batch(blocks, va_pairs[i:i + 512], W, K) for i in range(0, len(va_pairs), 512)]
    best, state, noimp, ep = 1e18, None, 0, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        sel = tr_pairs[rng.choice(len(tr_pairs), min(a.samples_per_epoch, len(tr_pairs)), replace=False)]
        for i in range(0, len(sel), 256):
            opt.zero_grad()
            loss = loss_of(make_batch(blocks, sel[i:i + 256], W, K))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_of(b).item() for b in Vb]))
        sch.step(vl)
        if vl < best - 1e-6:
            best, noimp = vl, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            noimp += 1
        if noimp >= a.patience:
            break
    net.load_state_dict(state)
    net.eval()

    out.mkdir(parents=True, exist_ok=True)
    for w in sp["test_wells"]:
        B = well_block(a.data, w, sp, W, K)
        df = B["df"]
        o_te = origins(df, sp["test"][0], sp["test"][1], K)
        rows = []
        with torch.no_grad():
            for i in range(0, len(o_te), 256):
                oo = o_te[i:i + 256]
                P, E, ha, mu, sd, dsd, _ = make_batch([B], [(0, o) for o in oo], W, K)
                pr = net(P, E, ha, mu, sd, dsd).numpy().astype(np.float64)
                for j, o in enumerate(oo):
                    rows.append(pd.DataFrame(dict(well_id=w, origin=df.index[o], lead=np.arange(1, K + 1), pred=pr[j],
                                                  pers=B["y"][o - 1], obs=df["level_raw"].to_numpy(float)[o:o + K],
                                                  ok=df["eval_ok"].to_numpy()[o:o + K])))
        pd.concat(rows).to_parquet(out / f"{w}.parquet", index=False)
    res = dict(hidden=a.hidden, window=W, lr=a.lr, horizon=K, seed=a.seed, epochs=ep, best_val=best,
               n_train_wells=len(blocks), n_train_samples=int(len(tr_pairs)), seconds=round(time.time() - t0))
    (out / "run.json").write_text(json.dumps(res))
    print(tag, f"val={best:.4f} epochs={ep} {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
