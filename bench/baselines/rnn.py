"""Per-well recursive LSTM/GRU baselines, optionally with a groundwater fluctuation-equation loss.

Adapted from the code of Park, Park & Kim (2026, submitted; Zenodo 10.5281/zenodo.23184760).
One run = one well x one model x one variant x one horizon x one seed. The network is trained on the
well's own record up to 2021-12-31, stopped early on 2022, and forecasts the test origins shared with
the Chronos-2 runs. It advances one day at a time: it predicts the daily change, adds it to the level and
feeds the predicted level back.

Inputs carry the same information as the foundation model: the observed levels with gaps (the series
Chronos-2 receives), and past and forecast-day rainfall. Gaps are filled causally by carrying the last
observation forward, and an observed/filled channel tells the network which days were observed. Losses
and validation use observed days only. No calendar features are used.

Penalty `ode`: ((h_k - h_{k-1}) - (g_r R_k - g_d (h_{k-1} - h_ref)))^2 / var(dh), the groundwater
fluctuation equation of Park & Parker (2008) with learned coefficients.
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
from common import load_split, load_well, origins  # noqa: E402


def trim(df: pd.DataFrame) -> pd.DataFrame:
    """Start the record at the first observed input level."""
    return df.iloc[df["level_in"].notna().to_numpy().argmax():]


def features(df: pd.DataFrame, train_end: str) -> dict:
    y_obs = df["level_in"].to_numpy(np.float64)                      # observed levels, NaN in gaps
    y_ff = pd.Series(y_obs).ffill().to_numpy(np.float64)            # last observation carried forward
    seen = np.isfinite(y_obs).astype(np.float64)
    tr = df.index <= pd.Timestamp(train_end)
    rain = df["rain"].to_numpy(np.float64)
    rs = np.column_stack([np.log1p(pd.Series(rain).rolling(k, min_periods=1).sum().to_numpy()) for k in (30, 90, 365)])
    exo = np.column_stack([rain, rs])
    z = lambda a: (a - a[tr].mean(0)) / (a[tr].std(0) + 1e-8)
    obs_tr = y_obs[tr][np.isfinite(y_obs[tr])]
    lev_mu, lev_sd = obs_tr.mean(), obs_tr.std() + 1e-8
    dlev = np.diff(y_ff, prepend=y_ff[0])
    d_ok = tr & (seen > 0) & (np.roll(seen, 1) > 0)
    dlev_sd = dlev[d_ok][1:].std() + 1e-8
    F_exo = np.nan_to_num(z(exo), nan=0.0).astype(np.float32)
    F_past = np.column_stack([(y_ff - lev_mu) / lev_sd, dlev / dlev_sd, seen, F_exo]).astype(np.float32)
    return dict(y_obs=y_obs, y_ff=y_ff, rain=rain, F_past=F_past, F_exo=F_exo,
                norm=dict(lev_mu=float(lev_mu), lev_sd=float(lev_sd), dlev_sd=float(dlev_sd)))


def anchored(y_obs: np.ndarray, lo: int, hi: int, W: int, K: int) -> np.ndarray:
    """Origins o in [lo, hi - K] with a full window and an observed level on day o - 1."""
    o = np.arange(max(lo, W + 1), hi - K + 1)
    return o[np.isfinite(y_obs[o - 1])]


def window(F_past: np.ndarray, o: int, W: int) -> np.ndarray:
    """Input window ending the day before origin o; days before the record start are zeros (observed flag 0)."""
    if o >= W:
        return F_past[o - W:o]
    return np.vstack([np.zeros((W - o, F_past.shape[1]), np.float32), F_past[:o]])


class Net(nn.Module):
    def __init__(self, kind: str, n_past: int, n_exo: int, hidden: int, norm: dict):
        super().__init__()
        R = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = R(n_past, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_exo, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.lev_mu, self.lev_sd, self.dlev_sd = norm["lev_mu"], norm["lev_sd"], norm["dlev_sd"]

    def forward(self, P, E, h_anchor):
        out, state = self.rnn(P)
        h = out[:, -1]
        lev = h_anchor
        preds = []
        one = torch.ones_like(h_anchor)[:, None]
        for k in range(E.size(1)):
            d = self.head(torch.cat([h, E[:, k]], -1)).squeeze(-1) * self.dlev_sd
            lev = lev + d
            preds.append(lev)
            x = torch.cat([((lev - self.lev_mu) / self.lev_sd)[:, None], (d / self.dlev_sd)[:, None], one, E[:, k]], -1)[:, None]
            out, state = self.rnn(x, state)
            h = out[:, -1]
        return torch.stack(preds, 1)


class ODEParams(nn.Module):
    def __init__(self, h_ref0: float):
        super().__init__()
        self.th_r = nn.Parameter(torch.tensor(float(np.log(0.004 / (0.05 - 0.004)))))
        self.th_d = nn.Parameter(torch.tensor(float(np.log(0.02 / (0.2 - 0.02)))))
        self.h_ref = nn.Parameter(torch.tensor(float(h_ref0)))

    def values(self):
        return 0.05 * torch.sigmoid(self.th_r), 0.2 * torch.sigmoid(self.th_d), self.h_ref


def ode_penalty(pred, h_anchor, rain, ode, dsd2):
    traj = torch.cat([h_anchor[:, None], pred], 1)
    gr, gd, href = ode.values()
    dh = traj[:, 1:] - traj[:, :-1]
    return ((dh - (gr * rain - gd * (traj[:, :-1] - href))) ** 2).mean() / dsd2


def masked_mse(pred, ha, T, lead_sd):
    """Mean squared error of the cumulative change, scaled per lead, over observed target days only."""
    m = torch.isfinite(T)
    err = ((pred - ha[:, None]) - torch.nan_to_num(T)) / lead_sd
    return (err[m] ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--well", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--variant", default="plain", choices=["plain", "ode"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    tag = (f"{a.model}_{a.variant}" + (f"_lam{a.lam:g}" if a.variant == "ode" else "")
           + f"_h{a.hidden}_w{a.window}_H{a.horizon}_s{a.seed}")
    out = Path(a.out) / a.well
    if (out / f"{tag}.json").exists():
        return
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    sp = load_split(a.data)
    df = trim(load_well(a.data, a.well))
    f = features(df, sp["train_end"])
    y_obs, y_ff, rain, F_past, F_exo, norm = f["y_obs"], f["y_ff"], f["rain"], f["F_past"], f["F_exo"], f["norm"]
    W, K = a.window, a.horizon
    idx = df.index
    i_tr = idx.searchsorted(pd.Timestamp(sp["train_end"]), side="right")
    i_va0, i_va1 = idx.searchsorted(pd.Timestamp(sp["val"][0])), idx.searchsorted(pd.Timestamp(sp["val"][1]), side="right")
    o_tr, o_va = anchored(y_obs, 0, i_tr, W, K), anchored(y_obs, i_va0, i_va1, W, K)
    if len(o_tr) < 64 or len(o_va) < 8:
        W = min(W, 60)
        o_tr, o_va = anchored(y_obs, 0, i_tr, W, K), anchored(y_obs, i_va0, i_va1, W, K)
    tgt = lambda os_: np.stack([y_obs[o:o + K] - y_ff[o - 1] for o in os_])
    lead_sd = torch.from_numpy((np.nanstd(tgt(o_tr), 0) + 1e-8).astype(np.float32))
    dsd2 = norm["dlev_sd"] ** 2

    def batch(os_):
        P = torch.from_numpy(np.stack([F_past[o - W:o] for o in os_]))
        E = torch.from_numpy(np.stack([F_exo[o:o + K] for o in os_]))
        ha = torch.from_numpy(y_ff[os_ - 1].astype(np.float32))
        R = torch.from_numpy(np.stack([rain[o:o + K] for o in os_]).astype(np.float32))
        T = torch.from_numpy(tgt(os_).astype(np.float32))
        return P, E, ha, R, T

    net = Net(a.model, F_past.shape[1], F_exo.shape[1], a.hidden, norm)
    ode = ODEParams(float(np.nanmin(y_obs[:i_tr]))) if a.variant == "ode" else None
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    opt_p = torch.optim.Adam(ode.parameters(), lr=1e-2) if ode else None
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-5)
    Vb = batch(o_va)
    best, state, pstate, noimp, ep = 1e18, None, None, 0, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = np.random.permutation(o_tr)
        for i in range(0, len(perm), 64):
            P, E, ha, R, T = batch(perm[i:i + 64])
            opt.zero_grad()
            if opt_p:
                opt_p.zero_grad()
            pred = net(P, E, ha)
            loss = masked_mse(pred, ha, T, lead_sd)
            if ode:
                loss = loss + a.lam * ode_penalty(pred, ha, R, ode, dsd2)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if opt_p:
                opt_p.step()
        net.eval()
        with torch.no_grad():
            P, E, ha, R, T = Vb
            vl = masked_mse(net(P, E, ha), ha, T, lead_sd).item()
        sch.step(vl)
        if vl < best - 1e-6:
            best, noimp = vl, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
            pstate = {k: v.clone() for k, v in ode.state_dict().items()} if ode else None
        else:
            noimp += 1
        if noimp >= a.patience:
            break
    net.load_state_dict(state)
    if ode:
        ode.load_state_dict(pstate)
    net.eval()

    # competence check (G1): lead-1 training RMSE below persistence, on observed days
    with torch.no_grad():
        P, E, ha, R, T = batch(o_tr)
        p1 = net(P, E, ha)[:, 0].numpy()
    ok1 = np.isfinite(y_obs[o_tr])
    g1 = bool(np.sqrt(((p1 - y_obs[o_tr])[ok1] ** 2).mean()) < np.sqrt(((y_ff[o_tr - 1] - y_obs[o_tr])[ok1] ** 2).mean()))

    o_te = origins(df, sp["test"][0], sp["test"][1], K)
    P = torch.from_numpy(np.stack([window(F_past, o, W) for o in o_te]))
    E = torch.from_numpy(np.stack([F_exo[o:o + K] for o in o_te]))
    ha = torch.from_numpy(y_ff[o_te - 1].astype(np.float32))
    with torch.no_grad():
        pr = net(P, E, ha).numpy().astype(np.float64)
    obs = df["level_raw"].to_numpy(float)
    ok = df["eval_ok"].to_numpy()
    lead = np.arange(1, K + 1)
    rows = pd.concat([pd.DataFrame(dict(well_id=a.well, origin=idx[o], lead=lead, pred=pr[j], pers=y_ff[o - 1],
                                        obs=obs[o:o + K], ok=ok[o:o + K])) for j, o in enumerate(o_te)])
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / f"{tag}.parquet", index=False)
    res = dict(well=a.well, model=a.model, variant=a.variant, lam=a.lam, horizon=K, seed=a.seed, hidden=a.hidden,
               window=W, epochs=ep, best_val=best, g1=g1, n_train=int(len(o_tr)), seconds=round(time.time() - t0, 1))
    if ode:
        gr, gd, href = ode.values()
        res.update(gamma_r=float(gr.detach()), gamma_d=float(gd.detach()), h_ref=float(href.detach()))
    (out / f"{tag}.json").write_text(json.dumps(res))
    print(tag, a.well, f"val={best:.4f} G1={g1} {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
