"""Rung 5: an LSTM with NO graph. One set of weights shared by all 207 sensors, each sensor's own 12-step history in,
12 horizons out. It isolates "deep learning" from "graph": if a graph model later beats this, the gain is the graph.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SharedLSTM(nn.Module):
    def __init__(self, n_in: int = 7, hidden: int = 64, layers: int = 2, horizon: int = 12, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x (B, 12, n_in) -> (B, horizon), in z-score units
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def _batch(X: np.ndarray, rows: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(X[rows]))          # (b, 12, N, F)
    b, t, n, f = x.shape
    return x.permute(0, 2, 1, 3).reshape(b * n, t, f).to(device)


def masked_mae_loss(pred_mph: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred_mph - y) * m).sum() / m.sum().clamp(min=1)


def predict(model: SharedLSTM, X: np.ndarray, scaler: dict, device, windows_per_batch: int = 64) -> np.ndarray:
    """-> rows (S * N, 12) in mph, same layout as the feature table."""
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(X), windows_per_batch):
            rows = np.arange(a, min(a + windows_per_batch, len(X)))
            z = model(_batch(X, rows, device))
            out.append((z * scaler["sd"] + scaler["mu"]).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def fit(Xtr, Ytr, Mtr, Xva, Yva, Mva, scaler: dict, epochs: int = 12, patience: int = 3, windows_per_batch: int = 32,
        lr: float = 2e-3, seed: int = 42, log=print) -> tuple[SharedLSTM, list[dict]]:
    """Masked-MAE training in mph; early stopping on validation MAE. Windows are shuffled only inside the train block."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    model = SharedLSTM(n_in=Xtr.shape[-1], horizon=Ytr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    N = Xtr.shape[2]
    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(Xtr))
        for a in range(0, len(order), windows_per_batch):
            rows = np.sort(order[a:a + windows_per_batch])
            y = torch.from_numpy(np.nan_to_num(Ytr[rows])).permute(0, 2, 1).reshape(-1, Ytr.shape[1]).to(device)
            m = torch.from_numpy(Mtr[rows].astype(np.float32)).permute(0, 2, 1).reshape(-1, Ytr.shape[1]).to(device)
            pred = model(_batch(Xtr, rows, device)) * scaler["sd"] + scaler["mu"]
            loss = masked_mae_loss(pred, y, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        P = predict(model, Xva, scaler, device)
        Yr = Yva.transpose(0, 2, 1).reshape(-1, Yva.shape[1])
        Mr = Mva.transpose(0, 2, 1).reshape(-1, Mva.shape[1])
        val = float((np.abs(P - np.nan_to_num(Yr)) * Mr).sum() / Mr.sum())
        sched.step(val)
        history.append({"epoch": ep, "train_mae": float(np.mean(losses)), "val_mae": val})
        log(f"  lstm epoch {ep:2d}: train MAE {np.mean(losses):.3f}  val MAE {val:.3f}  ({time.time() - t0:.0f}s, N={N})")
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history
