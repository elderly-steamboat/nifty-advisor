# Kronos price-path forecasting: deterministic sampling of N paths, calibration,
# quantiles and touch probabilities, with an on-disk cache of raw paths.
from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "kronos"
CHUNK = 50  # paths per Kronos batch (memory bound); fixed so results are deterministic

Sampler = Callable[[pd.DataFrame, list, int, int], np.ndarray]
_MODELS: dict = {}


def forecast_seed(base_seed: int, d: date, expiry: date) -> int:
    """Deterministic 32-bit seed for one forecast: sha256 of base_seed|date|expiry."""
    h = hashlib.sha256(f"{base_seed}|{d.isoformat()}|{expiry.isoformat()}".encode()).hexdigest()
    return int(h[:8], 16)


def _load_kronos(cfg):
    key = (cfg.kronos.model_id, cfg.kronos.tokenizer_id, cfg.kronos.device)
    if key not in _MODELS:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from model import Kronos, KronosTokenizer  # vendored, pinned commit
        tok = KronosTokenizer.from_pretrained(cfg.kronos.tokenizer_id).to(cfg.kronos.device).eval()
        mdl = Kronos.from_pretrained(cfg.kronos.model_id).to(cfg.kronos.device).eval()
        _MODELS[key] = (tok, mdl)
    return _MODELS[key]


def _stamps(dates) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(list(dates)))
    return np.stack([ts.dt.minute, ts.dt.hour, ts.dt.day_of_week, ts.dt.day, ts.dt.month],
                    axis=1).astype(np.float32)


def kronos_sampler(cfg) -> Sampler:
    """Sampler running Kronos on CPU: returns (n_paths, horizon, 3) array of (high, low, close).
    One torch seed per call; paths are a batch of identical histories with sample_count=1,
    so each row is an independent sample (Kronos's predict() would average them)."""
    import torch

    def sample(history: pd.DataFrame, future_dates: list, n_paths: int, seed: int) -> np.ndarray:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from model.kronos import auto_regressive_inference
        tok, mdl = _load_kronos(cfg)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        h = history[["open", "high", "low", "close", "volume"]].astype(float).copy()
        h["amount"] = h.volume * h[["open", "high", "low", "close"]].mean(axis=1)
        x = h.to_numpy(np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        xn = np.clip((x - mean) / (std + 1e-5), -5, 5).astype(np.float32)
        xs, ys = _stamps(history.index), _stamps(future_dates)
        horizon = len(future_dates)
        out = []
        for start in range(0, n_paths, CHUNK):
            b = min(CHUNK, n_paths - start)
            preds = auto_regressive_inference(
                tok, mdl,
                torch.from_numpy(np.repeat(xn[None], b, 0)).to(cfg.kronos.device),
                torch.from_numpy(np.repeat(xs[None], b, 0)).to(cfg.kronos.device),
                torch.from_numpy(np.repeat(ys[None], b, 0)).to(cfg.kronos.device),
                cfg.kronos.context, horizon, 5, cfg.kronos.temperature, 0,
                cfg.kronos.top_p, 1, False)
            preds = preds[:, -horizon:, :] * (std + 1e-5) + mean
            out.append(preds[:, :, [1, 2, 3]])
        return np.concatenate(out, axis=0).astype(np.float64)

    return sample
