# Loads and validates config.yaml, events.yaml and holidays.yaml; computes the config hash
# recorded with every output so any recommendation can be reproduced.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Kronos(_M):
    model_id: str
    tokenizer_id: str
    context: int
    n_paths: int
    temperature: float
    top_p: float
    device: str
    base_seed: int


class Calibration(_M):
    k: float
    window_cycles: int
    target_coverage: float


class Weights(_M):
    kronos_drift: float
    oi: float
    trend: float
    skew: float


class Scoring(_M):
    weights: Weights
    pcr_low: float
    pcr_high: float
    neutral_band: float
    strong_band: float
    oi_window_pct: float
    trend_scales: list[float]
    skew_scale: float

    @model_validator(mode="after")
    def _weights_sum(self):
        w = self.weights
        total = w.kronos_drift + w.oi + w.trend + w.skew
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"scoring.weights must sum to 1.0, got {total}")
        return self


class Edge(_M):
    r_good: float
    ivr_good: float
    r_naked: float
    ivr_naked: float
    butterfly_width_ratio: float


class QDelta(_M):
    q: float
    delta: float


class Strikes(_M):
    iron_condor: QDelta
    skewed_favoured: QDelta
    skewed_other: QDelta
    credit_spread: QDelta
    naked_strangle: QDelta


class Walls(_M):
    oi_multiple: float
    top_n: int
    min_delta: float
    max_q: float


class Liquidity(_M):
    min_oi: float
    min_volume: float


class Wings(_M):
    widths: list[int]
    min_credit_ratio: float
    butterfly_min_credit_ratio: float


class Sizing(_M):
    defined_risk_pct: float
    naked_stress_pct: float
    naked_gap: float
    max_margin_pct: float
    naked_margin_per_lot: float


class NakedRisk(_M):
    amber_touch: float
    red_touch: float
    amber_delta: float
    red_delta: float


class PreEntry(_M):
    touch: float
    naked_touch: float
    gap: float
    gap_loss_mult: float
    naked_gap: float
    naked_gap_loss_mult: float


class Risk(_M):
    window_days: int
    event_days: int
    amber_touch: float
    red_touch: float
    amber_touch_mult: float
    red_touch_mult: float
    amber_delta: float
    red_delta: float
    amber_delta_add: float
    red_delta_add: float
    amber_gap: float
    amber_gap_loss_mult: float
    red_loss_mult: float
    butterfly_amber_frac: float
    butterfly_red_frac: float
    wall_oi_drop: float
    naked: NakedRisk
    pre_entry: PreEntry
    stress_moves: list[float]
    stress_iv_up: float


class Exits(_M):
    profit_target: float
    hard_stop_mult: float
    time_exit_days: int
    max_rolls: int


class Config(_M):
    capital: float
    risk_free_rate: float
    lot_size_fallback: int
    strike_step: float
    data_dir: Path
    kronos: Kronos
    calibration: Calibration
    scoring: Scoring
    edge: Edge
    strikes: Strikes
    walls: Walls
    liquidity: Liquidity
    wings: Wings
    sizing: Sizing
    risk: Risk
    exits: Exits

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v):
        return Path(v).expanduser()


@dataclass(frozen=True)
class Event:
    date: date
    name: str
    kind: str


def load_config(path: Path = Path("config.yaml")) -> Config:
    """Read and validate config.yaml. Raises pydantic.ValidationError on any bad/unknown key."""
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))


def config_hash(cfg: Config) -> str:
    """First 12 hex chars of sha256 over the canonical JSON of the config."""
    canon = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def load_events(path: Path) -> list[Event]:
    """Read events.yaml (list of {date, name, kind}), sorted by date."""
    rows = yaml.safe_load(Path(path).read_text()) or []
    return sorted((Event(r["date"], r["name"], r["kind"]) for r in rows), key=lambda e: e.date)


def load_holidays(path: Path) -> set[date]:
    """Read holidays.yaml ({year: [dates]}) into one set of dates."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return {d for dates in data.values() for d in dates}
