# Orchestration: recommend() runs data → analytics → forecast → scoring → strategy → risk for a
# date and expiry; log_trade() and check_positions() connect the journal to the risk engine.
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.analytics import (atm_iv, enrich_chain, implied_move, implied_range, iv_rank,
                               risk_reversal_25d)
from advisor.calendar import calendar_days_to, trading_days_between
from advisor.config import config_hash, load_events, load_holidays
from advisor.data import (DataError, data_status, list_expiries, load_chain, load_index_history,
                          load_vix_history, monthly_expiries)
from advisor.forecast import run_forecast, terminal_quantiles
from advisor.journal import add_trade, get_open_positions, save_snapshot
from advisor.risk import (Action, Position, RiskReport, assess, baseline_for, decide_action, mark,
                          pnl, pre_entry_check)
from advisor.scoring import DirectionScore, Edge, direction_score, premium_edge
from advisor.strategy import (IRON_BUTTERFLY, IRON_CONDOR, NAKED_STRANGLE, Candidate, Leg,
                              Selection, build_candidate, select_structures)

ROOT = Path(__file__).resolve().parents[1]
METRIC_DAYS = 252          # trailing days of ATM IV / RR25 history for IV rank and skew median
MIN_METRIC_DAYS = 20       # fewer than this → IV rank falls back to India VIX rank
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@dataclass
class Recommendation:
    date: date
    expiry: date
    data_ok: bool
    data_reason: str | None
    spot: float | None = None
    future: float | None = None
    atm_iv: float | None = None
    ivr: float | None = None
    ivr_source: str | None = None
    direction: DirectionScore | None = None
    edge: Edge | None = None
    selection: Selection | None = None
    primary: Candidate | None = None
    alternatives: list[Candidate] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    calibration_k: float = 1.0
    config_hash: str = ""
    quantiles: dict = field(default_factory=dict)
    implied_range: list = field(default_factory=list)
    lot_size: int | None = None


def _clean(x):
    if dataclasses.is_dataclass(x):
        return {f.name: _clean(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not math.isfinite(x) else round(x, 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


def recommendation_json(rec: Recommendation) -> str:
    """Deterministic JSON: sorted keys, floats rounded to 6 dp, ISO dates, NaN/inf → null."""
    return json.dumps(_clean(rec), sort_keys=True, separators=(",", ":"))


def _defaults(holidays, events):
    if holidays is None:
        holidays = load_holidays(ROOT / "holidays.yaml")
    if events is None:
        events = load_events(ROOT / "events.yaml")
    return holidays, events


# ---------- historical ATM IV / RR25 (for IV rank and skew median) ----------

def _metrics_path(cfg) -> Path:
    return Path(cfg.data_dir) / "metrics_daily.csv"


def _front_monthly(d: date, cfg) -> date | None:
    ms = [e for e in monthly_expiries(list_expiries(d, cfg)) if calendar_days_to(d, e) > cfg.exits.time_exit_days]
    return ms[0] if ms else None


def _day_metrics(d: date, cfg) -> dict | None:
    try:
        e = _front_monthly(d, cfg)
        if e is None:
            return None
        ch = load_chain(d, e, cfg)
        en = enrich_chain(ch, cfg.risk_free_rate)
        return {"date": d.isoformat(), "atm_iv": atm_iv(en, ch.future), "rr25": risk_reversal_25d(en)}
    except DataError:
        return None


def history_metrics(d: date, cfg) -> pd.DataFrame:
    """Front-monthly ATM IV and RR25 for each cached bhavcopy day ≤ d (last 252), cached in
    data_dir/metrics_daily.csv. Columns: date, atm_iv, rr25."""
    days = sorted(date.fromisoformat(p.name[:10]) for p in (Path(cfg.data_dir) / "bhavcopy").glob("*.csv.zip"))
    days = [x for x in days if x <= d][-METRIC_DAYS:]
    p = _metrics_path(cfg)
    cache = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["date", "atm_iv", "rr25"])
    have = set(cache.date)
    new = [m for x in days if x.isoformat() not in have and (m := _day_metrics(x, cfg))]
    if new:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True).sort_values("date")
        p.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(p, index=False)
    keep = {x.isoformat() for x in days}
    return cache[cache.date.isin(keep)].reset_index(drop=True)


# ---------- candidate evaluation with pre-entry downgrade (§10.3) ----------

def _shorts(c: Candidate) -> dict[str, float]:
    return {("put" if l.option_type == "PE" else "call"): l.strike for l in c.legs if l.side == "SELL"}


def _legs_key(c: Candidate):
    return c.structure, tuple(sorted((l.option_type, l.strike, l.side) for l in c.legs))


def evaluate_structures(structures: list[str], enriched, chain, fc, events, d: date, cfg,
                        holidays: set, open_margin: float) -> tuple[list[Candidate], list[str]]:
    """Build each structure in order and run the pre-entry check; on failure downgrade once:
    naked → iron condor at the same shorts; defined risk → shorts one strike_step further out
    (iron butterfly: wings one step wider). Returns (survivors in order, rejection notes)."""
    survivors, rejected, seen = [], [], set()

    def check(c):
        return pre_entry_check(c, chain, enriched, fc, events, d, cfg, holidays)

    for s in structures:
        c, why = build_candidate(s, enriched, chain, fc, cfg, open_margin)
        if c is None:
            rejected.append(f"{s}: {why}")
            continue
        ok, reasons = check(c)
        if not ok:
            rejected.append(f"{s}: pre-entry failed ({'; '.join(reasons)})")
            step = cfg.strike_step
            if s == NAKED_STRANGLE:
                c, why = build_candidate(IRON_CONDOR, enriched, chain, fc, cfg, open_margin, _shorts(c))
            elif s == IRON_BUTTERFLY:
                wider = cfg.model_copy(update={"wings": cfg.wings.model_copy(
                    update={"widths": [w for w in cfg.wings.widths if w >= c.width + step]})})
                c, why = build_candidate(s, enriched, chain, fc, wider, open_margin) if wider.wings.widths \
                    else (None, "no wider wing")
            else:
                out = {k: (v - step if k == "put" else v + step) for k, v in _shorts(c).items()}
                c, why = build_candidate(s, enriched, chain, fc, cfg, open_margin, out)
            if c is None:
                rejected.append(f"{s} downgrade: {why}")
                continue
            ok, reasons = check(c)
            if not ok:
                rejected.append(f"{c.structure} downgrade: pre-entry failed ({'; '.join(reasons)})")
                continue
            c.notes.append(f"downgraded from {s}")
        key = _legs_key(c)
        if key not in seen:
            seen.add(key)
            survivors.append(c)
    return survivors, rejected


# ---------- public entry points ----------

def recommend(d: date, expiry: date, cfg, sampler=None, open_margin: float = 0.0,
              holidays: set | None = None, events: list | None = None) -> Recommendation:
    """Full deterministic recommendation for entering a trade on d's close for `expiry`."""
    holidays, events = _defaults(holidays, events)
    base = dict(date=d, expiry=expiry, calibration_k=cfg.calibration.k, config_hash=config_hash(cfg))
    st = data_status(d, cfg.data_dir)
    if not st.complete:
        return Recommendation(data_ok=False, data_reason="missing: " + ", ".join(st.missing), **base)
    try:
        chain = load_chain(d, expiry, cfg)
        fc = run_forecast(d, expiry, cfg, holidays, sampler)
    except DataError as e:
        return Recommendation(data_ok=False, data_reason=str(e), **base)
    en = enrich_chain(chain, cfg.risk_free_rate)
    t = calendar_days_to(d, expiry) / 365.0
    iv = atm_iv(en, chain.future)
    hist = history_metrics(d, cfg)
    if len(hist.dropna(subset=["atm_iv"])) >= MIN_METRIC_DAYS:
        ivr, src = iv_rank(hist.atm_iv, iv), "atm_iv"
    else:
        vix = load_vix_history(d, METRIC_DAYS, cfg.data_dir)
        ivr, src = iv_rank(vix, float(vix.iloc[-1])), "india_vix"
    direction = direction_score(chain.spot, implied_move(chain.future, iv, t), fc, en,
                                load_index_history(d, 60, cfg.data_dir), hist.rr25, cfg)
    edge = premium_edge(chain.future, iv, t, fc, ivr, cfg)
    sel = select_structures(d, expiry, direction, edge, events, cfg)
    surv, rejected = evaluate_structures(sel.structures, en, chain, fc, events, d, cfg, holidays, open_margin) \
        if sel.structures else ([], [])
    return Recommendation(
        data_ok=True, data_reason=None, spot=chain.spot, future=chain.future, atm_iv=iv, ivr=ivr,
        ivr_source=src, direction=direction, edge=edge, selection=sel,
        primary=surv[0] if surv else None, alternatives=surv[1:], rejected=rejected,
        quantiles={f"p{int(round(q * 100)):02d}": v for q, v in terminal_quantiles(fc, QUANTILES).items()},
        implied_range=list(implied_range(chain.future, iv, t, 0.10)), lot_size=chain.lot_size, **base)


def _position(structure, expiry, legs, d) -> Position:
    for l in legs:
        l.entry_date = d
    credit = sum(l.price if l.side == "SELL" else -l.price for l in legs)
    return Position(None, structure, expiry, legs, credit, 0, {})


def log_trade(conn, d: date, structure: str, expiry: date, legs: list[Leg], cfg, sampler=None,
              holidays: set | None = None, notes: str = "") -> int:
    """Validate legs against d's chain, compute the entry baseline, and store the trade."""
    holidays, _ = _defaults(holidays, [])
    try:
        chain = load_chain(d, expiry, cfg)
    except DataError as e:
        raise ValueError(f"no chain for {expiry} on {d}: {e}") from e
    listed = set(zip(chain.rows.option_type, chain.rows.strike))
    for l in legs:
        if l.side not in ("SELL", "BUY") or l.lots < 1:
            raise ValueError(f"bad leg: {l.option_type} {l.strike:.0f} side={l.side} lots={l.lots}")
        if (l.option_type, float(l.strike)) not in listed:
            raise ValueError(f"leg not in chain: {l.option_type} {l.strike:.0f} {expiry.isoformat()}")
    en = enrich_chain(chain, cfg.risk_free_rate)
    fc = run_forecast(d, expiry, cfg, holidays, sampler)
    pos = _position(structure, expiry, legs, d)
    return add_trade(conn, structure, expiry, d, legs, baseline_for(pos, chain, en, fc, cfg),
                     chain.lot_size, notes)


def check_positions(conn, d: date, cfg, sampler=None, holidays: set | None = None,
                    events: list | None = None) -> list[tuple[Position, RiskReport, Action]]:
    """Risk level and action call for every open trade on d; saves a snapshot each. Never applies actions."""
    holidays, events = _defaults(holidays, events)
    out = []
    for pos in get_open_positions(conn):
        if pos.expiry <= d or trading_days_between(d, pos.expiry, holidays) < 1:
            rep, act = RiskReport("RED", ["at or past expiry"], {}), Action("EXIT", "expiry")
            save_snapshot(conn, pos.trade_id, d, None, None, rep, act)
            out.append((pos, rep, act))
            continue
        chain = load_chain(d, pos.expiry, cfg)
        en = enrich_chain(chain, cfg.risk_free_rate)
        fc = run_forecast(d, pos.expiry, cfg, holidays, sampler)
        rep = assess(pos, chain, en, fc, events, d, cfg, holidays)
        act = decide_action(pos, rep, chain, en, fc, d, cfg, events, holidays)
        save_snapshot(conn, pos.trade_id, d, mark(pos, en), pnl(pos, en), rep, act)
        out.append((pos, rep, act))
    return out
