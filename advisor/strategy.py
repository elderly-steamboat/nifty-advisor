# Strategy selection (§10.3 table, gates, priority) and trade construction
# (§11 short strikes, OI walls, wings, sizing) — produces Candidate trades.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from advisor.calendar import calendar_days_to
from advisor.scoring import DirectionScore, Edge

BEAR_CALL = "BEAR_CALL"
BULL_PUT = "BULL_PUT"
IRON_CONDOR = "IRON_CONDOR"
SKEW_IC_BEAR = "SKEW_IC_BEAR"
SKEW_IC_BULL = "SKEW_IC_BULL"
IRON_BUTTERFLY = "IRON_BUTTERFLY"
NAKED_STRANGLE = "NAKED_STRANGLE"

BAND_MAP = {"strong_bearish": [BEAR_CALL], "mild_bearish": [SKEW_IC_BEAR],
            "mild_bullish": [SKEW_IC_BULL], "strong_bullish": [BULL_PUT]}


@dataclass
class Selection:
    structures: list[str]
    cell: str
    gates: list[str] = field(default_factory=list)
    no_trade_reason: str | None = None


def select_structures(d: date, expiry: date, direction: DirectionScore, edge: Edge,
                      events, cfg) -> Selection:
    """Ordered structures for the day (primary first), per §10.3. Config: exits.time_exit_days,
    edge.r_naked/ivr_naked/butterfly_width_ratio, scoring.neutral_band."""
    edge_cell = "edge_good" if edge.good else "edge_low"
    cell = f"{direction.band}/{edge_cell}"
    if calendar_days_to(d, expiry) <= cfg.exits.time_exit_days:
        return Selection([], cell, [], "too close to expiry")
    if not edge.good:
        return Selection([], cell, [], f"premium edge low (R={edge.r:.2f}, IVR={edge.ivr:.0f})")
    if direction.band in BAND_MAP:
        return Selection(list(BAND_MAP[direction.band]), cell)
    gates = []
    bfly = edge.kronos_width <= cfg.edge.butterfly_width_ratio * edge.implied_width
    gates.append(f"butterfly_tight: {'pass' if bfly else 'fail'} "
                 f"(kronos {edge.kronos_width:.0f} vs {cfg.edge.butterfly_width_ratio}×implied {edge.implied_width:.0f})")
    fails = []
    if abs(direction.value) > cfg.scoring.neutral_band:
        fails.append(f"|direction| {abs(direction.value):.1f} > {cfg.scoring.neutral_band}")
    if edge.r < cfg.edge.r_naked:
        fails.append(f"R {edge.r:.2f} < {cfg.edge.r_naked}")
    if edge.ivr < cfg.edge.ivr_naked:
        fails.append(f"IVR {edge.ivr:.0f} < {cfg.edge.ivr_naked}")
    ev = [e for e in events if d < e.date <= expiry]
    if ev:
        fails.append("event before expiry: " + ", ".join(f"{e.name} {e.date}" for e in ev))
    naked = not fails
    gates.append("naked_gate: pass" if naked else "naked_gate: fail (" + "; ".join(fails) + ")")
    if bfly:
        s = [IRON_BUTTERFLY] + ([NAKED_STRANGLE] if naked else []) + [IRON_CONDOR]
    elif naked:
        s = [NAKED_STRANGLE, IRON_CONDOR]
    else:
        s = [IRON_CONDOR]
    return Selection(s, cell, gates)


# ---------- trade construction (§11) ----------

import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from advisor.analytics import atm_strike, black76_price, find_walls, strike_for_delta  # noqa: E402
from advisor.forecast import Forecast, terminal_quantiles  # noqa: E402

PARAMS = {  # structure → (put-side strikes key, call-side strikes key); None = side unused
    IRON_CONDOR: ("iron_condor", "iron_condor"),
    SKEW_IC_BULL: ("skewed_favoured", "skewed_other"),
    SKEW_IC_BEAR: ("skewed_other", "skewed_favoured"),
    BULL_PUT: ("credit_spread", None),
    BEAR_CALL: (None, "credit_spread"),
    NAKED_STRANGLE: ("naked_strangle", "naked_strangle"),
    IRON_BUTTERFLY: ("atm", "atm"),
}
TYPE = {"put": "PE", "call": "CE"}


@dataclass
class Leg:
    option_type: str
    strike: float
    side: str            # "SELL" | "BUY"
    lots: int
    price: float         # points per unit
    entry_date: date | None = None
    exit_price: float | None = None
    exit_date: date | None = None
    id: int | None = None


@dataclass
class Candidate:
    structure: str
    legs: list[Leg]
    credit: float              # points per unit
    max_loss: float | None     # points per unit; None for naked
    width: float | None
    lots: int
    expected_pnl: float        # points per unit, mean over Kronos terminal closes
    pop: float
    margin_est: float          # INR, approximate
    strike_reasons: dict[str, str]
    notes: list[str] = field(default_factory=list)


def payoff(legs: list[Leg], s: np.ndarray) -> np.ndarray:
    """Expiry P&L per unit at underlying s: SELL legs earn price − intrinsic, BUY legs the reverse."""
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    for l in legs:
        intrinsic = np.maximum(s - l.strike, 0) if l.option_type == "CE" else np.maximum(l.strike - s, 0)
        sign = 1.0 if l.side == "SELL" else -1.0
        out += sign * (l.price - intrinsic)
    return out


def _row(enriched, typ, k):
    r = enriched[(enriched.option_type == typ) & (enriched.strike == k)]
    return None if r.empty else r.iloc[0]


def _liquid(row, cfg) -> bool:
    return (row is not None and row.oi >= cfg.liquidity.min_oi and row.volume >= cfg.liquidity.min_volume
            and np.isfinite(row.iv) and row.settle > 0)


def _listed_outward(enriched, typ, x, side) -> float:
    """Listed strike at or beyond x away from spot (put: ≤ x, call: ≥ x)."""
    ks = np.sort(enriched[enriched.option_type == typ].strike.unique())
    pick = ks[ks <= x + 1e-9] if side == "put" else ks[ks >= x - 1e-9]
    if len(pick) == 0:
        return math.nan
    return float(pick.max() if side == "put" else pick.min())


def pick_short_strike(enriched: pd.DataFrame, fc: Forecast, side: str, q: float, delta: float,
                      walls: dict, cfg) -> tuple[float, str]:
    """Short strike for one side (§11): the further from spot of the Kronos quantile (q / 1−q)
    and the `delta` strike, rounded outward; then moved out to the next OI wall beyond it unless
    that wall is beyond Kronos P(walls.max_q) or below walls.min_delta.
    Returns (strike, reason) with reason in kronos | delta | wall | '<base> (wall ignored: too far)'."""
    typ = TYPE[side]
    kq = terminal_quantiles(fc, [q if side == "put" else 1 - q])[q if side == "put" else 1 - q]
    kd = strike_for_delta(enriched, typ, delta)
    if side == "put":
        use_kronos = not np.isfinite(kd) or kq <= kd
    else:
        use_kronos = not np.isfinite(kd) or kq >= kd
    base_x, why = (kq, "kronos") if use_kronos else (kd, "delta")
    base = _listed_outward(enriched, typ, base_x, side)
    beyond = [w for w in walls.get(side, []) if (w < base if side == "put" else w > base)]
    if not beyond:
        return base, why
    wall = min(beyond, key=lambda w: abs(w - base))
    lim_q = cfg.walls.max_q if side == "put" else 1 - cfg.walls.max_q
    lim = terminal_quantiles(fc, [lim_q])[lim_q]
    row = _row(enriched, typ, wall)
    too_far = (wall < lim if side == "put" else wall > lim) or row is None or \
        not np.isfinite(row.delta) or abs(row.delta) < cfg.walls.min_delta
    return (base, f"{why} (wall ignored: too far)") if too_far else (wall, "wall")


def choose_wing(enriched: pd.DataFrame, short_strike: float, side: str, width: float, cfg) -> Leg | None:
    """Long wing `width` points beyond short_strike on `side`, priced at settle; None if illiquid."""
    typ = TYPE[side]
    k = short_strike - width if side == "put" else short_strike + width
    row = _row(enriched, typ, k)
    return Leg(typ, float(k), "BUY", 0, float(row.settle)) if _liquid(row, cfg) else None


def size_defined(max_loss: float, lot_size: int, cfg) -> int:
    """floor(sizing.defined_risk_pct × capital / (max_loss × lot_size))."""
    return int(math.floor(cfg.sizing.defined_risk_pct * cfg.capital / (max_loss * lot_size))) if max_loss > 0 else 0


def size_naked(gap_loss: float, lot_size: int, cfg) -> int:
    """floor(sizing.naked_stress_pct × capital / (gap_loss × lot_size))."""
    return int(math.floor(cfg.sizing.naked_stress_pct * cfg.capital / (gap_loss * lot_size))) if gap_loss > 0 else 0


def cap_by_margin(lots: int, margin_per_lot: float, open_margin: float, cfg) -> int:
    """Largest lots ≤ given so open_margin + lots × margin_per_lot ≤ sizing.max_margin_pct × capital."""
    room = cfg.sizing.max_margin_pct * cfg.capital - open_margin
    return max(0, min(lots, int(math.floor(room / margin_per_lot)) if margin_per_lot > 0 else lots))


def naked_gap_loss(legs: list[Leg], chain, enriched, gap: float, r: float) -> float:
    """Worse-side loss per unit after a ±gap move in the forward, one day later, same leg IVs."""
    t = max((chain.expiry - chain.date).days - 1, 0) / 365.0
    credit = sum(l.price if l.side == "SELL" else -l.price for l in legs)
    worst = 0.0
    for m in (1 - gap, 1 + gap):
        F = chain.future * m
        val = 0.0
        for l in legs:
            iv = _row(enriched, l.option_type, l.strike).iv
            px = black76_price(F, l.strike, t, r, iv, l.option_type)
            val += px if l.side == "SELL" else -px
        worst = max(worst, val - credit)
    return worst


def build_candidate(structure: str, enriched: pd.DataFrame, chain, fc: Forecast, cfg,
                    open_margin: float, shorts_override: dict[str, float] | None = None
                    ) -> tuple[Candidate | None, str]:
    """Build one structure's trade (§11) or return (None, reason). Config: strikes.*, walls.*,
    liquidity.*, wings.*, sizing.*, strike_step."""
    put_key, call_key = PARAMS[structure]
    walls = find_walls(enriched, chain.spot, cfg.scoring.oi_window_pct, cfg.walls.oi_multiple, cfg.walls.top_n)
    shorts, reasons = {}, {}
    for side, key in (("put", put_key), ("call", call_key)):
        if key is None:
            continue
        if shorts_override and side in shorts_override:
            shorts[side], reasons[side] = float(shorts_override[side]), "override"
        elif key == "atm":
            shorts[side], reasons[side] = atm_strike(enriched, chain.future, cfg.strike_step), "atm"
        else:
            p = getattr(cfg.strikes, key)
            shorts[side], reasons[side] = pick_short_strike(enriched, fc, side, p.q, p.delta, walls, cfg)
    short_legs = {}
    for side, k in shorts.items():
        row = _row(enriched, TYPE[side], k)
        if not _liquid(row, cfg):
            return None, f"illiquid short {TYPE[side]} {k:.0f}"
        short_legs[side] = Leg(TYPE[side], float(k), "SELL", 0, float(row.settle))
    terms = fc.closes[:, -1]

    if structure == NAKED_STRANGLE:
        q = terminal_quantiles(fc, [0.05, 0.95])
        if not (shorts["put"] <= q[0.05] and shorts["call"] >= q[0.95]):
            return None, "naked gate: shorts inside P5–P95"
        legs = [short_legs["put"], short_legs["call"]]
        credit = sum(l.price for l in legs)
        gap_loss = naked_gap_loss(legs, chain, enriched, cfg.sizing.naked_gap, cfg.risk_free_rate)
        lots = cap_by_margin(size_naked(gap_loss, chain.lot_size, cfg), cfg.sizing.naked_margin_per_lot,
                             open_margin, cfg)
        if lots < 1:
            return None, "size < 1 lot"
        for l in legs:
            l.lots = lots
        pnl = payoff(legs, terms)
        return Candidate(structure, legs, credit, None, None, lots, float(pnl.mean()),
                         float((pnl >= 0).mean()), cfg.sizing.naked_margin_per_lot * lots, reasons,
                         [f"gap loss at ±{cfg.sizing.naked_gap:.0%}: {gap_loss:.1f} pts"]), ""

    ratio = cfg.wings.butterfly_min_credit_ratio if structure == IRON_BUTTERFLY else cfg.wings.min_credit_ratio
    best, any_ratio_ok = None, False
    for width in cfg.wings.widths:
        wings = {side: choose_wing(enriched, k, side, float(width), cfg) for side, k in shorts.items()}
        if any(w is None for w in wings.values()):
            continue
        legs = []
        if "put" in shorts:
            legs += [wings["put"], short_legs["put"]]
        if "call" in shorts:
            legs += [short_legs["call"], wings["call"]]
        credit = sum(l.price if l.side == "SELL" else -l.price for l in legs)
        if credit < ratio * width:
            continue
        any_ratio_ok = True
        grid = np.array(sorted({l.strike for l in legs} | {0.0, 2 * chain.future}))
        max_loss = float(-payoff(legs, grid).min())
        margin_per_lot = max_loss * chain.lot_size
        lots = cap_by_margin(size_defined(max_loss, chain.lot_size, cfg), margin_per_lot, open_margin, cfg)
        if lots < 1:
            continue
        pnl = payoff(legs, terms)
        ev = float(pnl.mean())
        total = ev * lots  # compare widths on the sized position (equal risk budget)
        if best is None or total > best[0] + 1e-9:
            best = (total, ev, width, legs, credit, pnl, max_loss, lots, margin_per_lot)
    if best is None:
        return None, "size < 1 lot" if any_ratio_ok else "no wing width meets credit ratio"
    _, ev, width, legs, credit, pnl, max_loss, lots, margin_per_lot = best
    legs = [Leg(l.option_type, l.strike, l.side, lots, l.price) for l in legs]
    return Candidate(structure, legs, float(credit), max_loss, float(width), lots, ev,
                     float((pnl >= 0).mean()), margin_per_lot * lots, reasons), ""
