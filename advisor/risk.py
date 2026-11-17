# Risk engine (§12): position marking, stress P&L, GREEN/AMBER/RED levels, pre-entry check,
# and HOLD/EXIT/ADJUST action calls with rolls.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from advisor.analytics import atm_iv, black76_greeks, black76_price, find_walls
from advisor.calendar import trading_days_between
from advisor.forecast import Forecast, iv_touch_prob, touch_prob
from advisor.strategy import IRON_BUTTERFLY, NAKED_STRANGLE, Candidate, Leg

SIDE_TYPE = {"put": "PE", "call": "CE"}
TRADING_YEAR = 252.0  # trading days per year, for short look-ahead windows


@dataclass
class Position:
    trade_id: int | None
    structure: str
    expiry: date
    legs: list[Leg]           # open legs only
    net_credit: float         # running net credit, points per unit
    rolls: int
    baseline: dict = field(default_factory=dict)


@dataclass
class RiskReport:
    level: str                # GREEN | AMBER | RED
    triggers: list[str]
    metrics: dict


def _row(enriched, typ, k):
    r = enriched[(enriched.option_type == typ) & (enriched.strike == k)]
    return None if r.empty else r.iloc[0]


def _shorts(pos: Position) -> dict[str, Leg]:
    return {("put" if l.option_type == "PE" else "call"): l for l in pos.legs if l.side == "SELL"}


def mark(position: Position, enriched: pd.DataFrame) -> float:
    """Cost to close all open legs now at settle (points per unit; positive for a short-premium position)."""
    total = 0.0
    for l in position.legs:
        row = _row(enriched, l.option_type, l.strike)
        px = float(row.settle) if row is not None else l.price
        total += px if l.side == "SELL" else -px
    return total


def pnl(position: Position, enriched: pd.DataFrame) -> float:
    """Running net credit minus cost to close (points per unit)."""
    return position.net_credit - mark(position, enriched)


def danger_levels(position: Position) -> dict[str, float]:
    """Per side: short strike; for an iron butterfly the breakeven (short ∓ net credit)."""
    out = {}
    for side, l in _shorts(position).items():
        if position.structure == IRON_BUTTERFLY:
            out[side] = l.strike - position.net_credit if side == "put" else l.strike + position.net_credit
        else:
            out[side] = l.strike
    return out


def _leg_iv(enriched, l: Leg, fallback: float) -> float:
    row = _row(enriched, l.option_type, l.strike)
    iv = float(row.iv) if row is not None and np.isfinite(row.iv) else fallback
    return iv


def _value_after(position, chain, enriched, move: float, iv_add: float, r: float) -> float:
    t = max((chain.expiry - chain.date).days - 1, 0) / 365.0
    F = chain.future * (1 + move)
    fb = atm_iv(enriched, chain.future)
    v = 0.0
    for l in position.legs:
        px = black76_price(F, l.strike, t, r, _leg_iv(enriched, l, fb) + iv_add, l.option_type)
        v += px if l.side == "SELL" else -px
    return v


def stress(position: Position, chain, enriched: pd.DataFrame, cfg) -> dict[str, float]:
    """P&L per unit after one day for each risk.stress_moves spot move and an IV shock of
    risk.stress_iv_up (Black-76, each leg's current IV)."""
    r = cfg.risk_free_rate
    out = {f"{m:+.0%}": position.net_credit - _value_after(position, chain, enriched, m, 0.0, r)
           for m in cfg.risk.stress_moves}
    out[f"iv+{cfg.risk.stress_iv_up * 100:.0f}"] = position.net_credit - _value_after(
        position, chain, enriched, 0.0, cfg.risk.stress_iv_up, r)
    return out


def _gap_loss_mult(position, chain, enriched, gap, cfg) -> float:
    r = cfg.risk_free_rate
    worst = max(_value_after(position, chain, enriched, m, 0.0, r) - position.net_credit for m in (-gap, gap))
    return worst / position.net_credit if position.net_credit > 0 else math.inf


def _touch_3d(fc, level, side, F, iv, cfg) -> float:
    w = cfg.risk.window_days
    return max(touch_prob(fc, level, "below" if side == "put" else "above", w),
               iv_touch_prob(F, level, iv, w / TRADING_YEAR))


def side_metrics(position: Position, chain, enriched, fc: Forecast, cfg) -> dict[str, dict]:
    """touch_3d, |delta|, and (butterfly) fraction of entry-spot→breakeven distance covered, per side."""
    iv = atm_iv(enriched, chain.future)
    danger = danger_levels(position)
    entry_spot = position.baseline.get("entry_spot")
    out = {}
    for side, l in _shorts(position).items():
        row = _row(enriched, l.option_type, l.strike)
        if row is not None and np.isfinite(row.delta):
            dl = abs(float(row.delta))
        else:
            t = max((chain.expiry - chain.date).days, 0) / 365.0
            dl = abs(black76_greeks(chain.future, l.strike, t, cfg.risk_free_rate, iv, l.option_type)["delta"])
        frac = None
        if position.structure == IRON_BUTTERFLY and entry_spot:
            dist = entry_spot - danger[side] if side == "put" else danger[side] - entry_spot
            moved = entry_spot - chain.spot if side == "put" else chain.spot - entry_spot
            frac = max(0.0, moved / dist) if dist > 0 else 0.0
        out[side] = {"touch_3d": _touch_3d(fc, danger[side], side, chain.future, iv, cfg),
                     "touch_exp": max(touch_prob(fc, danger[side], "below" if side == "put" else "above", None),
                                      iv_touch_prob(chain.future, danger[side], iv,
                                                    max((chain.expiry - chain.date).days, 0) / 365.0)),
                     "delta": dl, "frac": frac}
    return out


def _wall_shift(position, chain, enriched, cfg) -> str | None:
    b_oi, b_k = position.baseline.get("wall_oi", {}), position.baseline.get("wall_strike", {})
    walls = find_walls(enriched, chain.spot, cfg.scoring.oi_window_pct, cfg.walls.oi_multiple, cfg.walls.top_n)
    for side, l in _shorts(position).items():
        k, oi0 = b_k.get(side), b_oi.get(side)
        if k is not None and oi0:
            row = _row(enriched, SIDE_TYPE[side], k)
            oi = float(row.oi) if row is not None else 0.0
            if oi < (1 - cfg.risk.wall_oi_drop) * oi0:
                return f"wall shift {side}: OI at {k:.0f} {oi0:.0f} → {oi:.0f}"
        inside = [w for w in walls[side] if (chain.spot > w > l.strike if side == "put" else chain.spot < w < l.strike)]
        if k is not None and inside:
            return f"wall shift {side}: wall moved inside short to {inside[0]:.0f}"
    return None


def _event_within(events, d, holidays, cfg) -> str | None:
    for e in events:
        if e.date > d and trading_days_between(d, e.date, holidays) <= cfg.risk.event_days:
            return f"{e.name} {e.date.isoformat()}"
    return None


def level_from_metrics(m: dict, baseline: dict, structure: str, cfg) -> tuple[str, list[str]]:
    """Apply §12 thresholds. m: {'sides': {side: {touch_3d, delta, frac}}, loss_mult, gap_loss_mult,
    event, wall_shift}. Config: risk.* (risk.naked.* for naked strangles)."""
    rk, naked, bfly = cfg.risk, structure == NAKED_STRANGLE, structure == IRON_BUTTERFLY
    bt, bd = baseline.get("touch_3d", {}), baseline.get("delta", {})
    amber, red = [], []
    for side, s in m["sides"].items():
        t, dl = s["touch_3d"], s["delta"]
        if naked:
            a_t, r_t, a_d, r_d = rk.naked.amber_touch, rk.naked.red_touch, rk.naked.amber_delta, rk.naked.red_delta
            a_rel = r_rel = True
        else:
            b0 = bt.get(side)
            a_t, r_t = rk.amber_touch, rk.red_touch
            a_rel = b0 is None or t > rk.amber_touch_mult * b0
            r_rel = b0 is None or t > rk.red_touch_mult * b0
            d0 = bd.get(side)
            a_d = max(rk.amber_delta, (d0 or 0) + rk.amber_delta_add)
            r_d = max(rk.red_delta, (d0 or 0) + rk.red_delta_add)
        if t > r_t and r_rel:
            red.append(f"touch_3d {side} {t:.2f} > {r_t:.2f}")
        elif t > a_t and a_rel:
            amber.append(f"touch_3d {side} {t:.2f} > {a_t:.2f}")
        if bfly:
            f = s.get("frac") or 0.0
            if f > rk.butterfly_red_frac:
                red.append(f"breakeven {side} {f:.0%} covered > {rk.butterfly_red_frac:.0%}")
            elif f > rk.butterfly_amber_frac:
                amber.append(f"breakeven {side} {f:.0%} covered > {rk.butterfly_amber_frac:.0%}")
        else:
            if dl > r_d:
                red.append(f"delta {side} {dl:.2f} > {r_d:.2f}")
            elif dl > a_d:
                amber.append(f"delta {side} {dl:.2f} > {a_d:.2f}")
    if m["loss_mult"] > rk.red_loss_mult:
        red.append(f"loss {m['loss_mult']:.2f}× credit > {rk.red_loss_mult}×")
    if m["gap_loss_mult"] > rk.amber_gap_loss_mult:
        amber.append(f"±{rk.amber_gap:.0%} gap loss {m['gap_loss_mult']:.2f}× credit > {rk.amber_gap_loss_mult}×")
    if m.get("event"):
        amber.append(f"event within {rk.event_days} trading days: {m['event']}")
    if m.get("wall_shift"):
        amber.append(m["wall_shift"])
    level = "RED" if red else "AMBER" if amber else "GREEN"
    return level, red + amber


def assess(position: Position, chain, enriched: pd.DataFrame, fc: Forecast, events, d: date, cfg,
           holidays: set = frozenset()) -> RiskReport:
    """Risk level for an open position on date d (§12)."""
    sides = side_metrics(position, chain, enriched, fc, cfg)
    p = pnl(position, enriched)
    nc = position.net_credit
    m = {"sides": sides,
         "loss_mult": (-p / nc) if nc > 0 and p < 0 else 0.0,
         "gap_loss_mult": _gap_loss_mult(position, chain, enriched, cfg.risk.amber_gap, cfg),
         "event": _event_within(events, d, holidays, cfg),
         "wall_shift": _wall_shift(position, chain, enriched, cfg)}
    level, triggers = level_from_metrics(m, position.baseline, position.structure, cfg)
    metrics = dict(m, pnl=p, mark=mark(position, enriched), pnl_pct=p / nc if nc > 0 else 0.0,
                   spot=chain.spot, days_left=(position.expiry - d).days,
                   stress=stress(position, chain, enriched, cfg))
    return RiskReport(level, triggers, metrics)


def candidate_position(c: Candidate, expiry: date) -> Position:
    """Position view of a not-yet-entered candidate."""
    return Position(None, c.structure, expiry, c.legs, c.credit, 0, {})


def pre_entry_check(candidate: Candidate, chain, enriched, fc: Forecast, events, d: date, cfg,
                    holidays: set = frozenset()) -> tuple[bool, list[str]]:
    """§12.1: fail on 3-day touch of a danger level, gap loss vs credit, or a near event.
    Config: risk.pre_entry.*, risk.event_days, risk.window_days."""
    pe, naked = cfg.risk.pre_entry, candidate.structure == NAKED_STRANGLE
    pos = candidate_position(candidate, chain.expiry)
    reasons = []
    lim = pe.naked_touch if naked else pe.touch
    for side, s in side_metrics(pos, chain, enriched, fc, cfg).items():
        if s["touch_3d"] > lim:
            reasons.append(f"touch_3d {side} {s['touch_3d']:.2f} > {lim:.2f}")
    gap, mult = (pe.naked_gap, pe.naked_gap_loss_mult) if naked else (pe.gap, pe.gap_loss_mult)
    g = _gap_loss_mult(pos, chain, enriched, gap, cfg)
    if g > mult:
        reasons.append(f"±{gap:.0%} gap loss {g:.2f}× credit > {mult}×")
    ev = _event_within(events, d, holidays, cfg)
    if ev:
        reasons.append(f"event within {cfg.risk.event_days} trading days: {ev}")
    return not reasons, reasons


def baseline_for(position: Position, chain, enriched, fc: Forecast, cfg) -> dict:
    """Entry-day values stored with a logged trade: per-side touch_3d and delta, the wall each
    short sits at/behind (strike + OI), and entry spot."""
    sides = side_metrics(position, chain, enriched, fc, cfg)
    walls = find_walls(enriched, chain.spot, cfg.scoring.oi_window_pct, cfg.walls.oi_multiple, cfg.walls.top_n)
    w_k, w_oi = {}, {}
    for side, l in _shorts(position).items():
        behind = [w for w in walls[side] if (w <= l.strike if side == "put" else w >= l.strike)]
        if behind:
            k = min(behind, key=lambda w: abs(w - l.strike))
            w_k[side], w_oi[side] = k, float(_row(enriched, SIDE_TYPE[side], k).oi)
    return {"touch_3d": {s: v["touch_3d"] for s, v in sides.items()},
            "delta": {s: v["delta"] for s, v in sides.items()},
            "wall_strike": w_k, "wall_oi": w_oi, "entry_spot": chain.spot}


# ---------- action calls and rolls (§12 exit list, §12.2) ----------

from advisor.calendar import calendar_days_to  # noqa: E402
from advisor.strategy import PARAMS, choose_wing, pick_short_strike  # noqa: E402

EPS = 1e-9


@dataclass
class Roll:
    side: str
    close_legs: list[Leg]
    open_legs: list[Leg]
    net_cash: float           # points per unit; > 0 = credit received


@dataclass
class Action:
    kind: str                 # HOLD | HOLD_PREPARE | EXIT | ADJUST
    reason: str
    roll: Roll | None = None


def propose_roll(position: Position, side: str, chain, enriched, fc: Forecast, cfg) -> Roll | None:
    """Roll the tested side out (§12.2): close its short and wing, open a new short at today's
    strike-selection rule (iron butterflies use strikes.iron_condor) with a wing of the same width.
    None if the new short is not further from spot or any new leg is illiquid/unlisted."""
    typ = SIDE_TYPE[side]
    short = next((l for l in position.legs if l.side == "SELL" and l.option_type == typ), None)
    if short is None:
        return None
    wing = next((l for l in position.legs if l.side == "BUY" and l.option_type == typ), None)
    key = PARAMS[position.structure][0 if side == "put" else 1]
    p = getattr(cfg.strikes, "iron_condor" if key in (None, "atm") else key)
    walls = find_walls(enriched, chain.spot, cfg.scoring.oi_window_pct, cfg.walls.oi_multiple, cfg.walls.top_n)
    k, _ = pick_short_strike(enriched, fc, side, p.q, p.delta, walls, cfg)
    if not np.isfinite(k) or not (k < short.strike if side == "put" else k > short.strike):
        return None
    row = _row(enriched, typ, k)
    if row is None or not np.isfinite(row.iv) or row.oi < cfg.liquidity.min_oi:
        return None
    new = [Leg(typ, float(k), "SELL", short.lots, float(row.settle))]
    close = [short]
    if wing is not None:
        w = choose_wing(enriched, k, side, abs(short.strike - wing.strike), cfg)
        if w is None:
            return None
        w.lots = wing.lots
        new = [w] + new
        close = [wing, short]

    def px(l):
        r = _row(enriched, l.option_type, l.strike)
        return float(r.settle) if r is not None else l.price

    close_cost = sum(px(l) if l.side == "SELL" else -px(l) for l in close)
    open_credit = sum(l.price if l.side == "SELL" else -l.price for l in new)
    return Roll(side, close, new, open_credit - close_cost)


def _tested_side(report: RiskReport) -> str | None:
    sides = report.metrics.get("sides", {})
    if not sides:
        return None
    return max(sorted(sides), key=lambda s: sides[s].get("touch_3d", 0.0))


def decide_action(position: Position, report: RiskReport, chain, enriched, fc: Forecast, d: date, cfg,
                  events=(), holidays: set = frozenset()) -> Action:
    """Action call in fixed order (§12): hard stop → RED (adjust if a credit roll leaves it not RED,
    else exit) → profit target → time exit → AMBER (hold, prepare roll) → hold.
    Config: exits.hard_stop_mult/profit_target/time_exit_days/max_rolls."""
    ex, nc = cfg.exits, position.net_credit
    p = pnl(position, enriched)
    if nc > 0 and -p >= ex.hard_stop_mult * nc - EPS:
        return Action("EXIT", "hard stop")
    if report.level == "RED":
        side = _tested_side(report)
        if side and position.rolls < ex.max_rolls:
            roll = propose_roll(position, side, chain, enriched, fc, cfg)
            if roll is not None and roll.net_cash > 0:
                if _roll_ok(position, roll, chain, enriched, fc, d, cfg, events, holidays):
                    return Action("ADJUST", f"red: roll {side} side", roll)
        return Action("EXIT", "red")
    if nc > 0 and p >= ex.profit_target * nc - EPS:
        return Action("EXIT", "profit target")
    if calendar_days_to(d, position.expiry) <= ex.time_exit_days:
        return Action("EXIT", "time exit")
    if report.level == "AMBER":
        side = _tested_side(report)
        roll = propose_roll(position, side, chain, enriched, fc, cfg) if side else None
        return Action("HOLD_PREPARE", "amber: " + "; ".join(report.triggers), roll)
    return Action("HOLD", "green")


def apply_roll_to_position(position: Position, roll: Roll) -> Position:
    """New Position after a roll: closed legs removed, new legs added, running credit updated."""
    closed = {id(l) for l in roll.close_legs}
    legs = [l for l in position.legs if id(l) not in closed] + list(roll.open_legs)
    return Position(position.trade_id, position.structure, position.expiry, legs,
                    position.net_credit + roll.net_cash, position.rolls + 1, dict(position.baseline))


def _roll_ok(position, roll, chain, enriched, fc, d, cfg, events, holidays) -> bool:
    new = apply_roll_to_position(position, roll)
    new.baseline = baseline_for(new, chain, enriched, fc, cfg)
    return assess(new, chain, enriched, fc, events, d, cfg, holidays).level != "RED"
