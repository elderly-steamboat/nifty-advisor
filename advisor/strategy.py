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
