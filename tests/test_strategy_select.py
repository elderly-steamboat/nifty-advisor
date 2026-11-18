from datetime import date

import pytest

from advisor.config import Event
from advisor.scoring import DirectionScore, Edge
from advisor.strategy import (BEAR_CALL, BULL_PUT, IRON_BUTTERFLY, IRON_CONDOR, NAKED_STRANGLE,
                              SKEW_IC_BEAR, SKEW_IC_BULL, select_structures)

D, E = date(2026, 11, 18), date(2026, 12, 29)


def ds(v, band):
    return DirectionScore(v, band, [])


def edge(r=1.2, ivr=40, iw=1000.0, kw=None):
    return Edge(r, ivr, r >= 1.10 and ivr >= 30, iw, kw if kw is not None else iw / r)


def test_too_close_to_expiry(tmp_cfg):
    s = select_structures(D, date(2026, 11, 25), ds(0, "neutral"), edge(), [], tmp_cfg)
    assert s.structures == [] and s.no_trade_reason == "too close to expiry"


def test_edge_low_no_trade(tmp_cfg):
    s = select_structures(D, E, ds(0, "neutral"), edge(r=1.05), [], tmp_cfg)
    assert s.structures == [] and s.no_trade_reason.startswith("premium edge low")


@pytest.mark.parametrize("band,expected", [
    ("strong_bearish", [BEAR_CALL]), ("mild_bearish", [SKEW_IC_BEAR]),
    ("mild_bullish", [SKEW_IC_BULL]), ("strong_bullish", [BULL_PUT])])
def test_band_map(tmp_cfg, band, expected):
    v = {"strong_bearish": -60, "mild_bearish": -30, "mild_bullish": 30, "strong_bullish": 60}[band]
    assert select_structures(D, E, ds(v, band), edge(), [], tmp_cfg).structures == expected


def test_neutral_plain_condor(tmp_cfg):
    s = select_structures(D, E, ds(5, "neutral"), edge(r=1.2, ivr=40), [], tmp_cfg)
    assert s.structures == [IRON_CONDOR]
    assert s.cell == "neutral/edge_good"


def test_bfly_and_naked_order(tmp_cfg):
    s = select_structures(D, E, ds(5, "neutral"), edge(r=1.8, ivr=60, kw=1000 / 1.8 * 1.0), [], tmp_cfg)
    assert s.structures == [IRON_BUTTERFLY, NAKED_STRANGLE, IRON_CONDOR]


def test_bfly_without_naked(tmp_cfg):
    s = select_structures(D, E, ds(5, "neutral"), edge(r=1.8, ivr=40), [], tmp_cfg)
    assert s.structures == [IRON_BUTTERFLY, IRON_CONDOR]


def test_naked_only(tmp_cfg):
    s = select_structures(D, E, ds(5, "neutral"), edge(r=1.35, ivr=55), [], tmp_cfg)
    assert s.structures == [NAKED_STRANGLE, IRON_CONDOR]
    assert "naked_gate: pass" in s.gates


def test_naked_blocked_by_event(tmp_cfg):
    ev = [Event(date(2026, 12, 4), "RBI", "rbi")]
    s = select_structures(D, E, ds(5, "neutral"), edge(r=1.35, ivr=55), ev, tmp_cfg)
    assert s.structures == [IRON_CONDOR]
    assert any(g.startswith("naked_gate: fail") and "event" in g for g in s.gates)


def test_event_after_expiry_ignored(tmp_cfg):
    ev = [Event(date(2027, 1, 5), "x", "other")]
    s = select_structures(D, E, ds(5, "neutral"), edge(r=1.35, ivr=55), ev, tmp_cfg)
    assert s.structures[0] == NAKED_STRANGLE


def test_never_naked_straddle(tmp_cfg):
    for r, ivr in ((1.2, 40), (1.35, 55), (2.0, 90)):
        s = select_structures(D, E, ds(0, "neutral"), edge(r=r, ivr=ivr), [], tmp_cfg)
        assert "NAKED_STRADDLE" not in s.structures
