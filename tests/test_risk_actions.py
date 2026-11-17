from datetime import date

import pytest

from advisor.analytics import enrich_chain
from advisor.risk import (Position, RiskReport, assess, baseline_for, decide_action, propose_roll)
from advisor.strategy import IRON_CONDOR, NAKED_STRANGLE, Leg
from tests.conftest import cfg_with, make_chain, uniform_forecast

D = date(2026, 11, 18)
R = 0.065


def _px(e, typ, k):
    return float(e[(e.option_type == typ) & (e.strike == k)].settle.iloc[0])


def pos_from(e, put=23000.0, call=25000.0, width=200.0, credit=None, rolls=0, expiry=None):
    legs = [Leg("PE", put - width, "BUY", 2, _px(e, "PE", put - width), id=1),
            Leg("PE", put, "SELL", 2, _px(e, "PE", put), id=2),
            Leg("CE", call, "SELL", 2, _px(e, "CE", call), id=3),
            Leg("CE", call + width, "BUY", 2, _px(e, "CE", call + width), id=4)]
    nc = credit if credit is not None else sum(l.price if l.side == "SELL" else -l.price for l in legs)
    return Position(7, IRON_CONDOR, expiry or date(2026, 12, 28), legs, nc, rolls, {})


def rep(level, put_touch=0.05, call_touch=0.05):
    return RiskReport(level, [], {"sides": {"put": {"touch_3d": put_touch}, "call": {"touch_3d": call_touch}}})


@pytest.fixture
def env(tmp_cfg):
    ch = make_chain(d=D)
    return ch, enrich_chain(ch, R), tmp_cfg, uniform_forecast(half=600, d=D)


def test_hold_when_green(env):
    ch, e, cfg, fc = env
    a = decide_action(pos_from(e), rep("GREEN"), ch, e, fc, D, cfg)
    assert a.kind == "HOLD" and a.roll is None


def test_hard_stop(env):
    ch, e, cfg, fc = env
    p = pos_from(e)
    p.net_credit = (sum(l.price if l.side == "SELL" else -l.price for l in p.legs)) / 3  # loss = 2× credit
    a = decide_action(p, rep("GREEN"), ch, e, fc, D, cfg)
    assert a.kind == "EXIT" and a.reason == "hard stop"


def test_order_hard_stop_beats_profit_and_red(env):
    ch, e, cfg, fc = env
    p = pos_from(e)
    p.net_credit = (sum(l.price if l.side == "SELL" else -l.price for l in p.legs)) / 3
    assert decide_action(p, rep("RED"), ch, e, fc, D, cfg).reason == "hard stop"


def test_profit_target(env):
    ch, e, cfg, fc = env
    p = pos_from(e)
    p.net_credit *= 2.2  # mark = credit/2.2 → pnl ≈ 55% of credit
    a = decide_action(p, rep("GREEN"), ch, e, fc, D, cfg)
    assert a.kind == "EXIT" and a.reason == "profit target"


def test_time_exit(env):
    ch, e, cfg, fc = env
    p = pos_from(e, expiry=date(2026, 11, 25))
    a = decide_action(p, rep("GREEN"), ch, e, fc, D, cfg)
    assert a.kind == "EXIT" and a.reason == "time exit"


def test_amber_attaches_roll_suggestion(env):
    ch, e, cfg, fc = env
    p = pos_from(e, put=23700.0)
    a = decide_action(p, rep("AMBER", put_touch=0.2), ch, e, fc, D, cfg)
    assert a.kind == "HOLD_PREPARE"
    assert a.roll is None or a.roll.side == "put"


def test_propose_roll_moves_tested_side_out(env):
    ch, e, cfg, fc = env
    p = pos_from(e, put=23700.0)
    r = propose_roll(p, "put", ch, e, fc, cfg)
    assert r is not None
    assert {l.strike for l in r.close_legs} == {23700.0, 23500.0}
    new_short = next(l for l in r.open_legs if l.side == "SELL")
    new_wing = next(l for l in r.open_legs if l.side == "BUY")
    assert new_short.strike < 23700.0 and new_short.strike - new_wing.strike == 200.0
    assert all(l.lots == 2 for l in r.open_legs)
    expect = (new_short.price - new_wing.price) - (_px(e, "PE", 23700.0) - _px(e, "PE", 23500.0))
    assert r.net_cash == pytest.approx(expect)


def test_roll_requires_net_credit(env):
    ch, e, cfg, fc = env
    p = pos_from(e, put=23700.0)
    a = decide_action(p, rep("RED", put_touch=0.5), ch, e, fc, D, cfg)
    r = propose_roll(p, "put", ch, e, fc, cfg)
    assert r is not None and r.net_cash <= 0  # rolling the tested side out costs a debit
    assert a.kind == "EXIT" and a.reason == "red"


def test_red_adjust_when_roll_is_credit(env, monkeypatch):
    ch, e, cfg, fc = env
    import advisor.risk as risk
    from advisor.risk import Roll
    p = pos_from(e, put=23700.0)
    fake = Roll("put", [p.legs[0], p.legs[1]],
                [Leg("PE", 22800.0, "BUY", 2, 5.0), Leg("PE", 23000.0, "SELL", 2, 20.0)], 3.0)
    monkeypatch.setattr(risk, "propose_roll", lambda *a, **k: fake)
    a = decide_action(p, rep("RED", put_touch=0.5), ch, e, fc, D, cfg)
    assert a.kind == "ADJUST" and a.roll is fake


def test_third_roll_becomes_exit(env, monkeypatch):
    ch, e, cfg, fc = env
    import advisor.risk as risk
    from advisor.risk import Roll
    p = pos_from(e, put=23700.0, rolls=2)
    monkeypatch.setattr(risk, "propose_roll", lambda *a, **k: Roll("put", [], [], 5.0))
    a = decide_action(p, rep("RED", put_touch=0.5), ch, e, fc, D, cfg)
    assert a.kind == "EXIT" and a.reason == "red"


def test_naked_roll_short_only(env):
    ch, e, cfg, fc = env
    legs = [Leg("PE", 23700.0, "SELL", 1, _px(e, "PE", 23700.0)), Leg("CE", 25000.0, "SELL", 1, _px(e, "CE", 25000.0))]
    p = Position(9, NAKED_STRANGLE, date(2026, 12, 28), legs, sum(l.price for l in legs), 0, {})
    r = propose_roll(p, "put", ch, e, fc, cfg)
    assert r is not None and len(r.close_legs) == 1 and len(r.open_legs) == 1
