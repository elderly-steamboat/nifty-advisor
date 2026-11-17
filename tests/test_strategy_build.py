from datetime import date

import numpy as np
import pytest

from advisor.analytics import enrich_chain, find_walls
from advisor.forecast import terminal_quantiles
from advisor.strategy import (BEAR_CALL, BULL_PUT, IRON_BUTTERFLY, IRON_CONDOR, NAKED_STRANGLE,
                              SKEW_IC_BULL, Leg, build_candidate, cap_by_margin, choose_wing,
                              payoff, pick_short_strike, size_defined, size_naked)
from tests.conftest import cfg_with, make_chain, uniform_forecast

R = 0.065


@pytest.fixture
def setup(tmp_cfg):
    ch = make_chain()
    return ch, enrich_chain(ch, R), tmp_cfg


def _walls(e, cfg, spot=24000.0):
    return find_walls(e, spot, cfg.scoring.oi_window_pct, cfg.walls.oi_multiple, cfg.walls.top_n)


def test_short_strike_further_of_kronos_and_delta(setup):
    ch, e, cfg = setup
    narrow = uniform_forecast(half=600)
    k, why = pick_short_strike(e, narrow, "put", 0.10, 0.20, _walls(e, cfg), cfg)
    assert why == "delta" and k < 24000
    wide = uniform_forecast(half=3000)
    k2, why2 = pick_short_strike(e, wide, "put", 0.10, 0.20, _walls(e, cfg), cfg)
    assert why2 == "kronos" and k2 <= terminal_quantiles(wide, [0.10])[0.10]
    kc, whyc = pick_short_strike(e, wide, "call", 0.10, 0.20, _walls(e, cfg), cfg)
    assert kc >= terminal_quantiles(wide, [0.90])[0.90] and kc % 50 == 0


def test_wall_moves_short_out(tmp_cfg):
    fc = uniform_forecast(half=1000)
    e0 = enrich_chain(make_chain(), R)
    base, _ = pick_short_strike(e0, fc, "call", 0.10, 0.20, _walls(e0, tmp_cfg), tmp_cfg)
    wall = base + 100
    e = enrich_chain(make_chain(spikes={(wall, "CE"): 900000}), R)
    k, why = pick_short_strike(e, fc, "call", 0.10, 0.20, _walls(e, tmp_cfg), tmp_cfg)
    assert (k, why) == (wall, "wall")


def test_wall_inside_base_ignored(tmp_cfg):
    fc = uniform_forecast(half=1000)
    e0 = enrich_chain(make_chain(), R)
    base, why0 = pick_short_strike(e0, fc, "call", 0.10, 0.20, _walls(e0, tmp_cfg), tmp_cfg)
    e = enrich_chain(make_chain(spikes={(base - 150, "CE"): 900000}), R)
    assert pick_short_strike(e, fc, "call", 0.10, 0.20, _walls(e, tmp_cfg), tmp_cfg) == (base, why0)


def test_wall_too_far_ignored(tmp_cfg):
    fc = uniform_forecast(half=1000)
    e = enrich_chain(make_chain(spikes={(25600.0, "CE"): 900000}), R)
    k, why = pick_short_strike(e, fc, "call", 0.10, 0.20, _walls(e, tmp_cfg), tmp_cfg)
    assert k < 25600 and why.endswith("(wall ignored: too far)")


def test_payoff_iron_condor():
    legs = [Leg("PE", 23000, "BUY", 1, 10), Leg("PE", 23200, "SELL", 1, 30),
            Leg("CE", 24800, "SELL", 1, 30), Leg("CE", 25000, "BUY", 1, 10)]
    s = np.array([22000.0, 24000.0, 26000.0])
    assert payoff(legs, s).tolist() == [-160.0, 40.0, -160.0]


def test_condor_candidate(setup):
    ch, e, cfg = setup
    c, why = build_candidate(IRON_CONDOR, e, ch, uniform_forecast(half=600), cfg, 0.0)
    assert c is not None, why
    assert [(l.option_type, l.side) for l in c.legs] == [("PE", "BUY"), ("PE", "SELL"), ("CE", "SELL"), ("CE", "BUY")]
    assert c.credit >= cfg.wings.min_credit_ratio * c.width
    assert c.max_loss == pytest.approx(c.width - c.credit)
    assert c.lots == size_defined(c.max_loss, ch.lot_size, cfg) or c.lots < size_defined(c.max_loss, ch.lot_size, cfg)
    assert 0 <= c.pop <= 1 and set(c.strike_reasons) == {"put", "call"}


def test_wing_choice_none_qualifies(setup):
    ch, e, cfg = setup
    strict = cfg_with(cfg, "wings", min_credit_ratio=0.99)
    c, why = build_candidate(IRON_CONDOR, e, ch, uniform_forecast(half=600), strict, 0.0)
    assert c is None and why == "no wing width meets credit ratio"


def test_spreads_single_side(tmp_cfg):
    cfg = tmp_cfg
    ch = make_chain(skew=-0.2)  # milder put skew so a 0.30Δ put spread meets the 25% credit ratio
    e = enrich_chain(ch, R)
    bp, _ = build_candidate(BULL_PUT, e, ch, uniform_forecast(half=600), cfg, 0.0)
    bc, _ = build_candidate(BEAR_CALL, e, ch, uniform_forecast(half=600), cfg, 0.0)
    assert bp is not None and bc is not None
    assert {l.option_type for l in bp.legs} == {"PE"} and {l.option_type for l in bc.legs} == {"CE"}


def test_skewed_condor_favoured_side_closer(setup):
    ch, e, cfg = setup
    c, _ = build_candidate(SKEW_IC_BULL, e, ch, uniform_forecast(half=600), cfg, 0.0)
    put = next(l.strike for l in c.legs if l.option_type == "PE" and l.side == "SELL")
    call = next(l.strike for l in c.legs if l.option_type == "CE" and l.side == "SELL")
    assert 24000 - put < call - 24000


def test_butterfly_shorts_atm(setup):
    ch, e, cfg = setup
    c, why = build_candidate(IRON_BUTTERFLY, e, ch, uniform_forecast(half=400), cfg, 0.0)
    assert c is not None, why
    shorts = [l.strike for l in c.legs if l.side == "SELL"]
    assert shorts == [24000.0, 24000.0]
    assert c.credit >= cfg.wings.butterfly_min_credit_ratio * c.width


def test_naked_strangle_gate_and_size(setup):
    ch, e, cfg = setup
    c, why = build_candidate(NAKED_STRANGLE, e, ch, uniform_forecast(half=600), cfg, 0.0)
    assert c is not None, why
    assert c.max_loss is None and c.width is None and len(c.legs) == 2
    q = terminal_quantiles(uniform_forecast(half=600), [0.05, 0.95])
    assert min(l.strike for l in c.legs) <= q[0.05] and max(l.strike for l in c.legs) >= q[0.95]
    assert c.margin_est == cfg.sizing.naked_margin_per_lot * c.lots


def test_naked_gate_fails_when_shorts_inside(setup):
    ch, e, cfg = setup
    c, why = build_candidate(NAKED_STRANGLE, e, ch, uniform_forecast(half=600), cfg, 0.0,
                             shorts_override={"put": 23800.0, "call": 24200.0})
    assert c is None and why == "naked gate: shorts inside P5–P95"


def test_sizing_defined(tmp_cfg):
    assert size_defined(80.0, 65, tmp_cfg) == 3  # floor(20000 / 5200)


def test_sizing_naked(tmp_cfg):
    assert size_naked(300.0, 65, tmp_cfg) == 2  # floor(40000 / 19500)


def test_margin_cap_reduces_lots(tmp_cfg):
    assert cap_by_margin(10, 100000.0, 400000.0, tmp_cfg) == 1
    assert cap_by_margin(3, 100000.0, 0.0, tmp_cfg) == 3


def test_illiquid_short_returns_none(setup):
    ch, e, cfg = setup
    c0, _ = build_candidate(IRON_CONDOR, e, ch, uniform_forecast(half=600), cfg, 0.0)
    k = next(l.strike for l in c0.legs if l.option_type == "PE" and l.side == "SELL")
    e.loc[(e.strike == k) & (e.option_type == "PE"), "oi"] = 10
    c, why = build_candidate(IRON_CONDOR, e, ch, uniform_forecast(half=600), cfg, 0.0)
    assert c is None and why.startswith("illiquid short")


def test_choose_wing(setup):
    ch, e, cfg = setup
    w = choose_wing(e, 23000.0, "put", 200.0, cfg)
    assert w.strike == 22800.0 and w.side == "BUY" and w.option_type == "PE"


def test_size_below_one_lot(setup):
    ch, e, cfg = setup
    poor = cfg.model_copy(update={"capital": 1000.0})
    c, why = build_candidate(IRON_CONDOR, e, ch, uniform_forecast(half=600), poor, 0.0)
    assert c is None and why == "size < 1 lot"
