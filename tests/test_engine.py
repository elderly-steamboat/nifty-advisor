import json
from datetime import date

import pytest

import advisor.engine as engine
from advisor.analytics import enrich_chain
from advisor.engine import (check_positions, evaluate_structures, log_trade, recommend,
                            recommendation_json)
from advisor.journal import get_open_positions, get_trade, open_db
from advisor.strategy import IRON_CONDOR, NAKED_STRANGLE, Leg
from tests.conftest import make_chain, make_fake_sampler, uniform_forecast

D, E = date(2026, 11, 17), date(2026, 12, 29)
HOL = {date(2026, 11, 24), date(2026, 12, 25)}


def test_recommend_stale_data(seeded_cfg, fake_sampler):
    import pandas as pd
    p = seeded_cfg.data_dir / "index_daily.csv"
    df = pd.read_csv(p)
    df.loc[df.date == "2026-11-17", "vix_close"] = None
    df.to_csv(p, index=False)
    rec = recommend(D, E, seeded_cfg, sampler=fake_sampler, holidays=HOL, events=[])
    assert not rec.data_ok and "vix" in rec.data_reason and rec.primary is None


def test_recommend_runs_on_real_fixture(seeded_cfg, fake_sampler):
    rec = recommend(D, E, seeded_cfg, sampler=fake_sampler, holidays=HOL, events=[])
    assert rec.data_ok
    assert rec.spot == 25387.4 and rec.direction is not None and rec.edge is not None
    assert rec.selection is not None
    assert rec.primary is not None or rec.selection.no_trade_reason or rec.rejected
    j = json.loads(recommendation_json(rec))
    assert j["config_hash"] == rec.config_hash and j["date"] == "2026-11-17"


def test_recommend_deterministic(seeded_cfg, fake_sampler):
    a = recommendation_json(recommend(D, E, seeded_cfg, sampler=fake_sampler, holidays=HOL, events=[]))
    for f in (seeded_cfg.data_dir / "forecasts").glob("*"):
        f.unlink()
    b = recommendation_json(recommend(D, E, seeded_cfg, sampler=make_fake_sampler(), holidays=HOL, events=[]))
    assert a == b


def test_recommend_too_close(seeded_cfg, fake_sampler):
    rec = recommend(D, date(2026, 11, 23), seeded_cfg, sampler=fake_sampler, holidays=HOL, events=[])
    assert rec.primary is None and rec.selection.no_trade_reason == "too close to expiry"


def test_downgrade_naked_to_condor(tmp_cfg, monkeypatch):
    ch = make_chain(d=date(2026, 11, 18))
    e = enrich_chain(ch, tmp_cfg.risk_free_rate)
    fc = uniform_forecast(half=600)
    real = engine.pre_entry_check
    monkeypatch.setattr(engine, "pre_entry_check",
                        lambda c, *a, **k: (False, ["forced"]) if c.structure == NAKED_STRANGLE else real(c, *a, **k))
    surv, rejected = evaluate_structures([NAKED_STRANGLE, IRON_CONDOR], e, ch, fc, [], ch.date, tmp_cfg, set(), 0.0)
    from advisor.strategy import build_candidate
    naked, _ = build_candidate(NAKED_STRANGLE, e, ch, fc, tmp_cfg, 0.0)
    assert surv[0].structure == IRON_CONDOR
    assert {l.strike for l in surv[0].legs if l.side == "SELL"} == {l.strike for l in naked.legs}
    assert any("NAKED_STRANGLE" in r and "forced" in r for r in rejected)
    assert len(surv) == 1 or {l.strike for l in surv[1].legs} != {l.strike for l in surv[0].legs}


def test_downgrade_defined_moves_shorts_out(tmp_cfg, monkeypatch):
    ch = make_chain(d=date(2026, 11, 18))
    e = enrich_chain(ch, tmp_cfg.risk_free_rate)
    fc = uniform_forecast(half=600)
    calls = []
    def fake(c, *a, **k):
        calls.append(sorted(l.strike for l in c.legs if l.side == "SELL"))
        return (len(calls) > 1, ["forced"])
    monkeypatch.setattr(engine, "pre_entry_check", fake)
    surv, _ = evaluate_structures([IRON_CONDOR], e, ch, fc, [], ch.date, tmp_cfg, set(), 0.0)
    assert len(surv) == 1
    assert calls[1][0] == calls[0][0] - 50 and calls[1][1] == calls[0][1] + 50


def _legs_from_chain(seeded_cfg):
    from advisor.data import load_chain
    ch = load_chain(D, E, seeded_cfg)
    px = lambda t, k: float(ch.rows[(ch.rows.option_type == t) & (ch.rows.strike == k)].settle.iloc[0])
    return [Leg("PE", 24000.0, "BUY", 1, px("PE", 24000.0)), Leg("PE", 24200.0, "SELL", 1, px("PE", 24200.0)),
            Leg("CE", 26800.0, "SELL", 1, px("CE", 26800.0)), Leg("CE", 27000.0, "BUY", 1, px("CE", 27000.0))]


def test_log_trade_rejects_unknown_leg(seeded_cfg, fake_sampler, tmp_path):
    conn = open_db(tmp_path / "j.sqlite")
    legs = _legs_from_chain(seeded_cfg)
    legs[1] = Leg("PE", 24210.0, "SELL", 1, 50.0)
    with pytest.raises(ValueError, match="leg not in chain: PE 24210 2026-12-29"):
        log_trade(conn, D, IRON_CONDOR, E, legs, seeded_cfg, sampler=fake_sampler, holidays=HOL)
    assert conn.execute("select count(*) from trades").fetchone()[0] == 0


def test_log_trade_and_check_positions(seeded_cfg, fake_sampler, tmp_path):
    conn = open_db(tmp_path / "j.sqlite")
    tid = log_trade(conn, D, IRON_CONDOR, E, _legs_from_chain(seeded_cfg), seeded_cfg,
                    sampler=fake_sampler, holidays=HOL)
    [p] = get_open_positions(conn)
    assert p.baseline["entry_spot"] == 25387.4 and set(p.baseline["delta"]) == {"put", "call"}
    out = check_positions(conn, D, seeded_cfg, sampler=fake_sampler, holidays=HOL, events=[])
    [(pos, report, action)] = out
    assert report.level in ("GREEN", "AMBER", "RED") and action.kind in ("HOLD", "HOLD_PREPARE", "EXIT", "ADJUST")
    assert len(get_trade(conn, tid)["snapshots"]) == 1
