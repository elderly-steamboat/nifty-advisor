from datetime import date

import pytest

from advisor.journal import (add_trade, apply_roll, close_trade, get_open_positions, get_trade,
                             list_closed, open_db, save_recommendation, save_snapshot)
from advisor.risk import Action, RiskReport, Roll
from advisor.strategy import IRON_CONDOR, Leg

D = date(2026, 11, 18)


def legs():
    return [Leg("PE", 23000.0, "BUY", 2, 30.0), Leg("PE", 23200.0, "SELL", 2, 60.0),
            Leg("CE", 24800.0, "SELL", 2, 70.0), Leg("CE", 25000.0, "BUY", 2, 40.0)]


@pytest.fixture
def conn(tmp_path):
    return open_db(tmp_path / "j.sqlite")


def test_journal_roundtrip(conn):
    tid = add_trade(conn, IRON_CONDOR, date(2026, 12, 29), D, legs(), {"touch_3d": {"put": 0.1}}, 65, "note")
    [p] = get_open_positions(conn)
    assert p.trade_id == tid and p.structure == IRON_CONDOR and p.expiry == date(2026, 12, 29)
    assert p.net_credit == pytest.approx(60.0)
    assert p.baseline == {"touch_3d": {"put": 0.1}} and p.rolls == 0
    assert [l.strike for l in p.legs] == [23000.0, 23200.0, 24800.0, 25000.0]
    assert all(l.id is not None and l.entry_date == D for l in p.legs)


def test_roll_updates_running_credit(conn):
    tid = add_trade(conn, IRON_CONDOR, date(2026, 12, 29), D, legs(), {}, 65)
    [p] = get_open_positions(conn)
    put_wing, put_short = p.legs[0], p.legs[1]
    roll = Roll("put", [put_wing, put_short],
                [Leg("PE", 22600.0, "BUY", 2, 10.0), Leg("PE", 22800.0, "SELL", 2, 25.0)], 0.0)
    apply_roll(conn, tid, roll, date(2026, 11, 25), {put_wing.id: 20.0, put_short.id: 50.0})
    [p2] = get_open_positions(conn)
    # entry +60; roll: open +15, close −30 → net −15 → running 45
    assert p2.net_credit == pytest.approx(45.0) and p2.rolls == 1
    assert sorted(l.strike for l in p2.legs) == [22600.0, 22800.0, 24800.0, 25000.0]
    t = get_trade(conn, tid)
    closed = [l for l in t["legs"] if l["exit_date"]]
    assert {l["strike"] for l in closed} == {23000.0, 23200.0}
    assert [e["type"] for e in t["events"]] == ["entry", "roll"]


def test_close_trade_and_list_closed(conn):
    tid = add_trade(conn, IRON_CONDOR, date(2026, 12, 29), D, legs(), {}, 65)
    [p] = get_open_positions(conn)
    prices = {p.legs[0].id: 5.0, p.legs[1].id: 15.0, p.legs[2].id: 20.0, p.legs[3].id: 10.0}
    close_trade(conn, tid, prices, date(2026, 12, 10), "profit target")
    assert get_open_positions(conn) == []
    df = list_closed(conn)
    row = df.iloc[0]
    assert row.exit_reason == "profit target"
    assert row.pnl_points == pytest.approx(60.0 - 20.0)
    assert row.pnl_inr == pytest.approx(40.0 * 65 * 2)


def test_close_requires_all_open_legs(conn):
    tid = add_trade(conn, IRON_CONDOR, date(2026, 12, 29), D, legs(), {}, 65)
    [p] = get_open_positions(conn)
    with pytest.raises(ValueError, match="exit price missing"):
        close_trade(conn, tid, {p.legs[0].id: 1.0}, date(2026, 12, 10), "x")
    assert len(get_open_positions(conn)) == 1


def test_snapshot_and_recommendation_saved(conn):
    tid = add_trade(conn, IRON_CONDOR, date(2026, 12, 29), D, legs(), {}, 65)
    save_snapshot(conn, tid, D, 55.0, 5.0, RiskReport("GREEN", [], {"a": 1}), Action("HOLD", "green"))
    save_recommendation(conn, D, date(2026, 12, 29), "abc123", '{"x": 1}')
    t = get_trade(conn, tid)
    assert t["snapshots"][0]["level"] == "GREEN" and t["snapshots"][0]["action"] == "HOLD"
    assert conn.execute("select count(*) from recommendations").fetchone()[0] == 1
