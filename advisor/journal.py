# Trade journal in SQLite: trades, legs, cash events (entry/roll/exit), daily risk snapshots,
# and every recommendation produced. Plain functions; one transaction per write.
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from advisor.risk import Action, Position, RiskReport, Roll
from advisor.strategy import Leg

SCHEMA = """
create table if not exists trades(
  id integer primary key, structure text not null, expiry text not null, entry_date text not null,
  status text not null default 'open', lot_size integer not null, rolls integer not null default 0,
  baseline_json text not null default '{}', notes text not null default '',
  exit_date text, exit_reason text);
create table if not exists legs(
  id integer primary key, trade_id integer not null references trades(id), option_type text not null,
  strike real not null, side text not null, lots integer not null, entry_date text not null,
  price real not null, exit_date text, exit_price real);
create table if not exists events(
  id integer primary key, trade_id integer not null references trades(id), date text not null,
  type text not null, net_cash real not null, reason text not null default '');
create table if not exists snapshots(
  id integer primary key, trade_id integer not null references trades(id), date text not null,
  mark real, pnl real, level text, action text, reason text, metrics_json text);
create table if not exists recommendations(
  id integer primary key, date text not null, expiry text not null, config_hash text not null,
  json text not null);
"""


def open_db(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the journal database."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _cash(legs: list[Leg], use_exit: bool = False) -> float:
    px = (lambda l: l.exit_price) if use_exit else (lambda l: l.price)
    return sum(px(l) if l.side == "SELL" else -px(l) for l in legs)


def _insert_legs(conn, trade_id: int, legs: list[Leg], d: date) -> None:
    for l in legs:
        conn.execute("insert into legs(trade_id, option_type, strike, side, lots, entry_date, price) "
                     "values (?,?,?,?,?,?,?)",
                     (trade_id, l.option_type, float(l.strike), l.side, int(l.lots), d.isoformat(), float(l.price)))


def add_trade(conn, structure: str, expiry: date, entry_date: date, legs: list[Leg], baseline: dict,
              lot_size: int, notes: str = "") -> int:
    """Insert a trade with its legs and an 'entry' event (net_cash = Σ SELL − Σ BUY prices)."""
    with conn:
        cur = conn.execute("insert into trades(structure, expiry, entry_date, lot_size, baseline_json, notes) "
                           "values (?,?,?,?,?,?)",
                           (structure, expiry.isoformat(), entry_date.isoformat(), int(lot_size),
                            json.dumps(baseline, sort_keys=True), notes))
        tid = cur.lastrowid
        _insert_legs(conn, tid, legs, entry_date)
        conn.execute("insert into events(trade_id, date, type, net_cash, reason) values (?,?,?,?,?)",
                     (tid, entry_date.isoformat(), "entry", _cash(legs), ""))
    return tid


def _leg(r) -> Leg:
    return Leg(r["option_type"], r["strike"], r["side"], r["lots"], r["price"],
               date.fromisoformat(r["entry_date"]),
               r["exit_price"], date.fromisoformat(r["exit_date"]) if r["exit_date"] else None, r["id"])


def get_open_positions(conn) -> list[Position]:
    """Open trades as Positions (open legs only; net_credit = Σ entry + roll cash)."""
    out = []
    for t in conn.execute("select * from trades where status='open' order by id"):
        legs = [_leg(r) for r in conn.execute(
            "select * from legs where trade_id=? and exit_date is null order by id", (t["id"],))]
        nc = conn.execute("select coalesce(sum(net_cash),0) from events where trade_id=? and type in ('entry','roll')",
                          (t["id"],)).fetchone()[0]
        out.append(Position(t["id"], t["structure"], date.fromisoformat(t["expiry"]), legs, float(nc),
                            t["rolls"], json.loads(t["baseline_json"])))
    return out


def get_trade(conn, trade_id: int) -> dict:
    """Trade row with all legs, events and snapshots (dicts)."""
    t = conn.execute("select * from trades where id=?", (trade_id,)).fetchone()
    if t is None:
        raise ValueError(f"no trade {trade_id}")
    q = lambda sql: [dict(r) for r in conn.execute(sql, (trade_id,))]
    return dict(t) | {"legs": q("select * from legs where trade_id=? order by id"),
                      "events": q("select * from events where trade_id=? order by id"),
                      "snapshots": q("select * from snapshots where trade_id=? order by id")}


def _close_legs(conn, trade_id, legs: list[Leg], prices: dict[int, float], d: date) -> float:
    cash = 0.0
    for l in legs:
        if l.id not in prices:
            raise ValueError(f"exit price missing for leg {l.id} ({l.option_type} {l.strike:.0f})")
    for l in legs:
        px = float(prices[l.id])
        conn.execute("update legs set exit_date=?, exit_price=? where id=? and trade_id=?",
                     (d.isoformat(), px, l.id, trade_id))
        cash -= px if l.side == "SELL" else -px
    return cash


def apply_roll(conn, trade_id: int, roll: Roll, d: date, close_prices: dict[int, float]) -> None:
    """Record a roll: close roll.close_legs at close_prices, open roll.open_legs at their prices,
    add a 'roll' event with the actual net cash, increment rolls."""
    with conn:
        cash = _close_legs(conn, trade_id, roll.close_legs, close_prices, d)
        _insert_legs(conn, trade_id, roll.open_legs, d)
        cash += _cash(roll.open_legs)
        conn.execute("insert into events(trade_id, date, type, net_cash, reason) values (?,?,?,?,?)",
                     (trade_id, d.isoformat(), "roll", cash, f"roll {roll.side}"))
        conn.execute("update trades set rolls = rolls + 1 where id=?", (trade_id,))


def close_trade(conn, trade_id: int, exit_prices: dict[int, float], d: date, reason: str) -> None:
    """Close every open leg at exit_prices (all required) and mark the trade closed."""
    open_legs = [_leg(r) for r in conn.execute(
        "select * from legs where trade_id=? and exit_date is null", (trade_id,))]
    with conn:
        cash = _close_legs(conn, trade_id, open_legs, exit_prices, d)
        conn.execute("insert into events(trade_id, date, type, net_cash, reason) values (?,?,?,?,?)",
                     (trade_id, d.isoformat(), "exit", cash, reason))
        conn.execute("update trades set status='closed', exit_date=?, exit_reason=? where id=?",
                     (d.isoformat(), reason, trade_id))


def save_snapshot(conn, trade_id: int, d: date, mark: float, pnl: float, report: RiskReport, action: Action) -> None:
    """Store one day's risk check for a trade."""
    with conn:
        conn.execute("insert into snapshots(trade_id, date, mark, pnl, level, action, reason, metrics_json) "
                     "values (?,?,?,?,?,?,?,?)",
                     (trade_id, d.isoformat(), mark, pnl, report.level, action.kind, action.reason,
                      json.dumps({"triggers": report.triggers, "metrics": report.metrics}, default=str, sort_keys=True)))


def save_recommendation(conn, d: date, expiry: date, config_hash: str, rec_json: str) -> None:
    """Persist a recommendation's JSON for audit/reproduction."""
    with conn:
        conn.execute("insert into recommendations(date, expiry, config_hash, json) values (?,?,?,?)",
                     (d.isoformat(), expiry.isoformat(), config_hash, rec_json))


def list_closed(conn) -> pd.DataFrame:
    """Closed trades with realised P&L in points per unit and INR."""
    rows = conn.execute("""
      select t.id, t.structure, t.expiry, t.entry_date, t.exit_date, t.exit_reason, t.lot_size, t.rolls,
             (select max(lots) from legs where trade_id=t.id) as lots,
             (select net_cash from events where trade_id=t.id and type='entry') as credit,
             (select sum(net_cash) from events where trade_id=t.id) as pnl_points
      from trades t where t.status='closed' order by t.exit_date, t.id""").fetchall()
    df = pd.DataFrame([dict(r) for r in rows],
                      columns=["id", "structure", "expiry", "entry_date", "exit_date", "exit_reason",
                               "lot_size", "rolls", "lots", "credit", "pnl_points"])
    df["pnl_inr"] = df.pnl_points * df.lot_size * df.lots
    return df
