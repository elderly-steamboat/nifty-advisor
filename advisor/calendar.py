# Trading-day arithmetic (weekends + NSE holidays) and event-calendar lookups.
from __future__ import annotations

from datetime import date, timedelta

from advisor.config import Event


def is_trading_day(d: date, holidays: set[date]) -> bool:
    """True if d is a weekday and not an NSE holiday."""
    return d.weekday() < 5 and d not in holidays


def trading_days_between(start: date, end: date, holidays: set[date]) -> int:
    """Count trading days in (start, end] — start excluded, end included."""
    n, d = 0, start + timedelta(days=1)
    while d <= end:
        n += is_trading_day(d, holidays)
        d += timedelta(days=1)
    return n


def next_trading_days(start: date, n: int, holidays: set[date]) -> list[date]:
    """The next n trading days after start (start excluded), in order."""
    out, d = [], start
    while len(out) < n:
        d += timedelta(days=1)
        if is_trading_day(d, holidays):
            out.append(d)
    return out


def events_between(events: list[Event], start: date, end: date) -> list[Event]:
    """Events with start <= date <= end."""
    return [e for e in events if start <= e.date <= end]


def calendar_days_to(d: date, expiry: date) -> int:
    """Calendar days from d to expiry."""
    return (expiry - d).days
