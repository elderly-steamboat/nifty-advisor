from datetime import date

from advisor.calendar import (calendar_days_to, events_between, is_trading_day,
                              next_trading_days, trading_days_between)
from advisor.config import Event

H = {date(2026, 11, 24)}


def test_weekend_and_holiday_skipped():
    assert not is_trading_day(date(2026, 11, 21), H)  # Saturday
    assert not is_trading_day(date(2026, 11, 24), H)  # holiday
    assert is_trading_day(date(2026, 11, 23), H)
    assert trading_days_between(date(2026, 11, 20), date(2026, 11, 25), H) == 2  # 23, 25


def test_trading_days_between_same_day_zero():
    assert trading_days_between(date(2026, 11, 20), date(2026, 11, 20), H) == 0


def test_next_trading_days_len_and_order():
    days = next_trading_days(date(2026, 11, 20), 3, H)
    assert days == [date(2026, 11, 23), date(2026, 11, 25), date(2026, 11, 26)]


def test_events_between_inclusive():
    ev = [Event(date(2026, 12, 4), "RBI", "rbi"), Event(date(2026, 12, 9), "FOMC", "fomc")]
    assert events_between(ev, date(2026, 12, 4), date(2026, 12, 8)) == [ev[0]]
    assert events_between(ev, date(2026, 12, 5), date(2026, 12, 9)) == [ev[1]]


def test_calendar_days_to():
    assert calendar_days_to(date(2026, 11, 18), date(2026, 11, 25)) == 7
