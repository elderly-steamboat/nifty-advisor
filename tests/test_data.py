from datetime import date

import pandas as pd
import pytest
import requests

from advisor.data import (DataError, data_status, download_bhavcopy, ingest_index_file,
                          list_expiries, load_chain, load_index_history, load_vix_history,
                          monthly_expiries, parse_bhavcopy)
from tests.conftest import FIX

COLS = ["date", "instrument", "expiry", "strike", "option_type", "open", "high", "low",
        "close", "settle", "oi", "chg_oi", "volume", "lot_size"]


def test_parse_udiff_fixture():
    df = parse_bhavcopy(FIX / "udiff_2026-11-17.csv.zip")
    assert list(df.columns) == COLS
    assert set(df.instrument) == {"OPT", "FUT"}
    assert df.lot_size.notna().all() and (df.lot_size == 65).all()
    assert df.date.unique().tolist() == [date(2026, 11, 17)]
    assert set(df[df.instrument == "OPT"].option_type) == {"CE", "PE"}


def test_parse_legacy_fixture():
    df = parse_bhavcopy(FIX / "legacy_2023-06-15.csv.zip")
    assert list(df.columns) == COLS
    assert df.lot_size.isna().all()
    assert date(2023, 6, 29) in set(df.expiry)
    fut = df[df.instrument == "FUT"]
    assert fut.strike.isna().all() and fut.option_type.isna().all()


def test_unknown_format_raises(tmp_path):
    import zipfile
    p = tmp_path / "x.csv.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x.csv", "a,b\n1,2\n")
    with pytest.raises(DataError, match="unknown bhavcopy format"):
        parse_bhavcopy(p)


class _FailSession:
    def get(self, *a, **k):
        raise requests.ConnectionError("boom")


class _NotFound:
    status_code = 404
    content = b""


class _404Session:
    def get(self, *a, **k):
        return _NotFound()


def test_download_failure_keeps_cache(seeded_cfg):
    cached = seeded_cfg.data_dir / "bhavcopy" / "2026-11-17.csv.zip"
    before = cached.read_bytes()
    with pytest.raises(DataError, match="2026-11-18"):
        download_bhavcopy(date(2026, 11, 18), seeded_cfg.data_dir, session=_FailSession())
    assert cached.read_bytes() == before
    assert not (seeded_cfg.data_dir / "bhavcopy" / "2026-11-18.csv.zip").exists()


def test_download_404_is_not_published(tmp_cfg):
    with pytest.raises(DataError, match="not published"):
        download_bhavcopy(date(2026, 11, 18), tmp_cfg.data_dir, session=_404Session())


def test_download_uses_cache(seeded_cfg):
    p = download_bhavcopy(date(2026, 11, 17), seeded_cfg.data_dir, session=_FailSession())
    assert p.exists()


def test_index_and_vix_history(seeded_cfg):
    idx = load_index_history(date(2026, 11, 17), 10, seeded_cfg.data_dir)
    assert list(idx.columns) == ["open", "high", "low", "close", "volume"]
    assert idx.index[-1] == date(2026, 11, 17) and idx.close.iloc[-1] == 25387.4
    assert len(idx) == 2  # only two days cached
    vix = load_vix_history(date(2026, 11, 17), 10, seeded_cfg.data_dir)
    assert vix.iloc[-1] == 15.52


def test_index_history_no_lookahead(seeded_cfg):
    idx = load_index_history(date(2023, 6, 15), 10, seeded_cfg.data_dir)
    assert idx.index.max() == date(2023, 6, 15)


def test_ingest_is_idempotent(seeded_cfg):
    ingest_index_file(FIX / "ind_close_all_2026-11-17.csv", seeded_cfg.data_dir)
    assert len(load_index_history(date(2026, 11, 17), 10, seeded_cfg.data_dir)) == 2


def test_load_chain_uses_future_and_lot_size(seeded_cfg):
    ch = load_chain(date(2026, 11, 17), date(2026, 12, 29), seeded_cfg)
    raw = parse_bhavcopy(FIX / "udiff_2026-11-17.csv.zip")
    fut = raw[(raw.instrument == "FUT") & (raw.expiry == date(2026, 12, 29))].settle.iloc[0]
    assert ch.future == fut
    assert ch.lot_size == 65 and ch.spot == 25387.4
    assert set(ch.rows.option_type) == {"CE", "PE"}
    assert (ch.rows.expiry == date(2026, 12, 29)).all()


def test_load_chain_parity_forward_without_future(seeded_cfg):
    ch = load_chain(date(2026, 11, 17), date(2026, 12, 8), seeded_cfg)  # weekly, no FUT
    assert abs(ch.future - ch.spot) / ch.spot < 0.02


def test_load_chain_lot_size_fallback(seeded_cfg):
    ch = load_chain(date(2023, 6, 15), date(2023, 6, 29), seeded_cfg)
    assert ch.lot_size == seeded_cfg.lot_size_fallback


def test_load_chain_missing_expiry_raises(seeded_cfg):
    with pytest.raises(DataError, match="no options"):
        load_chain(date(2026, 11, 17), date(2026, 12, 1), seeded_cfg)


def test_data_status_reports_missing_vix(seeded_cfg):
    idx = seeded_cfg.data_dir / "index_daily.csv"
    df = pd.read_csv(idx)
    df.loc[df.date == "2026-11-17", ["vix_close"]] = None
    df.to_csv(idx, index=False)
    st = data_status(date(2026, 11, 17), seeded_cfg.data_dir)
    assert not st.complete and st.missing == ["vix"]
    assert data_status(date(2023, 6, 15), seeded_cfg.data_dir).complete


def test_data_status_nothing(tmp_cfg):
    st = data_status(date(2026, 11, 17), tmp_cfg.data_dir)
    assert st.missing == ["bhavcopy", "index", "vix"]


def test_list_and_monthly_expiries(seeded_cfg):
    ex = list_expiries(date(2026, 11, 17), seeded_cfg)
    assert ex == [date(2026, 11, 23), date(2026, 12, 8), date(2026, 12, 29)]
    assert monthly_expiries(ex) == [date(2026, 11, 23), date(2026, 12, 29)]
