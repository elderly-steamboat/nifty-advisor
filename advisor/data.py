# NSE end-of-day data: download + cache F&O bhavcopy and the daily index-close file
# (Nifty 50 OHLCV + India VIX), parse both bhavcopy formats, and load option chains.
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from advisor.calendar import is_trading_day

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
UDIFF_START = date(2024, 7, 8)  # NSE switched F&O bhavcopy to UDiFF format on this date
COLS = ["date", "instrument", "expiry", "strike", "option_type", "open", "high", "low",
        "close", "settle", "oi", "chg_oi", "volume", "lot_size"]
INDEX_COLS = ["date", "open", "high", "low", "close", "volume",
              "vix_open", "vix_high", "vix_low", "vix_close"]


class DataError(Exception):
    """Missing, unpublished, or unreadable market data."""


@dataclass
class DataStatus:
    date: date
    has_bhavcopy: bool
    has_index: bool
    has_vix: bool
    complete: bool
    missing: list[str] = field(default_factory=list)


@dataclass
class Chain:
    date: date
    expiry: date
    spot: float
    future: float
    lot_size: int
    rows: pd.DataFrame


# ---------- parsing ----------

def parse_bhavcopy(path: Path) -> pd.DataFrame:
    """Parse a UDiFF or legacy F&O bhavcopy zip into NIFTY rows with the COLS schema."""
    raw = pd.read_csv(path, compression="zip", dtype=str, keep_default_na=False)
    if "TckrSymb" in raw.columns:
        r = raw[(raw.TckrSymb == "NIFTY") & raw.FinInstrmTp.isin(["IDO", "IDF"])]
        df = pd.DataFrame({
            "date": pd.to_datetime(r.TradDt).dt.date,
            "instrument": r.FinInstrmTp.map({"IDO": "OPT", "IDF": "FUT"}),
            "expiry": pd.to_datetime(r.XpryDt).dt.date,
            "strike": pd.to_numeric(r.StrkPric, errors="coerce"),
            "option_type": r.OptnTp.where(r.OptnTp.isin(["CE", "PE"]), None),
            "open": r.OpnPric, "high": r.HghPric, "low": r.LwPric, "close": r.ClsPric,
            "settle": r.SttlmPric, "oi": r.OpnIntrst, "chg_oi": r.ChngInOpnIntrst,
            "volume": r.TtlTradgVol,
            "lot_size": pd.to_numeric(r.NewBrdLotQty, errors="coerce").astype("Int64"),
        })
    elif "INSTRUMENT" in raw.columns:
        r = raw[(raw.SYMBOL == "NIFTY") & raw.INSTRUMENT.isin(["OPTIDX", "FUTIDX"])]
        df = pd.DataFrame({
            "date": pd.to_datetime(r.TIMESTAMP, format="%d-%b-%Y").dt.date,
            "instrument": r.INSTRUMENT.map({"OPTIDX": "OPT", "FUTIDX": "FUT"}),
            "expiry": pd.to_datetime(r.EXPIRY_DT, format="%d-%b-%Y").dt.date,
            "strike": pd.to_numeric(r.STRIKE_PR, errors="coerce"),
            "option_type": r.OPTION_TYP.where(r.OPTION_TYP.isin(["CE", "PE"]), None),
            "open": r.OPEN, "high": r.HIGH, "low": r.LOW, "close": r.CLOSE,
            "settle": r.SETTLE_PR, "oi": r.OPEN_INT, "chg_oi": r.CHG_IN_OI,
            "volume": r.CONTRACTS,
            "lot_size": pd.array([pd.NA] * len(r), dtype="Int64"),
        })
    else:
        raise DataError(f"unknown bhavcopy format: {path}")
    for c in ["open", "high", "low", "close", "settle", "oi", "chg_oi", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    df.loc[df.instrument == "FUT", "strike"] = math.nan
    df = df.astype({"option_type": object})
    df.loc[df.instrument == "FUT", "option_type"] = None
    return df[COLS].sort_values(["instrument", "expiry", "strike", "option_type"],
                                na_position="first").reset_index(drop=True)


def _parse_index_file(path: Path) -> dict:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    by = {r["Index Name"].strip().lower(): r for _, r in raw.iterrows()}

    def num(r, col):
        try:
            return float(r[col])
        except (KeyError, TypeError, ValueError):
            return math.nan

    n, v = by.get("nifty 50"), by.get("india vix")
    if n is None:
        raise DataError(f"Nifty 50 row missing in {path}")
    d = pd.to_datetime(n["Index Date"], format="%d-%m-%Y").date()
    row = {"date": d.isoformat(), "open": num(n, "Open Index Value"),
           "high": num(n, "High Index Value"), "low": num(n, "Low Index Value"),
           "close": num(n, "Closing Index Value"), "volume": num(n, "Volume")}
    for k, col in [("vix_open", "Open Index Value"), ("vix_high", "High Index Value"),
                   ("vix_low", "Low Index Value"), ("vix_close", "Closing Index Value")]:
        row[k] = num(v, col) if v is not None else math.nan
    if math.isnan(row["volume"]):
        row["volume"] = 0.0
    return row


def ingest_index_file(path: Path, data_dir: Path) -> None:
    """Upsert one day's ind_close_all file into data_dir/index_daily.csv (sorted, one row per date)."""
    row = _parse_index_file(path)
    out = Path(data_dir) / "index_daily.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=INDEX_COLS)
    df = df[df.date != row["date"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    tmp = out.with_suffix(".tmp")
    df[INDEX_COLS].to_csv(tmp, index=False)
    tmp.replace(out)


# ---------- downloading ----------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    return s


def _bhav_urls(d: date) -> list[str]:
    udiff = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    legacy = (f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
              f"{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip")
    return [udiff, legacy] if d >= UDIFF_START else [legacy, udiff]


def _fetch(urls: list[str], dest: Path, d: date, what: str, session) -> Path:
    """Try each URL (3 attempts, backoff); write atomically. 404 on all → 'not published'."""
    session = session or _session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=30)
            except requests.RequestException as e:
                last_err = e
                time.sleep(0.5 * attempt)
                continue
            if resp.status_code == 404:
                break
            if resp.status_code != 200 or not resp.content:
                last_err = DataError(f"HTTP {resp.status_code} for {url}")
                time.sleep(0.5 * attempt)
                continue
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return dest
    if last_err is None:
        raise DataError(f"{what} not published for {d.isoformat()}")
    raise DataError(f"{what} download failed for {d.isoformat()}: {last_err}")


def download_bhavcopy(d: date, data_dir: Path, session=None) -> Path:
    """Return the cached bhavcopy for d, downloading it if missing."""
    dest = Path(data_dir) / "bhavcopy" / f"{d.isoformat()}.csv.zip"
    if dest.exists():
        return dest
    return _fetch(_bhav_urls(d), dest, d, "bhavcopy", session)


def download_index_close(d: date, data_dir: Path, session=None) -> Path:
    """Return the cached ind_close_all file for d (downloading if missing) and ingest it."""
    dest = Path(data_dir) / "indices" / f"{d.isoformat()}.csv"
    if not dest.exists():
        url = f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{d:%d%m%Y}.csv"
        _fetch([url], dest, d, "index close", session)
    ingest_index_file(dest, data_dir)
    return dest


def refresh(start: date, end: date, cfg, holidays: set[date], session=None) -> list[str]:
    """Download every missing trading day in [start, end]. Returns human-readable log lines."""
    session = session or _session()
    log, d = [], start
    while d <= end:
        if is_trading_day(d, holidays):
            st = data_status(d, cfg.data_dir)
            for what, have, fn in [("bhavcopy", st.has_bhavcopy, download_bhavcopy),
                                   ("index", st.has_index and st.has_vix, download_index_close)]:
                if have:
                    continue
                try:
                    fn(d, cfg.data_dir, session=session)
                    log.append(f"{d} {what}: downloaded")
                except DataError as e:
                    log.append(f"{d} {what}: {e}")
        d += timedelta(days=1)
    return log


# ---------- loading ----------

def _index_table(data_dir: Path) -> pd.DataFrame:
    p = Path(data_dir) / "index_daily.csv"
    if not p.exists():
        return pd.DataFrame(columns=INDEX_COLS).set_index("date")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df.date).dt.date
    return df.set_index("date").sort_index()


def load_index_history(end: date, n: int, data_dir: Path) -> pd.DataFrame:
    """Last n Nifty 50 daily bars with date <= end (index=date; open high low close volume)."""
    t = _index_table(data_dir)
    t = t[(t.index <= end) & t.close.notna()]
    return t[["open", "high", "low", "close", "volume"]].tail(n).astype(float)


def load_vix_history(end: date, n: int, data_dir: Path) -> pd.Series:
    """Last n India VIX closes with date <= end."""
    t = _index_table(data_dir)
    return t[(t.index <= end)].vix_close.dropna().tail(n).astype(float)


def data_status(d: date, data_dir: Path) -> DataStatus:
    """Which of bhavcopy / index / vix are present for d."""
    has_b = (Path(data_dir) / "bhavcopy" / f"{d.isoformat()}.csv.zip").exists()
    t = _index_table(data_dir)
    row = t.loc[d] if d in t.index else None
    has_i = row is not None and pd.notna(row.close)
    has_v = row is not None and pd.notna(row.vix_close)
    missing = [k for k, ok in [("bhavcopy", has_b), ("index", has_i), ("vix", has_v)] if not ok]
    return DataStatus(d, has_b, has_i, has_v, not missing, missing)


def _bhav(d: date, cfg) -> pd.DataFrame:
    p = Path(cfg.data_dir) / "bhavcopy" / f"{d.isoformat()}.csv.zip"
    if not p.exists():
        raise DataError(f"bhavcopy missing for {d.isoformat()} — refresh data")
    return parse_bhavcopy(p)


def list_expiries(d: date, cfg) -> list[date]:
    """Option expiries listed in d's bhavcopy that are after d."""
    df = _bhav(d, cfg)
    return sorted(e for e in set(df[df.instrument == "OPT"].expiry) if e > d)


def monthly_expiries(expiries: list[date]) -> list[date]:
    """Last listed expiry in each calendar month."""
    last: dict[tuple[int, int], date] = {}
    for e in sorted(expiries):
        last[(e.year, e.month)] = e
    return sorted(last.values())


def _parity_forward(opts: pd.DataFrame, d: date, expiry: date, r: float) -> float:
    piv = opts.pivot_table(index="strike", columns="option_type", values="settle")
    piv = piv[(piv.get("CE", 0) > 0) & (piv.get("PE", 0) > 0)]
    if piv.empty:
        raise DataError(f"cannot derive forward for {expiry}")
    k = (piv.CE - piv.PE).abs().idxmin()
    t = (expiry - d).days / 365.0
    return float(k + math.exp(r * t) * (piv.loc[k, "CE"] - piv.loc[k, "PE"]))


def load_chain(d: date, expiry: date, cfg) -> Chain:
    """Option chain for one expiry on date d, with spot, forward and lot size."""
    df = _bhav(d, cfg)
    opts = df[(df.instrument == "OPT") & (df.expiry == expiry)].reset_index(drop=True)
    if opts.empty:
        raise DataError(f"no options for expiry {expiry} on {d}")
    idx = load_index_history(d, 1, cfg.data_dir)
    if idx.empty or idx.index[-1] != d:
        raise DataError(f"index close missing for {d.isoformat()} — refresh data")
    spot = float(idx.close.iloc[-1])
    fut = df[(df.instrument == "FUT") & (df.expiry == expiry)]
    future = float(fut.settle.iloc[0]) if not fut.empty and fut.settle.iloc[0] > 0 \
        else _parity_forward(opts, d, expiry, cfg.risk_free_rate)
    lots = opts.lot_size.dropna()
    lot_size = int(lots.iloc[0]) if len(lots) else int(cfg.lot_size_fallback)
    return Chain(d, expiry, spot, future, lot_size, opts)
