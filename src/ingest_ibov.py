"""Fetch daily Ibovespa closing prices from Yahoo Finance chart API.

BCB SGS series 7 was discontinued after 2019-09-30.
This module uses Yahoo Finance's chart API (^BVSP) via requests.
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout

from src.utils import DATA_RAW

logger = logging.getLogger(__name__)

IBOV_RAW_PATH = DATA_RAW / "ibov.parquet"

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP"

MAX_RETRIES = 4
BACKOFF_BASE = 5


def _date_to_unix(d: date) -> int:
    """Convert date to Unix timestamp (UTC midnight)."""
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def fetch_ibov(start: date, end: date) -> pd.DataFrame:
    """Download Ibovespa daily closing values from Yahoo Finance chart API.

    Returns DataFrame with columns: date, ibov_close.
    """
    # Yahoo end is inclusive when using period2 = next day midnight
    params = {
        "period1": _date_to_unix(start),
        "period2": _date_to_unix(end + timedelta(days=1)),
        "interval": "1d",
        "events": "history",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    logger.info("Fetching Ibovespa from Yahoo Finance: %s to %s", start, end)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(YAHOO_CHART_URL, params=params, headers=headers, timeout=60)
            if resp.status_code == 404:
                logger.warning("Yahoo Finance returned 404 for Ibovespa")
                return pd.DataFrame(columns=["date", "ibov_close"])
            resp.raise_for_status()
            break
        except (ConnectionError, Timeout, OSError) as exc:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning("Ibov attempt %d/%d failed: %s. Retrying in %ds...",
                           attempt, MAX_RETRIES, exc, wait)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)

    data = resp.json()
    chart = data.get("chart", {})
    result = chart.get("result")
    if not result:
        error = chart.get("error", {})
        logger.warning("Yahoo Finance returned no data: %s", error)
        return pd.DataFrame(columns=["date", "ibov_close"])

    timestamps = result[0].get("timestamp", [])
    indicators = result[0].get("indicators", {})
    quotes = indicators.get("quote", [{}])[0]
    closes = quotes.get("close", [])

    if not timestamps or not closes:
        logger.warning("Yahoo Finance returned empty timestamps/closes")
        return pd.DataFrame(columns=["date", "ibov_close"])

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/Sao_Paulo").tz_localize(None),
        "ibov_close": closes,
    })
    # Normalize dates to midnight (remove time component)
    df["date"] = df["date"].dt.normalize()
    df["ibov_close"] = pd.to_numeric(df["ibov_close"], errors="coerce")
    df.dropna(subset=["ibov_close"], inplace=True)
    df.drop_duplicates(subset="date", keep="last", inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Ibovespa fetched: %d rows (%s to %s)", len(df),
                df["date"].min().date() if len(df) > 0 else "?",
                df["date"].max().date() if len(df) > 0 else "?")
    return df


def update_ibov(start: date, end: date) -> pd.DataFrame:
    """Fetch Ibovespa, merge with cached data, trim to window, and save."""
    new = fetch_ibov(start, end)

    if IBOV_RAW_PATH.exists():
        cached = pd.read_parquet(IBOV_RAW_PATH)
        cached["date"] = pd.to_datetime(cached["date"])
        combined = pd.concat([cached, new]).drop_duplicates(subset="date")
    else:
        combined = new

    # Trim to requested window
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined[
        (combined["date"] >= pd.Timestamp(start)) &
        (combined["date"] <= pd.Timestamp(end))
    ].copy()
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_parquet(IBOV_RAW_PATH, index=False)
    logger.info("Ibovespa saved: %d rows → %s", len(combined), IBOV_RAW_PATH)
    return combined
