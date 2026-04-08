"""Fetch daily CDI rates from the BCB SGS API (series 12)."""

import logging
from datetime import date

import pandas as pd
import requests

from src.utils import DATA_RAW

logger = logging.getLogger(__name__)

BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"

CDI_RAW_PATH = DATA_RAW / "cdi.parquet"


def fetch_cdi(start: date, end: date) -> pd.DataFrame:
    """Download CDI daily rates from BCB.

    Returns DataFrame with columns: date, cdi_rate (daily % rate, e.g. 0.04).
    Source: BCB SGS series 12 — CDI daily annualized rate.
    """
    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }
    headers = {"User-Agent": "Mozilla/5.0 (fund-peer-monitor)"}
    logger.info("Fetching CDI from BCB: %s to %s", start, end)
    resp = requests.get(BCB_SGS_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    if not data:
        logger.warning("BCB returned empty CDI data for %s–%s", start, end)
        return pd.DataFrame(columns=["date", "cdi_rate"])

    df = pd.DataFrame(data)
    df.rename(columns={"data": "date", "valor": "cdi_rate"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["cdi_rate"] = pd.to_numeric(df["cdi_rate"], errors="coerce")
    df.dropna(subset=["cdi_rate"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def update_cdi(start: date, end: date) -> pd.DataFrame:
    """Fetch CDI, merge with any cached data, trim to window, and save."""
    new = fetch_cdi(start, end)

    if CDI_RAW_PATH.exists():
        cached = pd.read_parquet(CDI_RAW_PATH)
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

    combined.to_parquet(CDI_RAW_PATH, index=False)
    logger.info("CDI saved: %d rows → %s", len(combined), CDI_RAW_PATH)
    return combined
