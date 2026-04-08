"""Fetch daily fund quota data from CVM open data portal.

Source: https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/
Files:  inf_diario_fi_YYYYMM.zip (all months, zipped)
        HIST/inf_diario_fi_YYYY.zip (yearly archives before 2021)

CVM Resolution 175 (2023-2025) changed the CSV schema:
  - Old: CNPJ_FUNDO, DT_COMPTC, VL_QUOTA, VL_PATRIM_LIQ, NR_COTST
  - New: CNPJ_FUNDO_CLASSE, DT_COMPTC, VL_QUOTA, VL_PATRIM_LIQ, NR_COTST, ...

This module normalizes both schemas to a single internal column: CNPJ_FUNDO.
"""

import io
import logging
import time
import zipfile
from datetime import date

import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

from src.utils import DATA_RAW, normalize_cnpj, month_range

logger = logging.getLogger(__name__)

CVM_BASE = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
_SESSION: requests.Session | None = None

MAX_RETRIES = 4
BACKOFF_BASE = 5  # seconds: 5, 10, 20, 40


def _get_session() -> requests.Session:
    """Reusable session with User-Agent and connection pooling."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": "Mozilla/5.0 (fund-peer-monitor)",
        })
    return _SESSION


def _robust_get(url: str, timeout: int = 180) -> requests.Response | None:
    """GET with retry + exponential backoff for transient failures.

    Returns Response on success, or None for 403/404 (expected missing files).
    Raises on persistent failure after all retries.
    """
    session = _get_session()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in (403, 404):
                return None
            resp.raise_for_status()
            return resp
        except (ChunkedEncodingError, ConnectionError, Timeout, OSError) as exc:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                attempt, MAX_RETRIES, url, exc, wait,
            )
            if attempt == MAX_RETRIES:
                logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
                raise
            time.sleep(wait)

# Internal column names after normalization
CNPJ_COL = "CNPJ_FUNDO"
KEEP_COLS = ["CNPJ_FUNDO", "DT_COMPTC", "VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"]


def _normalize_cvm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CVM columns to our internal standard.

    Handles both pre-CVM175 (CNPJ_FUNDO) and post-CVM175 (CNPJ_FUNDO_CLASSE).
    """
    if "CNPJ_FUNDO_CLASSE" in df.columns and "CNPJ_FUNDO" not in df.columns:
        df = df.rename(columns={"CNPJ_FUNDO_CLASSE": "CNPJ_FUNDO"})

    # Some new files have extra columns (TP_FUNDO_CLASSE, ID_SUBCLASSE, etc.)
    # Keep only what we need, ignoring missing optional columns
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # Ensure NR_COTST exists (optional in some files)
    if "NR_COTST" not in df.columns:
        df["NR_COTST"] = pd.NA

    return df


def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    """Read a CVM CSV from raw bytes, handling encoding and schema variations."""
    # CVM files are Latin-1 encoded
    text = data.decode("latin-1")
    # Detect CNPJ column to set dtype
    cnpj_col = "CNPJ_FUNDO_CLASSE" if "CNPJ_FUNDO_CLASSE" in text[:500] else "CNPJ_FUNDO"
    df = pd.read_csv(
        io.StringIO(text),
        sep=";",
        dtype={cnpj_col: str},
        parse_dates=["DT_COMPTC"],
        dayfirst=False,
    )
    df = _normalize_cvm_columns(df)
    df["VL_QUOTA"] = pd.to_numeric(df["VL_QUOTA"], errors="coerce")
    df["VL_PATRIM_LIQ"] = pd.to_numeric(df["VL_PATRIM_LIQ"], errors="coerce")
    return df


def _download_monthly_zip(year_month: str) -> pd.DataFrame | None:
    """Download inf_diario_fi_YYYYMM.zip from CVM.

    Since ~2021, CVM serves monthly files as .zip (not .csv).
    """
    url = f"{CVM_BASE}/inf_diario_fi_{year_month}.zip"
    logger.info("Downloading CVM monthly zip: %s", url)
    resp = _robust_get(url, timeout=180)

    if resp is None:
        logger.warning("CVM file not available: %s", url)
        return None

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile:
        logger.warning("CVM returned invalid zip for %s (possibly HTML error page)", url)
        return None

    frames = []
    for name in zf.namelist():
        if name.endswith(".csv"):
            with zf.open(name) as f:
                df = _read_csv_from_bytes(f.read())
                frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _download_yearly_zip(year: int) -> pd.DataFrame | None:
    """Download historical inf_diario_fi_YYYY.zip from CVM.

    For years before 2021, these live directly under DADOS/.
    For older years, they may be under HIST/.
    """
    # Try main directory first, then HIST/
    for base in [CVM_BASE, f"{CVM_BASE}/HIST"]:
        url = f"{base}/inf_diario_fi_{year}.zip"
        logger.info("Downloading CVM yearly zip: %s", url)
        resp = _robust_get(url, timeout=300)

        if resp is None:
            continue

        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile:
            continue

        frames = []
        for name in zf.namelist():
            if name.endswith(".csv"):
                with zf.open(name) as f:
                    df = _read_csv_from_bytes(f.read())
                    frames.append(df)

        if frames:
            return pd.concat(frames, ignore_index=True)

    logger.warning("CVM yearly zip not found for %d", year)
    return None


def fetch_fund_quotas(
    cnpjs: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Download CVM quota data for the given CNPJs and date range.

    Strategy:
    - For years before current: try yearly zip (cached as parquet)
    - For current year months: download monthly zips
    - Filter to requested CNPJs only
    - Normalize column names across old/new CVM schemas
    """
    target_cnpjs = {normalize_cnpj(c) for c in cnpjs}
    current_year = date.today().year
    all_frames = []

    # Determine which years/months we need
    years_needed = set()
    months_needed = []
    for ym in month_range(start, end):
        year = int(ym[:4])
        if year < current_year:
            years_needed.add(year)
        else:
            months_needed.append(ym)

    # Download historical yearly zips (fall back to monthly zips for 2021+)
    for year in sorted(years_needed):
        cache_path = DATA_RAW / f"cvm_{year}.parquet"
        if cache_path.exists():
            logger.info("Using cached CVM year: %s", cache_path)
            df = pd.read_parquet(cache_path)
        else:
            df = _download_yearly_zip(year)
            if df is None:
                # CVM stopped yearly zips after 2020 — fall back to monthly
                logger.info("Yearly zip unavailable for %d, falling back to monthly zips", year)
                year_start = max(start, date(year, 1, 1))
                year_end = min(end, date(year, 12, 31))
                monthly_frames = []
                for ym in month_range(year_start, year_end):
                    mdf = _download_monthly_zip(ym)
                    if mdf is not None:
                        monthly_frames.append(mdf)
                if not monthly_frames:
                    continue
                df = pd.concat(monthly_frames, ignore_index=True)
            df["CNPJ_FUNDO"] = df["CNPJ_FUNDO"].apply(normalize_cnpj)
            df = df[KEEP_COLS].copy()
            df.to_parquet(cache_path, index=False)
            logger.info("Cached CVM year %d: %d rows", year, len(df))

        filtered = df[df["CNPJ_FUNDO"].isin(target_cnpjs)]
        all_frames.append(filtered)

    # Download current-year monthly zips
    for ym in months_needed:
        cache_path = DATA_RAW / f"cvm_{ym}.parquet"
        is_current_month = ym == date.today().strftime("%Y%m")

        if cache_path.exists() and not is_current_month:
            logger.info("Using cached CVM month: %s", cache_path)
            df = pd.read_parquet(cache_path)
        else:
            df = _download_monthly_zip(ym)
            if df is None:
                continue
            df["CNPJ_FUNDO"] = df["CNPJ_FUNDO"].apply(normalize_cnpj)
            df = df[KEEP_COLS].copy()
            if not is_current_month:
                df.to_parquet(cache_path, index=False)
                logger.info("Cached CVM month %s: %d rows", ym, len(df))

        filtered = df[df["CNPJ_FUNDO"].isin(target_cnpjs)]
        all_frames.append(filtered)

    if not all_frames:
        logger.warning("No CVM data found for the requested range.")
        return pd.DataFrame(columns=KEEP_COLS)

    result = pd.concat(all_frames, ignore_index=True)
    result.drop_duplicates(subset=["CNPJ_FUNDO", "DT_COMPTC"], inplace=True)
    result.sort_values(["CNPJ_FUNDO", "DT_COMPTC"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    logger.info(
        "CVM data loaded: %d rows for %d funds", len(result), result["CNPJ_FUNDO"].nunique()
    )
    return result
