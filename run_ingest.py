"""CLI entry point: download CVM + CDI + Ibovespa data for the fund registry.

Usage:
    python run_ingest.py                  # rolling 10-year window to today
    python run_ingest.py --start 2020-01-01 --end 2024-12-31
"""

import argparse
import logging
import re
import sys
from datetime import date, timedelta

import pandas as pd

from src.utils import CONFIG_DIR, DATA_RAW, DATA_PROCESSED, normalize_cnpj
from src.ingest_cvm import fetch_fund_quotas
from src.ingest_cdi import update_cdi
from src.ingest_ibov import update_ibov
from src.processing import (
    compute_daily_returns,
    compute_cdi_daily_returns,
    compute_ibov_daily_returns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Registry loader ─────────────────────────────────────────────────────────

def load_cnpjs_from_registry() -> list[str]:
    """Read fund_registry.csv and return list of normalized CNPJs for active funds."""
    path = CONFIG_DIR / "fund_registry.csv"
    if not path.exists():
        logger.error("Registry not found: %s", path)
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["active"] = df["active"].str.strip().str.lower() == "true"
    active = df[df["active"]].copy()

    cnpjs = []
    for _, row in active.iterrows():
        raw = row.get("cnpj", "")
        if pd.notna(raw) and str(raw).strip():
            cnpjs.append(normalize_cnpj(str(raw).strip()))

    if not cnpjs:
        logger.error("No active funds with CNPJs found in registry.")
        sys.exit(1)

    logger.info("Registry: %d active funds with CNPJs", len(cnpjs))
    return cnpjs


# ── Stale file cleanup ─────────────────────────────────────────────────────

def cleanup_stale_cvm_files(window_start_year: int):
    """Remove cached CVM yearly parquet files for years before the window."""
    pattern = re.compile(r"^cvm_(\d{4})\.parquet$")
    removed = 0
    for f in DATA_RAW.iterdir():
        m = pattern.match(f.name)
        if m and int(m.group(1)) < window_start_year:
            f.unlink()
            logger.info("Removed stale CVM cache: %s", f.name)
            removed += 1
    if removed:
        logger.info("Cleaned up %d stale CVM yearly file(s)", removed)


# ── Validation ──────────────────────────────────────────────────────────────

def validate_ingestion(cnpjs: list[str]) -> bool:
    """Check that all expected output files exist and contain data."""
    ok = True
    checks = {
        "quotas.parquet": DATA_PROCESSED / "quotas.parquet",
        "fund_returns.parquet": DATA_PROCESSED / "fund_returns.parquet",
        "cdi_returns.parquet": DATA_PROCESSED / "cdi_returns.parquet",
        "ibov_returns.parquet": DATA_PROCESSED / "ibov_returns.parquet",
        "cdi.parquet (raw)": DATA_RAW / "cdi.parquet",
        "ibov.parquet (raw)": DATA_RAW / "ibov.parquet",
    }

    for label, path in checks.items():
        if not path.exists():
            logger.error("VALIDATION FAIL: %s not found at %s", label, path)
            ok = False
        else:
            df = pd.read_parquet(path)
            if df.empty:
                logger.warning("VALIDATION WARN: %s is empty", label)
            else:
                logger.info("VALIDATION OK: %s — %d rows", label, len(df))

    # Check fund coverage
    fund_ret_path = DATA_PROCESSED / "fund_returns.parquet"
    if fund_ret_path.exists():
        fr = pd.read_parquet(fund_ret_path)
        found_cnpjs = set(fr["CNPJ_FUNDO"].unique())
        missing = set(cnpjs) - found_cnpjs
        if missing:
            logger.warning(
                "VALIDATION WARN: %d/%d CNPJs have no return data: %s",
                len(missing), len(cnpjs), missing,
            )
        else:
            logger.info("VALIDATION OK: all %d CNPJs have return data", len(cnpjs))

        # Date coverage
        fr["date"] = pd.to_datetime(fr["date"])
        logger.info(
            "Fund data range: %s to %s",
            fr["date"].min().date(), fr["date"].max().date(),
        )

    # Ibovespa range check
    ibov_path = DATA_RAW / "ibov.parquet"
    if ibov_path.exists():
        ib = pd.read_parquet(ibov_path)
        ib["date"] = pd.to_datetime(ib["date"])
        logger.info(
            "Ibovespa data range: %s to %s (%d rows)",
            ib["date"].min().date(), ib["date"].max().date(), len(ib),
        )

    return ok


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest fund + CDI + Ibovespa data")
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD (default: rolling 10Y from today)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        # Rolling 10-year window
        start = date(end.year - 10, end.month, end.day)

    cnpjs = load_cnpjs_from_registry()

    logger.info("=" * 60)
    logger.info("Ingesting data: %s to %s (%.1f years)", start, end,
                (end - start).days / 365.25)
    logger.info("Fund universe: %d CNPJs from registry", len(cnpjs))
    logger.info("=" * 60)

    # 0. Clean up stale CVM yearly files outside the window
    logger.info("--- Step 0/5: Cleaning stale cache ---")
    cleanup_stale_cvm_files(start.year)

    # 1. Download fund quotas from CVM
    logger.info("--- Step 1/5: CVM fund quotas ---")
    quotas = fetch_fund_quotas(cnpjs, start, end)
    quotas_path = DATA_PROCESSED / "quotas.parquet"
    quotas.to_parquet(quotas_path, index=False)
    logger.info("Quotas saved: %s (%d rows)", quotas_path, len(quotas))

    # 2. Download CDI from BCB (incremental merge + trim to window)
    logger.info("--- Step 2/5: CDI (BCB series 12) ---")
    cdi = update_cdi(start, end)
    logger.info("CDI loaded: %d rows (%s to %s)", len(cdi),
                cdi["date"].min().date() if len(cdi) > 0 else "?",
                cdi["date"].max().date() if len(cdi) > 0 else "?")

    # 3. Download Ibovespa (yfinance ^BVSP, incremental merge + trim)
    logger.info("--- Step 3/5: Ibovespa (Yahoo Finance ^BVSP) ---")
    ibov = update_ibov(start, end)
    logger.info("Ibovespa loaded: %d rows (%s to %s)", len(ibov),
                ibov["date"].min().date() if len(ibov) > 0 else "?",
                ibov["date"].max().date() if len(ibov) > 0 else "?")

    # 4. Compute daily returns
    logger.info("--- Step 4/5: Computing daily returns ---")
    fund_returns = compute_daily_returns(quotas)
    fund_returns_path = DATA_PROCESSED / "fund_returns.parquet"
    fund_returns.to_parquet(fund_returns_path, index=False)
    logger.info("Fund returns saved: %d rows", len(fund_returns))

    cdi_returns = compute_cdi_daily_returns(cdi)
    cdi_returns_path = DATA_PROCESSED / "cdi_returns.parquet"
    cdi_returns.to_parquet(cdi_returns_path, index=False)
    logger.info("CDI returns saved: %d rows", len(cdi_returns))

    ibov_returns = compute_ibov_daily_returns(ibov)
    ibov_returns_path = DATA_PROCESSED / "ibov_returns.parquet"
    ibov_returns.to_parquet(ibov_returns_path, index=False)
    logger.info("Ibovespa returns saved: %d rows", len(ibov_returns))

    # 5. Validation
    logger.info("--- Step 5/5: Validation ---")
    valid = validate_ingestion(cnpjs)

    logger.info("=" * 60)
    if valid:
        logger.info("Ingestion complete — all checks passed.")
    else:
        logger.warning("Ingestion complete — some checks failed (see warnings above).")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
