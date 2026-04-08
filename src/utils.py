"""Shared utilities: paths, CNPJ normalization, date helpers."""

import os
from pathlib import Path
from datetime import date, timedelta

import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"

for d in (DATA_RAW, DATA_PROCESSED):
    d.mkdir(parents=True, exist_ok=True)


# ── CNPJ ─────────────────────────────────────────────────────────────────────

def normalize_cnpj(cnpj: str) -> str:
    """Remove formatting from CNPJ, keeping only digits."""
    return "".join(c for c in cnpj if c.isdigit())


def format_cnpj(cnpj: str) -> str:
    """Format 14-digit CNPJ string as XX.XXX.XXX/XXXX-XX."""
    d = normalize_cnpj(cnpj).zfill(14)
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:14]}"


# ── Date helpers ─────────────────────────────────────────────────────────────

def month_range(start: date, end: date) -> list[str]:
    """Return list of 'YYYYMM' strings covering the months from start to end."""
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current.strftime("%Y%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def business_days_ago(n: int, ref: date | None = None) -> date:
    """Return the date n business days before ref (default: today)."""
    ref = ref or date.today()
    bdays = pd.bdate_range(end=ref, periods=n + 1)
    return bdays[0].date()
