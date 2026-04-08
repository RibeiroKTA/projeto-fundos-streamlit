"""Return calculations and peer statistics.

All return calculations are based on quota variation (VL_QUOTA).
CDI returns are converted from annual % rate to daily factor.
Ibovespa returns are computed from daily closing index values.
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.utils import DATA_RAW, DATA_PROCESSED

logger = logging.getLogger(__name__)


# ── Daily returns ────────────────────────────────────────────────────────────

def compute_daily_returns(quotas: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from quota values.

    Input:  DataFrame with CNPJ_FUNDO, DT_COMPTC, VL_QUOTA
    Output: DataFrame with CNPJ_FUNDO, date, daily_return
    """
    df = quotas[["CNPJ_FUNDO", "DT_COMPTC", "VL_QUOTA"]].copy()
    df.rename(columns={"DT_COMPTC": "date"}, inplace=True)
    df.sort_values(["CNPJ_FUNDO", "date"], inplace=True)

    df["daily_return"] = df.groupby("CNPJ_FUNDO")["VL_QUOTA"].pct_change()
    df.dropna(subset=["daily_return"], inplace=True)
    return df[["CNPJ_FUNDO", "date", "daily_return"]].reset_index(drop=True)


def compute_cdi_daily_returns(cdi: pd.DataFrame) -> pd.DataFrame:
    """Convert CDI annual rate to daily return factor.

    BCB series 12 gives the daily CDI rate already as a daily %.
    E.g., 0.04 means 0.04% daily → daily_return = 0.0004
    """
    df = cdi[["date", "cdi_rate"]].copy()
    df["daily_return"] = df["cdi_rate"] / 100.0
    return df[["date", "daily_return"]].reset_index(drop=True)


def compute_ibov_daily_returns(ibov: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from Ibovespa closing index values.

    Input:  DataFrame with date, ibov_close
    Output: DataFrame with date, daily_return
    """
    df = ibov[["date", "ibov_close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df["daily_return"] = df["ibov_close"].pct_change()
    df.dropna(subset=["daily_return"], inplace=True)
    return df[["date", "daily_return"]].reset_index(drop=True)


# ── Cumulative returns ───────────────────────────────────────────────────────

def cumulative_returns(daily_returns: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Compute cumulative returns from start to end for each fund.

    Input:  DataFrame with CNPJ_FUNDO, date, daily_return
    Output: DataFrame with CNPJ_FUNDO, date, cumulative_return (1-based, i.e., 1.05 = +5%)
    """
    mask = (daily_returns["date"] >= pd.Timestamp(start)) & (daily_returns["date"] <= pd.Timestamp(end))
    df = daily_returns.loc[mask].copy()

    df.sort_values(["CNPJ_FUNDO", "date"], inplace=True)
    df["cumulative_return"] = df.groupby("CNPJ_FUNDO")["daily_return"].transform(
        lambda x: (1 + x).cumprod()
    )
    return df[["CNPJ_FUNDO", "date", "cumulative_return"]].reset_index(drop=True)


def cumulative_cdi(cdi_daily: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Compute cumulative CDI return from start to end."""
    mask = (cdi_daily["date"] >= pd.Timestamp(start)) & (cdi_daily["date"] <= pd.Timestamp(end))
    df = cdi_daily.loc[mask].copy()
    df.sort_values("date", inplace=True)
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod()
    return df[["date", "cumulative_return"]].reset_index(drop=True)


# ── Period returns ───────────────────────────────────────────────────────────

def period_return(daily_returns: pd.DataFrame, start: date, end: date) -> pd.Series:
    """Compute total return for each fund over a period.

    Returns Series indexed by CNPJ_FUNDO with the total return (e.g., 0.05 = +5%).
    """
    mask = (daily_returns["date"] >= pd.Timestamp(start)) & (daily_returns["date"] <= pd.Timestamp(end))
    df = daily_returns.loc[mask].copy()

    def _total_return(group):
        return (1 + group["daily_return"]).prod() - 1

    return df.groupby("CNPJ_FUNDO").apply(_total_return, include_groups=False)


def cdi_period_return(cdi_daily: pd.DataFrame, start: date, end: date) -> float:
    """Compute total CDI return for a period."""
    mask = (cdi_daily["date"] >= pd.Timestamp(start)) & (cdi_daily["date"] <= pd.Timestamp(end))
    df = cdi_daily.loc[mask]
    if df.empty:
        return 0.0
    return float((1 + df["daily_return"]).prod() - 1)


def cumulative_ibov(ibov_daily: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Compute cumulative Ibovespa return from start to end."""
    mask = (ibov_daily["date"] >= pd.Timestamp(start)) & (ibov_daily["date"] <= pd.Timestamp(end))
    df = ibov_daily.loc[mask].copy()
    df.sort_values("date", inplace=True)
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod()
    return df[["date", "cumulative_return"]].reset_index(drop=True)


def ibov_period_return(ibov_daily: pd.DataFrame, start: date, end: date) -> float:
    """Compute total Ibovespa return for a period."""
    mask = (ibov_daily["date"] >= pd.Timestamp(start)) & (ibov_daily["date"] <= pd.Timestamp(end))
    df = ibov_daily.loc[mask]
    if df.empty:
        return 0.0
    return float((1 + df["daily_return"]).prod() - 1)


# ── Date range helpers for standard periods ──────────────────────────────────

def get_period_dates(period: str, ref_date: date | None = None) -> tuple[date, date]:
    """Return (start, end) dates for standard return periods.

    Periods: 'day', 'mtd', 'ytd', '1y', '3y', '5y', '10y'
    """
    end = ref_date or date.today()

    match period:
        case "day":
            return end, end
        case "mtd":
            return end.replace(day=1), end
        case "ytd":
            return date(end.year, 1, 1), end
        case "1y":
            return date(end.year - 1, end.month, end.day), end
        case "3y":
            return date(end.year - 3, end.month, end.day), end
        case "5y":
            return date(end.year - 5, end.month, end.day), end
        case "10y":
            return date(end.year - 10, end.month, end.day), end
        case _:
            raise ValueError(f"Unknown period: {period}")


# ── Peer statistics ──────────────────────────────────────────────────────────

def peer_stats(returns_series: pd.Series) -> dict:
    """Compute mean and median from a Series of fund returns."""
    return {
        "mean": float(returns_series.mean()) if len(returns_series) > 0 else 0.0,
        "median": float(returns_series.median()) if len(returns_series) > 0 else 0.0,
    }


# ── Ranking table ────────────────────────────────────────────────────────────

def build_ranking(
    fund_returns: pd.Series,
    cdi_return: float,
    names: dict[str, str],
    ibov_return: float | None = None,
    benchmarks: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a ranking DataFrame for a given period.

    Args:
        fund_returns: Series indexed by CNPJ with total return
        cdi_return: CDI total return for the same period
        names: dict mapping CNPJ → display name
        ibov_return: Ibovespa total return for the same period (optional)
        benchmarks: list of selected benchmarks, e.g. ["CDI", "Ibovespa"]

    Returns:
        (ranking_df, stats_df) with columns: name, return_pct, vs_cdi_pct[, vs_ibov_pct]
    """
    if benchmarks is None:
        benchmarks = ["CDI"]

    df = fund_returns.reset_index()
    df.columns = ["cnpj", "return"]
    df["name"] = df["cnpj"].map(names).fillna(df["cnpj"])
    df["return_pct"] = df["return"] * 100

    cols = ["name", "return_pct"]

    if "CDI" in benchmarks:
        df["vs_cdi_pct"] = (df["return"] - cdi_return) * 100
        cols.append("vs_cdi_pct")

    if "Ibovespa" in benchmarks and ibov_return is not None:
        df["vs_ibov_pct"] = (df["return"] - ibov_return) * 100
        cols.append("vs_ibov_pct")

    df.sort_values("return_pct", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.index.name = "rank"

    stats = peer_stats(fund_returns)
    stats_list = [
        {"name": "Peer Mean", "return_pct": stats["mean"] * 100},
        {"name": "Peer Median", "return_pct": stats["median"] * 100},
    ]

    if "CDI" in benchmarks:
        stats_list[0]["vs_cdi_pct"] = (stats["mean"] - cdi_return) * 100
        stats_list[1]["vs_cdi_pct"] = (stats["median"] - cdi_return) * 100
        stats_list.append({
            "name": "CDI", "return_pct": cdi_return * 100,
            "vs_cdi_pct": 0.0,
        })
        if "Ibovespa" in benchmarks and ibov_return is not None:
            stats_list[-1]["vs_ibov_pct"] = (cdi_return - ibov_return) * 100

    if "Ibovespa" in benchmarks and ibov_return is not None:
        stats_list[0]["vs_ibov_pct"] = (stats["mean"] - ibov_return) * 100
        stats_list[1]["vs_ibov_pct"] = (stats["median"] - ibov_return) * 100
        stats_list.append({
            "name": "Ibovespa", "return_pct": ibov_return * 100,
            "vs_ibov_pct": 0.0,
        })
        if "CDI" in benchmarks:
            stats_list[-1]["vs_cdi_pct"] = (ibov_return - cdi_return) * 100

    stats_rows = pd.DataFrame(stats_list)
    # Ensure all cols exist in stats_rows
    for c in cols:
        if c not in stats_rows.columns:
            stats_rows[c] = None

    return df[cols], stats_rows[cols]


# ── History eligibility ──────────────────────────────────────────────────────

# Periods that require strict full-history coverage
_STRICT_PERIODS = {"1y", "3y", "5y", "10y"}

# Maximum allowed gap (calendar days) between period start and first data point
_MAX_START_GAP_DAYS = 30


def eligible_funds_for_period(
    daily_returns: pd.DataFrame,
    period: str,
    period_start: date,
    period_end: date,
) -> set[str]:
    """Return set of CNPJs that have sufficient history for a given period.

    For strict periods (1Y+), the fund's first data point in the window must
    be within _MAX_START_GAP_DAYS of the requested period_start.
    For short periods (day/mtd/ytd), all funds with any data are eligible.
    """
    if period not in _STRICT_PERIODS:
        mask = (daily_returns["date"] >= pd.Timestamp(period_start)) & \
               (daily_returns["date"] <= pd.Timestamp(period_end))
        return set(daily_returns.loc[mask, "CNPJ_FUNDO"].unique())

    mask = (daily_returns["date"] >= pd.Timestamp(period_start)) & \
           (daily_returns["date"] <= pd.Timestamp(period_end))
    window = daily_returns.loc[mask]

    first_dates = window.groupby("CNPJ_FUNDO")["date"].min()
    threshold = pd.Timestamp(period_start) + pd.Timedelta(days=_MAX_START_GAP_DAYS)
    return set(first_dates[first_dates <= threshold].index)


def period_return_strict(
    daily_returns: pd.DataFrame,
    period: str,
    period_start: date,
    period_end: date,
) -> pd.Series:
    """Like period_return, but excludes funds without full history for strict periods."""
    eligible = eligible_funds_for_period(daily_returns, period, period_start, period_end)
    mask = (daily_returns["date"] >= pd.Timestamp(period_start)) & \
           (daily_returns["date"] <= pd.Timestamp(period_end)) & \
           (daily_returns["CNPJ_FUNDO"].isin(eligible))
    df = daily_returns.loc[mask].copy()

    if df.empty:
        return pd.Series(dtype=float)

    def _total_return(group):
        return (1 + group["daily_return"]).prod() - 1

    return df.groupby("CNPJ_FUNDO").apply(_total_return, include_groups=False)


# ── Audit helpers ────────────────────────────────────────────────────────────

def build_audit_table(
    cnpj: str,
    fund_name: str,
    quotas: pd.DataFrame,
    daily_returns: pd.DataFrame,
    cdi_daily: pd.DataFrame,
    ref_date: date,
) -> pd.DataFrame:
    """Build a per-period audit table for a single fund.

    Returns DataFrame with columns:
        period, fund_name, cnpj, start_date_used, end_date_used,
        start_quota_used, end_quota_used, return_pct,
        row_count_in_window, has_full_history
    """
    # Prepare fund quotas sorted
    fq = quotas[quotas["CNPJ_FUNDO"] == cnpj][["DT_COMPTC", "VL_QUOTA"]].copy()
    fq["DT_COMPTC"] = pd.to_datetime(fq["DT_COMPTC"])
    fq.sort_values("DT_COMPTC", inplace=True)

    fr = daily_returns[daily_returns["CNPJ_FUNDO"] == cnpj].copy()
    fr.sort_values("date", inplace=True)

    periods = ["day", "mtd", "ytd", "1y", "3y", "5y", "10y"]
    labels = {"day": "Daily", "mtd": "MTD", "ytd": "YTD",
              "1y": "1Y", "3y": "3Y", "5y": "5Y", "10y": "10Y"}
    rows = []

    for p in periods:
        p_start, p_end = get_period_dates(p, ref_date=ref_date)

        # Window of daily returns
        mask = (fr["date"] >= pd.Timestamp(p_start)) & (fr["date"] <= pd.Timestamp(p_end))
        window = fr.loc[mask]
        row_count = len(window)

        # Eligibility
        eligible = eligible_funds_for_period(daily_returns, p, p_start, p_end)
        has_full = cnpj in eligible

        # Quota at start and end of window
        q_before_start = fq[fq["DT_COMPTC"] <= pd.Timestamp(p_start)]
        q_at_end = fq[fq["DT_COMPTC"] <= pd.Timestamp(p_end)]
        start_quota = float(q_before_start["VL_QUOTA"].iloc[-1]) if len(q_before_start) > 0 else None
        end_quota = float(q_at_end["VL_QUOTA"].iloc[-1]) if len(q_at_end) > 0 else None

        # Actual dates used
        start_date_used = q_before_start["DT_COMPTC"].iloc[-1].date() if len(q_before_start) > 0 else None
        end_date_used = q_at_end["DT_COMPTC"].iloc[-1].date() if len(q_at_end) > 0 else None

        # Return
        if has_full and start_quota and end_quota and start_quota > 0:
            ret = (end_quota / start_quota - 1) * 100
        else:
            ret = None

        rows.append({
            "period": labels[p],
            "fund_name": fund_name,
            "cnpj": cnpj,
            "start_date_used": start_date_used,
            "end_date_used": end_date_used,
            "start_quota": f"{start_quota:.6f}" if start_quota else "—",
            "end_quota": f"{end_quota:.6f}" if end_quota else "—",
            "return_pct": f"{ret:.2f}%" if ret is not None else "—",
            "rows_in_window": row_count,
            "has_full_history": has_full,
        })

    return pd.DataFrame(rows)
