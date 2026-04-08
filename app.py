"""Streamlit dashboard: Brazilian Fund Peer Comparison.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

from src.utils import CONFIG_DIR, DATA_PROCESSED, normalize_cnpj, format_cnpj
from src.processing import (
    period_return_strict,
    cdi_period_return,
    ibov_period_return,
    get_period_dates,
    build_ranking,
    cumulative_returns,
    cumulative_cdi,
    cumulative_ibov,
    build_audit_table,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fund Peer Monitor", layout="wide")
st.title("Fund Peer Monitor")


# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_registry():
    """Load fund registry CSV. Returns DataFrame."""
    path = CONFIG_DIR / "fund_registry.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["active"] = df["active"].str.strip().str.lower() == "true"
    df["cnpj_norm"] = df["cnpj"].apply(
        lambda x: normalize_cnpj(x) if pd.notna(x) and str(x).strip() else ""
    )
    _valid_statuses = {"open", "closed", "unknown"}
    df["subscription_status"] = (
        df.get("subscription_status", pd.Series("unknown", index=df.index))
        .fillna("unknown").str.strip().str.lower()
        .apply(lambda v: v if v in _valid_statuses else "unknown")
    )
    return df


@st.cache_data(ttl=300)
def load_market_data():
    """Load processed parquet files."""
    fund_returns_path = DATA_PROCESSED / "fund_returns.parquet"
    cdi_returns_path = DATA_PROCESSED / "cdi_returns.parquet"
    ibov_returns_path = DATA_PROCESSED / "ibov_returns.parquet"
    quotas_path = DATA_PROCESSED / "quotas.parquet"

    if not fund_returns_path.exists() or not cdi_returns_path.exists():
        return None, None, None, None

    fund_ret = pd.read_parquet(fund_returns_path)
    fund_ret["date"] = pd.to_datetime(fund_ret["date"])

    cdi_ret = pd.read_parquet(cdi_returns_path)
    cdi_ret["date"] = pd.to_datetime(cdi_ret["date"])

    ibov_ret = None
    if ibov_returns_path.exists():
        ibov_ret = pd.read_parquet(ibov_returns_path)
        ibov_ret["date"] = pd.to_datetime(ibov_ret["date"])

    quotas = None
    if quotas_path.exists():
        quotas = pd.read_parquet(quotas_path)
        quotas["DT_COMPTC"] = pd.to_datetime(quotas["DT_COMPTC"])

    return fund_ret, cdi_ret, ibov_ret, quotas


registry = load_registry()
fund_ret, cdi_ret, ibov_ret, quotas = load_market_data()

if fund_ret is None:
    st.error(
        "No data found. Run the ingestion first:\n\n"
        "```\npython run_ingest.py\n```"
    )
    st.stop()


# ── Sidebar: bucket + fund + benchmark selection ────────────────────────────

st.sidebar.header("Settings")

# Strategy bucket selector
buckets = sorted(registry[registry["active"]]["strategy_bucket"].unique())
selected_bucket = st.sidebar.selectbox(
    "Strategy bucket", buckets,
    index=buckets.index("Multimercados") if "Multimercados" in buckets else 0,
)

# Subscription status filter
_STATUS_OPTIONS = ["all", "open", "closed", "unknown"]
_STATUS_LABELS = {
    "all": "All",
    "open": "Only open",
    "closed": "Only closed",
    "unknown": "Only unknown",
}
sub_status_filter = st.sidebar.selectbox(
    "Subscription status",
    options=_STATUS_OPTIONS,
    format_func=lambda x: _STATUS_LABELS[x],
    index=0,
)

# Filter registry to selected bucket
bucket_funds = registry[
    (registry["strategy_bucket"] == selected_bucket) & registry["active"]
].copy()

# Apply subscription status filter
if sub_status_filter != "all":
    bucket_funds = bucket_funds[
        bucket_funds["subscription_status"] == sub_status_filter
    ]

# Separate funds with and without CNPJ
bucket_with_cnpj = bucket_funds[bucket_funds["cnpj_norm"] != ""]
bucket_no_cnpj = bucket_funds[bucket_funds["cnpj_norm"] == ""]

# Build names dict for funds with CNPJ (only these can show data)
names = {row["cnpj_norm"]: row["display_name"] for _, row in bucket_with_cnpj.iterrows()}

# CNPJs that actually have return data loaded
cnpjs_with_data = set(fund_ret["CNPJ_FUNDO"].unique()) & set(names.keys())
names_with_data = {c: names[c] for c in cnpjs_with_data}

# Fund highlight selector (only funds with data)
all_fund_names = sorted(names_with_data.values())
if all_fund_names:
    selected_funds = st.sidebar.multiselect(
        "Highlight funds",
        options=all_fund_names,
        default=all_fund_names[:1],
    )
else:
    selected_funds = []
    st.sidebar.info("No funds with data in this bucket yet.")

selected_cnpjs = {cnpj for cnpj, name in names_with_data.items() if name in selected_funds}

# Benchmark selector
benchmark_options = ["CDI"]
if ibov_ret is not None and not ibov_ret.empty:
    benchmark_options.append("Ibovespa")

selected_benchmarks = st.sidebar.multiselect(
    "Benchmarks",
    options=benchmark_options,
    default=["CDI"],
)
if not selected_benchmarks:
    selected_benchmarks = ["CDI"]

# Show funds without CNPJ or data
if len(bucket_no_cnpj) > 0 or len(names) > len(names_with_data):
    with st.sidebar.expander(f"Funds pending data ({len(bucket_funds) - len(names_with_data)})"):
        for _, row in bucket_no_cnpj.iterrows():
            st.caption(f"  {row['display_name']} — no CNPJ")
        for c in set(names.keys()) - cnpjs_with_data:
            st.caption(f"  {names[c]} — CNPJ ok, no data loaded")

# Date range info
min_date = fund_ret["date"].min().date()
max_date = fund_ret["date"].max().date()
cdi_max_date = cdi_ret["date"].max().date()

st.sidebar.markdown("---")
st.sidebar.caption("Data availability")
st.sidebar.caption(f"Fund data: {min_date} to **{max_date}**")
st.sidebar.caption(f"CDI data: to **{cdi_max_date}**")

if ibov_ret is not None and not ibov_ret.empty:
    ibov_max_date = ibov_ret["date"].max().date()
    st.sidebar.caption(f"Ibovespa data: to **{ibov_max_date}**")

for cnpj in selected_cnpjs:
    fd = fund_ret[fund_ret["CNPJ_FUNDO"] == cnpj]
    if not fd.empty:
        st.sidebar.caption(
            f"{names_with_data.get(cnpj, cnpj)}: {fd['date'].min().date()} to **{fd['date'].max().date()}**"
        )


# ── Custom date range (sidebar) ────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Date Range")
custom_start = st.sidebar.date_input(
    "Start date", value=date(max_date.year - 1, max_date.month, max_date.day),
    min_value=min_date, max_value=max_date,
    key="custom_start",
)
custom_end = st.sidebar.date_input(
    "End date", value=max_date,
    min_value=min_date, max_value=max_date,
    key="custom_end",
)
if custom_start >= custom_end:
    st.sidebar.warning("Start date must be before end date.")


# ── Guard: need at least one fund with data ──────────────────────────────────

if not names_with_data:
    st.warning(
        f"No funds with loaded data in **{selected_bucket}**. "
        "Add CNPJs to `config/fund_registry.csv` and re-run ingestion."
    )
    st.stop()

# Filter fund_ret to only bucket funds
bucket_cnpjs = set(names_with_data.keys())
fund_ret_bucket = fund_ret[fund_ret["CNPJ_FUNDO"].isin(bucket_cnpjs)].copy()


# ── Ranking tables ───────────────────────────────────────────────────────────

st.header(f"Performance Rankings — {selected_bucket}")

PERIODS = ["day", "mtd", "ytd", "1y", "3y", "5y", "10y", "custom"]
PERIOD_LABELS = {
    "day": "Daily", "mtd": "MTD", "ytd": "YTD",
    "1y": "1Y", "3y": "3Y", "5y": "5Y", "10y": "10Y",
    "custom": "Custom",
}

tabs = st.tabs([PERIOD_LABELS[p] for p in PERIODS])

for tab, period in zip(tabs, PERIODS):
    with tab:
        # Resolve start/end for this period
        if period == "custom":
            if custom_start >= custom_end:
                st.warning("Set a valid custom date range in the sidebar.")
                continue
            start = custom_start
            end = custom_end
            st.caption(f"Custom range: **{start}** to **{end}**")
        else:
            try:
                start, end = get_period_dates(period, ref_date=max_date)
            except ValueError:
                st.warning(f"Could not compute dates for {period}")
                continue

        effective_start = max(start, min_date)

        # Use period="custom" for strict eligibility (treated as non-strict)
        p_returns = period_return_strict(fund_ret_bucket, period, effective_start, end)
        cdi_ret_total = cdi_period_return(cdi_ret, effective_start, end)

        ibov_ret_total = None
        if "Ibovespa" in selected_benchmarks and ibov_ret is not None:
            ibov_ret_total = ibov_period_return(ibov_ret, effective_start, end)

        if p_returns.empty:
            st.info(f"No data available for {PERIOD_LABELS[period]}")
            continue

        ranking_df, stats_df = build_ranking(
            p_returns, cdi_ret_total, names_with_data,
            ibov_return=ibov_ret_total,
            benchmarks=selected_benchmarks,
        )

        # Mark selected funds
        name_to_cnpj = {v: k for k, v in names_with_data.items()}
        ranking_df = ranking_df.copy()
        ranking_df["name"] = ranking_df["name"].apply(
            lambda n: f">> {n}" if name_to_cnpj.get(n, "") in selected_cnpjs else n
        )

        # Build format dict dynamically
        fmt = {"return_pct": "{:.2f}%"}
        if "vs_cdi_pct" in ranking_df.columns:
            fmt["vs_cdi_pct"] = "{:+.2f}%"
        if "vs_ibov_pct" in ranking_df.columns:
            fmt["vs_ibov_pct"] = "{:+.2f}%"

        st.dataframe(
            ranking_df.style.format(fmt),
            use_container_width=True,
        )

        st.caption("Benchmarks")
        stats_fmt = {"return_pct": "{:.2f}%"}
        if "vs_cdi_pct" in stats_df.columns:
            stats_fmt["vs_cdi_pct"] = "{:+.2f}%"
        if "vs_ibov_pct" in stats_df.columns:
            stats_fmt["vs_ibov_pct"] = "{:+.2f}%"

        st.dataframe(
            stats_df.style.format(stats_fmt),
            use_container_width=True,
            hide_index=True,
        )


# ── Cumulative performance chart ─────────────────────────────────────────────

st.header("Cumulative Performance")

CHART_PERIODS = ["mtd", "ytd", "1y", "3y", "5y", "10y", "custom"]
CHART_LABELS = {k: PERIOD_LABELS[k] for k in CHART_PERIODS}

chart_period = st.selectbox(
    "Chart period", CHART_PERIODS, index=1,
    format_func=lambda x: CHART_LABELS[x],
)

if chart_period == "custom":
    if custom_start >= custom_end:
        st.error("Set a valid custom date range in the sidebar.")
        st.stop()
    chart_start = custom_start
    chart_end = custom_end
    st.caption(f"Custom range: **{chart_start}** to **{chart_end}**")
else:
    try:
        chart_start, chart_end = get_period_dates(chart_period, ref_date=max_date)
    except ValueError:
        st.error("Invalid period")
        st.stop()

chart_start = max(chart_start, min_date)

cum_ret = cumulative_returns(fund_ret_bucket, chart_start, chart_end)

fig = go.Figure()

# Non-highlighted funds: gray
non_selected = bucket_cnpjs - selected_cnpjs
for cnpj in non_selected:
    fund_data = cum_ret[cum_ret["CNPJ_FUNDO"] == cnpj]
    if fund_data.empty:
        continue
    fig.add_trace(go.Scatter(
        x=fund_data["date"],
        y=(fund_data["cumulative_return"] - 1) * 100,
        mode="lines",
        name=names_with_data.get(cnpj, cnpj),
        line=dict(color="lightgray", width=1),
        showlegend=False,
        hovertemplate="%{fullData.name}: %{y:.2f}%<extra></extra>",
    ))

# Peer mean and median
all_cum = cum_ret.pivot(index="date", columns="CNPJ_FUNDO", values="cumulative_return")
if not all_cum.empty:
    peer_mean = all_cum.mean(axis=1)
    peer_median = all_cum.median(axis=1)

    fig.add_trace(go.Scatter(
        x=peer_mean.index, y=(peer_mean - 1) * 100,
        mode="lines", name="Peer Mean",
        line=dict(color="blue", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=peer_median.index, y=(peer_median - 1) * 100,
        mode="lines", name="Peer Median",
        line=dict(color="purple", width=2, dash="dot"),
    ))

# CDI benchmark line
if "CDI" in selected_benchmarks:
    cum_cdi = cumulative_cdi(cdi_ret, chart_start, chart_end)
    if not cum_cdi.empty:
        fig.add_trace(go.Scatter(
            x=cum_cdi["date"], y=(cum_cdi["cumulative_return"] - 1) * 100,
            mode="lines", name="CDI",
            line=dict(color="black", width=2),
        ))

# Ibovespa benchmark line
if "Ibovespa" in selected_benchmarks and ibov_ret is not None:
    cum_ibov = cumulative_ibov(ibov_ret, chart_start, chart_end)
    if not cum_ibov.empty:
        fig.add_trace(go.Scatter(
            x=cum_ibov["date"], y=(cum_ibov["cumulative_return"] - 1) * 100,
            mode="lines", name="Ibovespa",
            line=dict(color="orange", width=2, dash="dashdot"),
        ))

# Highlighted funds
colors = ["#e63946", "#2a9d8f", "#e9c46a", "#264653", "#f4a261", "#457b9d"]
for i, cnpj in enumerate(selected_cnpjs):
    fund_data = cum_ret[cum_ret["CNPJ_FUNDO"] == cnpj]
    if fund_data.empty:
        continue
    fig.add_trace(go.Scatter(
        x=fund_data["date"],
        y=(fund_data["cumulative_return"] - 1) * 100,
        mode="lines",
        name=names_with_data.get(cnpj, cnpj),
        line=dict(color=colors[i % len(colors)], width=3),
    ))

fig.update_layout(
    yaxis_title="Cumulative Return (%)",
    xaxis_title="Date",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
    margin=dict(l=40, r=20, t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)


# ── Audit section ────────────────────────────────────────────────────────────

st.header("Audit & Diagnostics")

if not selected_cnpjs:
    st.info("Select a fund in the sidebar to see audit details.")
else:
    for cnpj in selected_cnpjs:
        fund_name = names_with_data.get(cnpj, cnpj)

        st.subheader(f"{fund_name}")
        st.caption(f"CNPJ: `{format_cnpj(cnpj)}`")

        fd = fund_ret[fund_ret["CNPJ_FUNDO"] == cnpj]
        if fd.empty:
            st.warning("No return data for this fund.")
            continue

        fund_first = fd["date"].min().date()
        fund_last = fd["date"].max().date()
        st.caption(f"Data range: {fund_first} to {fund_last} ({len(fd)} daily return rows)")

        if quotas is not None:
            audit_df = build_audit_table(
                cnpj=cnpj,
                fund_name=fund_name,
                quotas=quotas,
                daily_returns=fund_ret,
                cdi_daily=cdi_ret,
                ref_date=max_date,
            )
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Quota data not available for audit.")


# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
footer_parts = [
    "Data source: CVM (dados.cvm.gov.br)",
    "CDI: BCB SGS series 12",
    "Ibovespa: Yahoo Finance (^BVSP)",
    f"Fund data through: {max_date}",
    f"CDI through: {cdi_max_date}",
]
if ibov_ret is not None and not ibov_ret.empty:
    footer_parts.append(f"Ibovespa through: {ibov_ret['date'].max().date()}")
footer_parts.append(f"Generated: {date.today()}")

st.caption(" | ".join(footer_parts))
