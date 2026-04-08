"""Enrich fund_registry.csv with subscription status from CVM official registries.

Sources used (in priority order):
  1. CVM175 class registry — registro_fundo_classe.zip → registro_classe.csv
       Fields: Situacao, Forma_Condominio
       URL: https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip
  2. Legacy CVM fund cadastro — cad_fi.csv
       Field: SIT (situation)
       URL: https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv
       Only used as fallback for funds NOT found in CVM175 registry.

Status mapping (explicit, no inference from flows or returns):
  - CVM175 found + Situacao = "Em Liquidação" → closed
  - CVM175 found + Situacao = "Cancelado"     → closed
  - CVM175 found + Forma_Condominio = "Fechado" → closed (closed-end structure; no
      ongoing subscriptions by CVM regulation)
  - CVM175 found + Forma_Condominio = "Aberto" + normal operation → unknown
      (CVM no longer publishes explicit captação_aberta in its open-data feed; the
       open-end structure is necessary but not sufficient to confirm captação is open)
  - NOT in CVM175 + legacy SIT = "CANCELADA"   → closed
      (fund registration cancelled and not migrated; definitively inactive)
  - Not found in any source                     → unknown

Manual override protection:
  If subscription_status_source already contains the word "manual"
  (case-insensitive), that row is skipped — never overwritten.

Usage:
    from src.enrich_subscription import enrich_registry
    summary = enrich_registry()           # fetch live + write to CSV
    summary = enrich_registry(dry_run=True)  # fetch live, no file changes
"""

import io
import logging
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

from src.utils import CONFIG_DIR, normalize_cnpj

logger = logging.getLogger(__name__)

# ── CVM URLs ─────────────────────────────────────────────────────────────────

_URL_CLASSE_ZIP = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip"
_URL_LEGACY_CSV = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"

SOURCE_CVM175 = "CVM registro_classe (dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip)"
SOURCE_LEGACY = "CVM cad_fi (dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv)"

_ENRICH_COLS = [
    "subscription_status",
    "subscription_status_source",
    "subscription_status_checked_at",
    "evidence_note",
]


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (fund-peer-monitor)"
    return s


def _get(session: requests.Session, url: str, timeout: int = 120) -> requests.Response | None:
    """GET with basic error handling; returns None on 404/403."""
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code in (403, 404):
            logger.warning("HTTP %s for %s", resp.status_code, url)
            return None
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_cvm175_registro(session: requests.Session) -> pd.DataFrame:
    """Download registro_fundo_classe.zip and return registro_classe.csv as DataFrame.

    Columns returned: cnpj_norm (str), Situacao (str), Forma_Condominio (str).
    Returns empty DataFrame on failure.
    """
    logger.info("Fetching CVM175 registro_classe: %s", _URL_CLASSE_ZIP)
    resp = _get(session, _URL_CLASSE_ZIP, timeout=120)
    if resp is None:
        return pd.DataFrame(columns=["cnpj_norm", "Situacao", "Forma_Condominio"])

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile:
        logger.warning("Invalid zip from %s", _URL_CLASSE_ZIP)
        return pd.DataFrame(columns=["cnpj_norm", "Situacao", "Forma_Condominio"])

    if "registro_classe.csv" not in zf.namelist():
        logger.warning("registro_classe.csv not found in zip (files: %s)", zf.namelist())
        return pd.DataFrame(columns=["cnpj_norm", "Situacao", "Forma_Condominio"])

    with zf.open("registro_classe.csv") as f:
        text = f.read().decode("latin-1")

    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, low_memory=False)

    if "CNPJ_Classe" not in df.columns:
        logger.warning("CNPJ_Classe column missing from registro_classe.csv")
        return pd.DataFrame(columns=["cnpj_norm", "Situacao", "Forma_Condominio"])

    df["cnpj_norm"] = df["CNPJ_Classe"].apply(
        lambda x: normalize_cnpj(str(x)) if pd.notna(x) and str(x).strip() else ""
    )
    for col in ("Situacao", "Forma_Condominio"):
        if col not in df.columns:
            df[col] = ""

    df = df[df["cnpj_norm"] != ""]
    df = df.drop_duplicates(subset=["cnpj_norm"], keep="first")
    logger.info("CVM175 registro_classe loaded: %d entries", len(df))
    return df[["cnpj_norm", "Situacao", "Forma_Condominio"]].reset_index(drop=True)


def _fetch_legacy_cadastro(session: requests.Session) -> pd.DataFrame:
    """Download cad_fi.csv and return relevant columns.

    Columns returned: cnpj_norm (str), SIT (str).
    Returns empty DataFrame on failure.
    """
    logger.info("Fetching CVM legacy cad_fi: %s", _URL_LEGACY_CSV)
    resp = _get(session, _URL_LEGACY_CSV, timeout=120)
    if resp is None:
        return pd.DataFrame(columns=["cnpj_norm", "SIT"])

    text = resp.content.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, low_memory=False)

    if "CNPJ_FUNDO" not in df.columns:
        logger.warning("CNPJ_FUNDO column missing from cad_fi.csv")
        return pd.DataFrame(columns=["cnpj_norm", "SIT"])

    df["cnpj_norm"] = df["CNPJ_FUNDO"].apply(
        lambda x: normalize_cnpj(str(x)) if pd.notna(x) and str(x).strip() else ""
    )
    if "SIT" not in df.columns:
        df["SIT"] = ""

    df = df[df["cnpj_norm"] != ""]
    df = df.drop_duplicates(subset=["cnpj_norm"], keep="first")
    logger.info("CVM legacy cad_fi loaded: %d entries", len(df))
    return df[["cnpj_norm", "SIT"]].reset_index(drop=True)


# ── Status resolution ─────────────────────────────────────────────────────────

def _resolve_from_cvm175(row: pd.Series) -> tuple[str, str, str]:
    """Return (status, source, evidence_note) from a CVM175 registro_classe row."""
    situacao = str(row.get("Situacao", "")).strip()
    forma = str(row.get("Forma_Condominio", "")).strip()

    sit_norm = situacao.lower()
    forma_norm = forma.lower()

    if "liquidação" in sit_norm or "liquidacao" in sit_norm or sit_norm == "cancelado":
        return (
            "closed",
            SOURCE_CVM175,
            f"Situacao={situacao!r} (fund in wind-down or cancelled per CVM175 registro)",
        )

    if forma_norm == "fechado":
        return (
            "closed",
            SOURCE_CVM175,
            f"Forma_Condominio=Fechado; Situacao={situacao!r} — closed-end fund structure; "
            "by CVM regulation closed-end funds do not accept ongoing subscriptions",
        )

    # Open-end fund in normal or other non-terminal operation
    if forma_norm == "aberto":
        return (
            "unknown",
            SOURCE_CVM175,
            f"Situacao={situacao!r}; Forma_Condominio=Aberto — open-end structure confirmed active; "
            "CVM open-data no longer publishes explicit captação_aberta flag",
        )

    # Forma_Condominio not present or unrecognised
    if not forma or forma in ("nan", ""):
        return (
            "unknown",
            SOURCE_CVM175,
            f"Situacao={situacao!r}; Forma_Condominio not available in CVM175 registro",
        )

    return (
        "unknown",
        SOURCE_CVM175,
        f"Situacao={situacao!r}; Forma_Condominio={forma!r} (unrecognised value)",
    )


def _resolve_from_legacy(row: pd.Series) -> tuple[str, str, str]:
    """Return (status, source, evidence_note) from a cad_fi.csv row.

    Only called for funds NOT found in CVM175 registro_classe.
    """
    sit = str(row.get("SIT", "")).strip()
    sit_norm = sit.lower()

    if sit_norm in ("cancelada", "cancelado", "encerrada", "encerrado"):
        return (
            "closed",
            SOURCE_LEGACY,
            f"SIT={sit!r} — fund registration cancelled in CVM legacy cadastro; "
            "not found in CVM175 registro (fund did not migrate or was merged)",
        )

    if "liquidação" in sit_norm or "liquidacao" in sit_norm:
        return (
            "closed",
            SOURCE_LEGACY,
            f"SIT={sit!r} — fund in liquidation per CVM legacy cadastro",
        )

    return (
        "unknown",
        SOURCE_LEGACY,
        f"SIT={sit!r} — found in CVM legacy cadastro; "
        "no explicit captação_aberta field available",
    )


# ── Main enrichment ───────────────────────────────────────────────────────────

def enrich_registry(dry_run: bool = False) -> dict:
    """Enrich fund_registry.csv with subscription status from CVM registries.

    Args:
        dry_run: if True, resolve statuses but do NOT write to disk.

    Returns:
        Summary dict with counts and per-fund resolution details.
    """
    path = CONFIG_DIR / "fund_registry.csv"
    registry = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    # Ensure enrichment columns exist
    for col in _ENRICH_COLS:
        if col not in registry.columns:
            registry[col] = ""

    registry["_cnpj_norm"] = registry["cnpj"].apply(
        lambda x: normalize_cnpj(str(x)) if pd.notna(x) and str(x).strip() else ""
    )

    # Fetch both CVM sources
    sess = _session()
    df_cvm175 = _fetch_cvm175_registro(sess)
    df_legacy = _fetch_legacy_cadastro(sess)

    # Build lookup dicts
    cvm175_lookup = (
        df_cvm175.set_index("cnpj_norm").to_dict("index") if not df_cvm175.empty else {}
    )
    legacy_lookup = (
        df_legacy.set_index("cnpj_norm").to_dict("index") if not df_legacy.empty else {}
    )

    logger.info(
        "Lookup ready: %d CVM175 entries, %d legacy entries",
        len(cvm175_lookup), len(legacy_lookup),
    )

    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts = {"open": 0, "closed": 0, "unknown": 0, "skipped_manual": 0, "no_cnpj": 0}
    resolved: list[dict] = []

    for idx, row in registry.iterrows():
        cnpj_norm = str(row.get("_cnpj_norm", "")).strip()
        display = row.get("display_name", f"row_{idx}")

        if not cnpj_norm:
            counts["no_cnpj"] += 1
            resolved.append({"fund": display, "status": "—", "reason": "no CNPJ"})
            continue

        # Respect manual overrides
        current_source = str(row.get("subscription_status_source", "")).strip().lower()
        if "manual" in current_source:
            counts["skipped_manual"] += 1
            current_status = str(row.get("subscription_status", "unknown")).strip()
            resolved.append({
                "fund": display,
                "cnpj": cnpj_norm,
                "status": current_status,
                "reason": "manual override preserved",
            })
            logger.info("[SKIP] %s — manual override preserved (%s)", display, current_status)
            continue

        # Resolution: CVM175 first, then legacy fallback
        cvm175_entry = cvm175_lookup.get(cnpj_norm)
        legacy_entry = legacy_lookup.get(cnpj_norm)

        if cvm175_entry is not None:
            status, source, note = _resolve_from_cvm175(pd.Series(cvm175_entry))
        elif legacy_entry is not None:
            status, source, note = _resolve_from_legacy(pd.Series(legacy_entry))
        else:
            status = "unknown"
            source = ""
            note = "CNPJ not found in CVM175 registro_classe or legacy cad_fi"

        counts[status] += 1
        resolved.append({
            "fund": display,
            "cnpj": cnpj_norm,
            "status": status,
            "note": note,
        })
        logger.info("[%s] %s (%s) | %s", status.upper(), display, cnpj_norm, note)

        if not dry_run:
            registry.at[idx, "subscription_status"] = status
            registry.at[idx, "subscription_status_source"] = source
            registry.at[idx, "subscription_status_checked_at"] = now_ts
            registry.at[idx, "evidence_note"] = note

    if not dry_run:
        save_df = registry.drop(columns=["_cnpj_norm"], errors="ignore")
        save_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("Registry saved: %s (%d rows)", path, len(save_df))

    counts["resolved"] = resolved
    return counts
