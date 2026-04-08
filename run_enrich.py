"""CLI: enrich fund_registry.csv with subscription status from CVM registries.

Sources:
  1. registro_fundo_classe.zip → registro_classe.csv  (CVM175, primary)
       Fields used: Situacao, Forma_Condominio
  2. cad_fi.csv  (legacy, fallback for funds not in CVM175)
       Field used: SIT

Mapping (honest — CVM no longer publishes an explicit captação_aberta flag):
  - Situacao = "Em Liquidação" / "Cancelado"  → closed
  - Forma_Condominio = "Fechado"              → closed (closed-end structure)
  - Forma_Condominio = "Aberto", normal op    → unknown
  - legacy SIT = "CANCELADA" (not in CVM175)  → closed
  - not found in any source                   → unknown
  - "open" is never assigned by this script   (no explicit source for it in CVM data)

Manual overrides (subscription_status_source containing 'manual') are preserved.

Usage:
    python run_enrich.py             # enrich and save
    python run_enrich.py --dry-run  # resolve only, no file changes
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Enrich fund registry with subscription status from CVM cadastro"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and resolve statuses but do not write to fund_registry.csv",
    )
    args = parser.parse_args()

    from src.enrich_subscription import enrich_registry

    logger.info("=" * 60)
    logger.info("Subscription status enrichment — CVM cadastro")
    if args.dry_run:
        logger.info("DRY RUN — no files will be modified")
    logger.info("=" * 60)

    summary = enrich_registry(dry_run=args.dry_run)

    open_funds   = [r for r in summary["resolved"] if r.get("status") == "open"]
    closed_funds = [r for r in summary["resolved"] if r.get("status") == "closed"]
    unknown_funds = [r for r in summary["resolved"] if r.get("status") == "unknown"]

    logger.info("=" * 60)
    logger.info("Results:")
    logger.info("  Open (%d):", summary["open"])
    for r in open_funds:
        logger.info("    + %s", r["fund"])
    logger.info("  Closed (%d):", summary["closed"])
    for r in closed_funds:
        logger.info("    - %s", r["fund"])
    logger.info("  Unknown (%d):", summary["unknown"])
    for r in unknown_funds:
        logger.info("    ? %s — %s", r["fund"], r.get("note", ""))
    logger.info("  Skipped / manual override: %d", summary["skipped_manual"])
    logger.info("  No CNPJ: %d", summary["no_cnpj"])
    logger.info("=" * 60)

    if not args.dry_run:
        logger.info("fund_registry.csv updated. Restart the app to see changes.")


if __name__ == "__main__":
    main()
