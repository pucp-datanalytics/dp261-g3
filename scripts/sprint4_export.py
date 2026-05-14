#!/usr/bin/env python3
"""
Genera artefactos Sprint 4: models/tuned/{svm,gb,ada}_tuned.pkl, models/final_model.pkl
y añade filas a models/experiments_log.csv.

Requisito: datos en data/processed/04_default_credit_features.csv (p. ej. `dvc pull`).
Ejecutar desde cualquier cwd:

    python scripts/sprint4_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tuning import export_sprint4_artifacts  # noqa: E402


def main() -> None:
    try:
        info = export_sprint4_artifacts()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print("Sprint 4 — export OK")
    print("  final:", info["final_model"])
    print("  tuned:", info["tuned"])
    print("  métricas test (final):", info["test_metrics_final"])


if __name__ == "__main__":
    main()
