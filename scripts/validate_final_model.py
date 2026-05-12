#!/usr/bin/env python3
"""
Comprueba que models/final_model.pkl existe, se puede cargar con joblib y
ejecuta predict / predict_proba sobre una muestra si hay datos locales.

    python scripts/validate_final_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import DEFAULT_DATA_PATH, load_data  # noqa: E402


def main() -> None:
    model_path = ROOT / "models" / "final_model.pkl"
    if not model_path.is_file():
        print("ERROR: no existe models/final_model.pkl — ejecuta scripts/sprint4_export.py", file=sys.stderr)
        sys.exit(1)

    model = joblib.load(model_path)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        print("ERROR: el objeto cargado no parece un clasificador sklearn.", file=sys.stderr)
        sys.exit(1)

    if not DEFAULT_DATA_PATH.is_file():
        print("Modelo cargado correctamente. Sin CSV local para smoke test (opcional: dvc pull).")
        sys.exit(0)

    X, _ = load_data()
    sample = X.iloc[:128]
    proba = model.predict_proba(sample)
    pred = model.predict(sample)
    if proba.shape != (len(sample), 2):
        print(f"ERROR: forma predict_proba inesperada {proba.shape}", file=sys.stderr)
        sys.exit(1)
    print("Smoke test OK:", "predict_proba", proba.shape, "predict", pred.shape)


if __name__ == "__main__":
    main()
