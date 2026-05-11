#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 16_final_training_data_v1.py

Propósito:
    Preparar la versión final de la tabla de entrenamiento, exportarla en
    formatos PKL/CSV y generar una versión discretizada apta para uso posterior,
    incluyendo una variante con '*' en ei_qnint para Netica.

Origen:
    Traducción inicial a Python del script R:
    16_final_training_data_v1.R

Resumen del flujo:
    1. Leer la tabla integrada de entrenamiento.
    2. Seleccionar el subconjunto de columnas definido para la tabla final.
    3. Exportar la tabla final en PKL y CSV.
    4. Discretizar las variables continuas.
    5. Exportar la versión discretizada.
    6. Reemplazar '0' por '*' en ei_qnint y exportar la variante final.

Insumos principales:
    - train_dat_2c.pkl

Salidas principales:
    - cei_final_train_v1.pkl
    - cei_final_train_v1.csv
    - cei_final_train_v1d.csv
    - cei_final_train_v1ask.csv

Supuestos y notas:
    - Esta traducción usa archivos .pkl en lugar de .rds por congruencia con
      el flujo Python.
    - La discretización se implementa como aproximación a discretizeCols()
      usando 5 intervalos iguales, igual que en misc_functions.R.
    - La columna ei_qnint debe existir previamente en la tabla de entrada.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, la principal aproximación corresponde a discretizeCols(),
    cuya implementación en R usa bnlearn::discretize con 5 intervalos iguales.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
TRAIN_TABLE = DROPBOX_DIR / "data_training_tables" / "train_dat_2c.pkl"
OUTPUT_DIR = DROPBOX_DIR / "data_training_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CON_VARS = [
    "erosion", "windspeed", "escollera", "espigon", "muro", "rompeolas",
    "puerto", "sp_inv_pot", "d_corales", "d_pastosmarinos", "bati_char",
    "d_grassland", "d_agriculture", "d_urban", "p_manglares",
]

FINAL_COLS = [
    "x", "y", "regionid.x",
    "NESTB_EDO", "CONSERV_ED", "erosion", "movdunas", "tipo_costa",
    "windspeed", "zvh", "escollera", "espigon", "muro", "rompeolas",
    "puerto", "sp_inv_pot", "d_corales", "d_pastosmarinos", "bati_char",
    "d_grassland", "d_agriculture", "d_urban", "p_manglares", "ei_qnint",
]


def discretize_cols(
        df: pd.DataFrame,
        cols: list[str],
        breaks: int = 5,
        method: str = "interval",
) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        s = pd.to_numeric(out[col], errors="coerce")

        if s.nunique(dropna=True) <= 1:
            out[col] = pd.Series(["1"] * len(s), index=s.index, dtype="object")
            continue

        if method != "interval":
            raise ValueError("Esta traducción de 16 usa method='interval' como en R.")

        disc = pd.cut(s, bins=breaks, labels=False, include_lowest=True)
        out[col] = (disc + 1).astype("Int64").astype(str)

    return out


def main() -> None:
    if not TRAIN_TABLE.exists():
        raise FileNotFoundError(f"No existe la tabla de entrada: {TRAIN_TABLE}")

    dat = pd.read_pickle(TRAIN_TABLE)

    existing_cols = [c for c in FINAL_COLS if c in dat.columns]
    final_train = dat[existing_cols].copy()

    final_train.to_pickle(OUTPUT_DIR / "cei_final_train_v1.pkl")
    final_train.to_csv(OUTPUT_DIR / "cei_final_train_v1.csv", index=False)

    datd = discretize_cols(final_train, CON_VARS, breaks=5, method="interval")
    datd.to_csv(OUTPUT_DIR / "cei_final_train_v1d.csv", index=False)

    datd_netica = datd.copy()
    if "ei_qnint" in datd_netica.columns:
        datd_netica["ei_qnint"] = datd_netica["ei_qnint"].astype(str)
        datd_netica.loc[datd_netica["ei_qnint"] == "0", "ei_qnint"] = "*"

    datd_netica.to_csv(OUTPUT_DIR / "cei_final_train_v1d.csv", index=False)
    datd_netica.to_csv(OUTPUT_DIR / "cei_final_train_v1ask.csv", index=False)

    print(f"OK -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()