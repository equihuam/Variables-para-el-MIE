#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 14_create_data_table.py

Propósito:
    Integrar las tablas de características generadas previamente en una sola
    tabla de entrenamiento y serializar el resultado en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    14_create_data_table.R

Resumen del flujo:
    1. Leer las tablas serializadas de data_features.
    2. Usar la tabla de tasa de erosión como base.
    3. Incorporar secuencialmente las demás variables por join o por asignación
       de columnas, siguiendo la lógica del script original.
    4. Reindexar el identificador de píxel.
    5. Guardar la tabla final en data_training_tables.

Insumos principales:
    - conjunto de archivos .pkl en ./data_features/
    - tablas de erosión, movimiento de dunas, tipo de costa, viento, zvh,
      infraestructura, invasoras, corales, pastos marinos, batimetría,
      uso de suelo, manglares y condición de dunas

Salidas principales:
    - train_dat_2c.pkl

Supuestos y notas:
    - Esta traducción usa archivos .pkl en lugar de .rds por congruencia con
      el flujo Python.
    - Se conserva la lógica general de join/concatenación del script original.
    - Se evita depender del orden de list.files(), usando nombres de archivo
      esperados, porque esa dependencia es frágil y no forma parte del objetivo analítico.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, se corrige una fragilidad importante del script R original:
    en lugar de depender del orden de los archivos devueltos por list.files(),
    la versión en Python selecciona los insumos por nombre esperado.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
FEATURES_DIR = DROPBOX_DIR / "data_features"
OUTPUT_PKL = DROPBOX_DIR / "data_training_tables" / "train_dat_2c.pkl"


EXPECTED_FILES = {
    "erosion": "1_tasa_erosion.pkl",
    "mov_dunas": "10_movimiento_dunas.pkl",
    "tipo_costa": "11_tipo_costa.pkl",
    "wind": "12_avg_windspeed.pkl",
    "zvh": "13_zvh.pkl",
    "infra": "2_infraestructura.pkl",
    "sp_inv": "3_sp_inv_potential.pkl",
    "corales": "4_coral_distance.pkl",
    "pastos": "5_seagrass_distance.pkl",
    "batimetria": "6_batimetria_charact.pkl",
    "landuse": "7_madmex_landuse_3_1_10_12.pkl",
    "manglares": "8_manglares.pkl",
    "cond_dunas": "9_condicion_dunas.pkl",
}


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return pd.read_pickle(path)


def require_column(df: pd.DataFrame, column: str, table_name: str) -> None:
    if column not in df.columns:
        raise KeyError(f"No se encontró la columna '{column}' en {table_name}")


def main() -> None:
    erosion = load_feature_table(FEATURES_DIR / EXPECTED_FILES["erosion"])

    # Base
    dat = erosion.copy()

    # 10 movimiento dunas -> inner_join by x,y
    mov_dunas = load_feature_table(FEATURES_DIR / EXPECTED_FILES["mov_dunas"])
    dat = dat.merge(mov_dunas, on=["x", "y"], how="inner", suffixes=(".x", ".y"))

    # Replica:
    # dat$pixid.y = NULL
    # dat$regionid.y = NULL
    for col in ["pixid.y", "regionid.y"]:
        if col in dat.columns:
            dat = dat.drop(columns=col)

    # 11 tipo de costa
    tipo_costa = load_feature_table(FEATURES_DIR / EXPECTED_FILES["tipo_costa"])
    require_column(tipo_costa, "tipo_costa", "11_tipo_costa.pkl")
    dat["tipo_costa"] = tipo_costa["tipo_costa"].to_numpy()

    # 12 average windspeed
    wind = load_feature_table(FEATURES_DIR / EXPECTED_FILES["wind"])
    wind_col = "windspeed" if "windspeed" in wind.columns else "wind_speed"
    require_column(wind, wind_col, "12_avg_windspeed.pkl")
    dat["windspeed"] = wind[wind_col].to_numpy()

    # 13 zonas de vida de holdridge
    zvh = load_feature_table(FEATURES_DIR / EXPECTED_FILES["zvh"])
    require_column(zvh, "zvh", "13_zvh.pkl")
    dat["zvh"] = zvh["zvh"].to_numpy()

    # 2 infraestructura
    infra = load_feature_table(FEATURES_DIR / EXPECTED_FILES["infra"])

    infra_map = {
        "escollera": "Escollera",
        "espigon": "Espigón",
        "muro": "Muro",
        "rompeolas": "Rompeolas",
        "puerto": "Puerto",
    }
    for out_col, in_col in infra_map.items():
        require_column(infra, in_col, "2_infraestructura.pkl")
        dat[out_col] = infra[in_col].to_numpy()

    # 3 especies invasoras
    sp_inv = load_feature_table(FEATURES_DIR / EXPECTED_FILES["sp_inv"])
    require_column(sp_inv, "sp_inv_potential", "3_sp_inv_potential.pkl")
    dat["sp_inv_pot"] = sp_inv["sp_inv_potential"].to_numpy()

    # 4 distancia corales
    corales = load_feature_table(FEATURES_DIR / EXPECTED_FILES["corales"])
    require_column(corales, "corals", "4_coral_distance.pkl")
    dat["d_corales"] = corales["corals"].to_numpy()

    # 5 distancia pastos marinos
    pastos = load_feature_table(FEATURES_DIR / EXPECTED_FILES["pastos"])
    pastos_col = "grass" if "grass" in pastos.columns else "pasto"
    require_column(pastos, pastos_col, "5_seagrass_distance.pkl / 5_pasto_marino.pkl")
    dat["d_pastosmarinos"] = pastos[pastos_col].to_numpy()

    # 6 batimetria caracteristica
    bati = load_feature_table(FEATURES_DIR / EXPECTED_FILES["batimetria"])
    bati_col = "batrimetria" if "batrimetria" in bati.columns else "batimetria"
    require_column(bati, bati_col, "6_batimetria_charact.pkl")
    dat["bati_char"] = bati[bati_col].to_numpy()

    # 7 landuse
    landuse = load_feature_table(FEATURES_DIR / EXPECTED_FILES["landuse"])
    for col in ["grassland", "agriculture", "urban"]:
        require_column(landuse, col, "7_madmex_landuse_3_1_10_12.pkl")

    dat["d_grassland"] = landuse["grassland"].to_numpy()
    dat["d_agriculture"] = landuse["agriculture"].to_numpy()
    dat["d_urban"] = landuse["urban"].to_numpy()

    # 8 proporcion manglares
    manglares = load_feature_table(FEATURES_DIR / EXPECTED_FILES["manglares"])
    require_column(manglares, "manglares", "8_manglares.pkl")
    dat["p_manglares"] = manglares["manglares"].to_numpy()

    # 9 condicion dunas -> inner_join by x,y
    cond_dunas = load_feature_table(FEATURES_DIR / EXPECTED_FILES["cond_dunas"])
    dat = dat.merge(cond_dunas, on=["x", "y"], how="inner")

    # Replica:
    # dat$pixid = NULL
    # dat$regionid = NULL
    for col in ["pixid", "regionid"]:
        if col in dat.columns:
            dat = dat.drop(columns=col)

    # idx <- 1:nrow(dat)
    # dat$pixid.x <- idx
    dat["pixid.x"] = range(1, len(dat) + 1)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    dat.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()