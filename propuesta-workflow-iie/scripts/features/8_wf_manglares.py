#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 8_wf_manglares.py

Propósito:
    Estimar, para una región específica, la probabilidad de manglares en cada
    píxel de una tabla base regional ya alineada y exportar el resultado como
    tabla congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script R original:
    8_manglares.R

Resumen del flujo:
    1. Leer el shapefile de manglares.
    2. Leer una tabla base regional ya alineada.
    3. Verificar el contrato mínimo de la tabla base.
    4. Transformar las coordenadas x, y de la tabla base al CRS de manglares.
    5. Etiquetar cada punto base como manglar/no manglar según su intersección
       espacial con los polígonos de manglar.
    6. Ajustar un clasificador k-NN binario sobre esos puntos etiquetados.
    7. Estimar la probabilidad de manglar para cada punto base.
    8. Exportar la tabla regional en Parquet conservando exactamente la misma
       malla tabular de la base.

Insumos principales:
    - shapefile de manglares
    - tabla base regional alineada

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y, p_manglares

Supuestos y notas:
    - La malla canónica se toma de la tabla base regional.
    - La clasificación se realiza en el CRS del shapefile de manglares.
    - La lógica es funcionalmente equivalente al script R original:
      presencia/ausencia de manglar sobre la malla y estimación de probabilidad
      local mediante vecinos cercanos.
    - Si no hay ningún punto etiquetado como manglar, la salida es 0.
    - Si todos los puntos resultan manglar, la salida es 1.

Observaciones:
    - Este script está diseñado para integrarse en un workflow Snakemake.
    - La ejecución es por región y con rutas parametrizadas.
    - La salida es compatible con el contrato mínimo del proyecto para tablas
      de features congruentes por píxel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import pyarrow.parquet as pq
from shapely.geometry import Point
from sklearn.neighbors import KNeighborsClassifier


KEY_COLUMNS = ["regionid", "pixid", "x", "y"]
OUTPUT_FIELD = "p_manglares"
K_NEIGHBORS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula probabilidad de manglares usando una tabla base regional."
    )
    parser.add_argument(
        "--mangroves-shp",
        required=True,
        help="Ruta al shapefile de manglares.",
    )
    parser.add_argument(
        "--base-table",
        required=True,
        help="Ruta a la tabla base regional .parquet con regionid, pixid, x, y.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet.",
    )
    return parser.parse_args()


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def read_base_table(path: Path) -> pd.DataFrame:
    table = pq.read_table(path, use_threads=False)
    df = table.to_pandas()

    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"La tabla base no cumple el contrato mínimo. Faltan columnas: {missing}"
        )

    out = df[KEY_COLUMNS].copy()
    out["x"] = pd.to_numeric(out["x"], errors="raise")
    out["y"] = pd.to_numeric(out["y"], errors="raise")

    if out.empty:
        raise ValueError(f"La tabla base está vacía: {path}")

    return out


def load_mangroves(path: Path) -> gpd.GeoDataFrame:
    manglares = gpd.read_file(path)

    if manglares.empty:
        raise ValueError(f"El shapefile de manglares está vacío: {path}")
    if manglares.crs is None:
        raise ValueError(f"El shapefile de manglares no tiene CRS: {path}")

    return manglares


def label_points_against_mangroves(base: pd.DataFrame, manglares: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # La base canónica del workflow actual viene en EPSG:4326
    pts = gpd.GeoDataFrame(
        base.copy(),
        geometry=gpd.points_from_xy(base["x"], base["y"]),
        crs="EPSG:4326",
    )

    pts = pts.to_crs(manglares.crs)

    joined = gpd.sjoin(
        pts,
        manglares[["geometry"]],
        how="left",
        predicate="intersects",
    )

    joined["label"] = np.where(joined["index_right"].notna(), 1, 0).astype(int)
    joined = joined.drop(columns=["index_right"], errors="ignore")

    return joined


def predict_mangrove_probability(labeled_points: gpd.GeoDataFrame) -> np.ndarray:
    coords = np.column_stack([labeled_points.geometry.x.to_numpy(), labeled_points.geometry.y.to_numpy()])
    labels = labeled_points["label"].to_numpy(dtype=int)

    if len(labels) == 0:
        return np.array([], dtype=float)

    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        if unique_labels[0] == 0:
            return np.zeros(len(labels), dtype=float)
        return np.ones(len(labels), dtype=float)

    k_eff = min(K_NEIGHBORS, len(labels))

    clf = KNeighborsClassifier(
        n_neighbors=k_eff,
        weights="uniform",
        algorithm="auto",
        metric="euclidean",
    )
    clf.fit(coords, labels)

    probs = clf.predict_proba(coords)

    # la columna de probabilidad para clase 1
    class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}
    idx_1 = class_to_idx[1]
    return probs[:, idx_1].astype(float)


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Este script requiere salida .parquet. Recibido: {output_path.suffix}"
        )

    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()

    mangroves_path = Path(args.mangroves_shp)
    base_table_path = Path(args.base_table)
    output_path = Path(args.output)

    validate_inputs(mangroves_path, base_table_path)

    base = read_base_table(base_table_path)
    print(f"filas en base regional: {len(base)}")

    manglares = load_mangroves(mangroves_path)
    labeled_points = label_points_against_mangroves(base, manglares)

    probs = predict_mangrove_probability(labeled_points)

    out = base.copy()
    out[OUTPUT_FIELD] = probs

    save_output(out, output_path)


if __name__ == "__main__":
    main()