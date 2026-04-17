#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import from_origin, rowcol
from rasterio.warp import reproject, Resampling


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal/")
TRAINING_TABLE = DROPBOX_DIR / "data" / "cei_final_train_v1ask.csv"
IE_PREDICTIONS_CSV = DROPBOX_DIR / "BN-results" / "EII-data" / "cei_final_ie_expected_port_5_equal_2026.csv"
REF_GRID_DIR = DROPBOX_DIR / "data" / "data_crude" / "DunasCost250116_malla_ref_50m"
OUTPUT_DIR = DROPBOX_DIR / "BN-results" / "BN_maps" / "cei_final_ie_expected_port_5_equal_2026/Python"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CRS = "EPSG:4326"
NODATA_VALUE = -9999.0


def normalize_ie_predictions(values: np.ndarray) -> np.ndarray:
    """Normaliza valores de 1.5–5.5 a 0–1."""
    return (values - 1.5) / (5.5 - 1.5)


def load_training_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, low_memory=False)

    required = ["regionId", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en training table: {missing}")

    df = df[required].copy()
    df["idx"] = np.arange(1, len(df) + 1)

    df["x"] = pd.to_numeric(df["x"], errors="raise")
    df["y"] = pd.to_numeric(df["y"], errors="raise")

    return df


def load_ie_predictions(path: Path) -> np.ndarray:
    ie_pred_df = pd.read_csv(path, header=0, names=["ie_2026"], low_memory=False)

    if ie_pred_df.empty:
        raise ValueError(f"El archivo de predicciones de IIE está vacío: {path}")

    values = pd.to_numeric(ie_pred_df["ie_2026"], errors="raise").to_numpy(dtype=float)
    return normalize_ie_predictions(values)


def build_reference_grid_map(ref_grid_dir: Path) -> dict[str, Path]:
    """
    Crea un mapa region -> ref_grid.tif usando el nombre del subdirectorio.
    """
    mapping: dict[str, Path] = {}

    for region_dir in sorted(ref_grid_dir.iterdir()):
        if not region_dir.is_dir():
            continue

        tif_path = region_dir / "ref_grid.tif"
        if not tif_path.exists():
            continue

        mapping[region_dir.name.strip().lower()] = tif_path

    if not mapping:
        raise ValueError(f"No se encontraron rasters de referencia en {ref_grid_dir}")

    return mapping


def find_reference_raster(region: str, mapping: dict[str, Path]) -> Path:
    region_key = str(region).strip().lower()

    if region_key not in mapping:
        raise FileNotFoundError(
            f"No encontré raster de referencia para la región '{region}'. "
            f"Disponibles: {list(mapping.keys())}"
        )

    return mapping[region_key]


def sort_regions_numerically(regions: list[str]) -> list[str]:
    return sorted(regions, key=lambda s: int(str(s).split("_")[-1]))


def build_source_grid_from_points(
        region_df: pd.DataFrame,
        ie_pred_values: np.ndarray,
        nodata_value: float = NODATA_VALUE,
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Construye una grilla raster en EPSG:4326 a partir de puntos x, y, z.
    La resolución se infiere del espaciamiento mínimo positivo de x e y.
    Si varias observaciones caen en la misma celda, promedia sus valores.
    """
    idx0 = region_df["idx"].to_numpy() - 1
    z = ie_pred_values[idx0]

    xs = region_df["x"].to_numpy(dtype=float)
    ys = region_df["y"].to_numpy(dtype=float)

    if len(xs) == 0:
        raise ValueError("No hay puntos para construir la grilla fuente.")

    unique_x = np.sort(np.unique(xs))
    unique_y = np.sort(np.unique(ys))

    if len(unique_x) < 2 or len(unique_y) < 2:
        raise ValueError("No hay suficientes coordenadas únicas para inferir resolución de grilla.")

    dx_candidates = np.diff(unique_x)
    dy_candidates = np.diff(unique_y)

    dx_candidates = dx_candidates[dx_candidates > 0]
    dy_candidates = dy_candidates[dy_candidates > 0]

    if len(dx_candidates) == 0 or len(dy_candidates) == 0:
        raise ValueError("No fue posible inferir resolución positiva de la grilla fuente.")

    dx = float(dx_candidates.min())
    dy = float(dy_candidates.min())

    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())

    width = int(round((xmax - xmin) / dx)) + 1
    height = int(round((ymax - ymin) / dy)) + 1

    transform = from_origin(
        west=xmin - dx / 2.0,
        north=ymax + dy / 2.0,
        xsize=dx,
        ysize=dy,
    )

    sums = np.zeros((height, width), dtype="float64")
    counts = np.zeros((height, width), dtype="uint32")

    rows, cols = rowcol(transform, xs, ys)

    for r, c, val in zip(rows, cols, z):
        if 0 <= r < height and 0 <= c < width:
            sums[r, c] += float(val)
            counts[r, c] += 1

    source_array = np.full((height, width), nodata_value, dtype="float32")
    mask = counts > 0
    source_array[mask] = (sums[mask] / counts[mask]).astype("float32")

    return source_array, transform


def warp_source_to_template(
        source_array: np.ndarray,
        source_transform,
        template_path: Path,
        output_path: Path,
        source_crs: str = SOURCE_CRS,
        nodata_value: float = NODATA_VALUE,
) -> tuple[int, float | None, float | None]:
    """
    Reproyecta una grilla fuente a la plantilla usando nearest neighbour.
    Devuelve cantidad de celdas válidas y rango de valores válidos.
    """
    with rasterio.open(template_path) as src:
        meta = src.meta.copy()
        template_transform = src.transform
        template_crs = src.crs
        height = src.height
        width = src.width

        if template_crs is None:
            raise ValueError(f"El raster plantilla no tiene CRS: {template_path}")

        dest_array = np.full((height, width), nodata_value, dtype="float32")

        reproject(
            source=source_array,
            destination=dest_array,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=nodata_value,
            dst_transform=template_transform,
            dst_crs=template_crs,
            dst_nodata=nodata_value,
            resampling=Resampling.nearest,
        )

        valid = dest_array[dest_array != nodata_value]
        valid_count = int(valid.size)
        min_valid = float(valid.min()) if valid_count > 0 else None
        max_valid = float(valid.max()) if valid_count > 0 else None

        meta.update(
            dtype="float32",
            count=1,
            nodata=nodata_value,
            compress="lzw",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            output_path.unlink()

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(dest_array, 1)

    return valid_count, min_valid, max_valid


def rasterize_points_to_template(
        region_df: pd.DataFrame,
        ie_pred_values: np.ndarray,
        template_path: Path,
        output_path: Path,
        source_crs: str = SOURCE_CRS,
        nodata_value: float = NODATA_VALUE,
) -> tuple[int, float | None, float | None]:
    """
    Flujo completo:
    1. arma raster fuente en CRS original de puntos
    2. lo reproyecta a la plantilla
    """
    source_array, source_transform = build_source_grid_from_points(
        region_df=region_df,
        ie_pred_values=ie_pred_values,
        nodata_value=nodata_value,
    )

    return warp_source_to_template(
        source_array=source_array,
        source_transform=source_transform,
        template_path=template_path,
        output_path=output_path,
        source_crs=source_crs,
        nodata_value=nodata_value,
    )


def save_histogram(
        values: np.ndarray,
        output_png: Path,
        title: str,
        xlabel: str = "IE normalizado",
        bins: int = 50,
) -> None:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def save_raster_histogram(
            raster_path: Path,
            output_png: Path,
            title: str,
            nodata_value: float = NODATA_VALUE,
            bins: int = 50,
    ) -> None:

    with rasterio.open(raster_path) as src:
            arr = src.read(1)

    valid = arr[np.isfinite(arr) & (arr != nodata_value)]
    if valid.size == 0:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=bins)
    plt.title(title)
    plt.xlabel("IE normalizado")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def main() -> None:
    df = load_training_table(TRAINING_TABLE)
    ie_pred_values = load_ie_predictions(IE_PREDICTIONS_CSV)
    ref_map = build_reference_grid_map(REF_GRID_DIR)

    if df["idx"].max() > len(ie_pred_values):
        raise ValueError(
            f"El máximo idx ({df['idx'].max()}) excede el número de predicciones IE ({len(ie_pred_values)})"
        )

    save_histogram(
        ie_pred_values,
        OUTPUT_DIR / "hist_ie_global.png",
        title="Histograma global de IE normalizado"
    )

    regiones = sort_regions_numerically(df["regionId"].dropna().unique().tolist())

    for region in regiones:
        region_df = df[df["regionId"] == region].copy()
        template_path = find_reference_raster(region, ref_map)
        output_path = OUTPUT_DIR / f"eicoastal_{region}.tif"

        valid_count, min_valid, max_valid = rasterize_points_to_template(
            region_df=region_df,
            ie_pred_values=ie_pred_values,
            template_path=template_path,
            output_path=output_path,
            source_crs=SOURCE_CRS,
            nodata_value=NODATA_VALUE,
        )

        save_raster_histogram(
            raster_path=output_path,
            output_png=OUTPUT_DIR / f"hist_eicoastal_{region}.png",
            title=f"Histograma IE {region}",
            nodata_value=NODATA_VALUE,
        )

        print(
            f"{region}: {valid_count} celdas válidas, "
            f"rango [{min_valid}, {max_valid}] -> {output_path}"
        )


if __name__ == "__main__":
    main()