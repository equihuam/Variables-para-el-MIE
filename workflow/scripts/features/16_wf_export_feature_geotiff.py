#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 16_wf_export_feature_geotiff.py

Propósito:
    Exportar una colección de GeoTIFFs por variable/región a partir de los
    Parquet regionales en results/features/<variable>/.

Resumen del flujo:
    1. Leer cada Parquet regional de una variable.
    2. Identificar columnas temáticas distintas de regionid, pixid, x, y.
    3. Construir, para cada región, una grilla fuente en EPSG:4326 usando x/y.
    4. Reproyectar esa grilla fuente a la plantilla ref_grid.tif regional.
    5. Escribir un GeoTIFF por región. Si la feature tiene varias columnas
       temáticas, se escribe un GeoTIFF multibanda.
    6. Para columnas alfanuméricas, codificar etiquetas a enteros y exportar
       una tabla de códigos CSV.

Notas:
    - Las coordenadas x/y de las features se asumen en EPSG:4326, que es la
      convención operativa validada para las tablas regionales.
    - La salida hereda CRS, transform, shape y nodata del ref_grid.tif.
    - Se usa nearest neighbour para conservar clases discretas y valores ya
      calculados por pixel.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.warp import Resampling, reproject


BASE_COLUMNS = ["regionid", "pixid", "x", "y"]
SOURCE_CRS = "EPSG:4326"
NODATA_VALUE = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta features regionales Parquet a GeoTIFFs alineados a ref_grid.tif."
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        help="Directorio de la variable, por ejemplo results/features/corales.",
    )
    parser.add_argument(
        "--ref-grid-dir",
        required=True,
        help="Directorio raíz con subdirectorios region_*/ref_grid.tif.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se escribirán los GeoTIFFs regionales de la variable.",
    )
    parser.add_argument(
        "--feature-name",
        default=None,
        help="Nombre de la feature. Si se omite, se usa el nombre del feature-dir.",
    )
    parser.add_argument(
        "--source-crs",
        default=SOURCE_CRS,
        help="CRS de las coordenadas x/y de entrada. Default: EPSG:4326.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=NODATA_VALUE,
        help="Valor NoData para los GeoTIFFs de salida.",
    )
    parser.add_argument(
        "--categorical-columns",
        default=None,
        help="Columnas a forzar como categóricas, separadas por coma. Default: autodetección.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime diagnóstico por región.",
    )
    return parser.parse_args()


def parse_csv_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {v.strip() for v in value.split(",") if v.strip()}


def region_sort_key(path: Path) -> int:
    m = re.search(r"region_(\d+)", path.stem)
    if not m:
        return 10**9
    return int(m.group(1))


def validate_feature_dir(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el directorio de feature: {path}")
    if not path.is_dir():
        raise ValueError(f"La ruta no es un directorio: {path}")

    files = sorted(path.glob("region_*.parquet"), key=region_sort_key)
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos region_*.parquet en {path}")
    return files


def ref_grid_for_region(ref_grid_dir: Path, region: str) -> Path:
    path = ref_grid_dir / region / "ref_grid.tif"
    if not path.exists():
        raise FileNotFoundError(f"No existe ref_grid para {region}: {path}")
    return path


def read_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} no cumple el contrato mínimo. Faltan: {missing}")
    if df.empty:
        raise ValueError(f"La tabla está vacía: {path}")
    return df


def value_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in BASE_COLUMNS]
    if not cols:
        raise ValueError("La tabla no tiene columnas temáticas.")
    return cols


def can_parse_all_nonmissing_as_numeric(s: pd.Series) -> bool:
    nonmissing = s.dropna()
    if nonmissing.empty:
        return True
    parsed = pd.to_numeric(nonmissing, errors="coerce")
    return bool(parsed.notna().all())


def prepare_band_values(
        df: pd.DataFrame,
        col: str,
        force_categorical: bool,
        feature_name: str,
        region: str,
) -> tuple[np.ndarray, list[dict]]:
    """Devuelve valores float32 y filas de codebook si aplica."""
    s = df[col]

    # Si el usuario fuerza categórica, o si no puede parsearse como numérica,
    # se codifica a enteros 1..n conservando NA como nodata.
    is_numeric_like = can_parse_all_nonmissing_as_numeric(s)
    if (not force_categorical) and is_numeric_like:
        values = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
        return values.astype("float32"), []

    labels = s.astype("string")
    valid_labels = sorted(labels.dropna().unique().tolist())
    label_to_code = {label: i + 1 for i, label in enumerate(valid_labels)}

    coded = labels.map(label_to_code).astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
    codebook_rows = [
        {
            "feature": feature_name,
            "region": region,
            "column": col,
            "code": code,
            "label": label,
        }
        for label, code in label_to_code.items()
    ]
    return coded.astype("float32"), codebook_rows


def infer_source_grid_geometry(xs: np.ndarray, ys: np.ndarray) -> tuple[int, int, rasterio.Affine]:
    unique_x = np.sort(np.unique(xs))
    unique_y = np.sort(np.unique(ys))

    if len(unique_x) < 2 or len(unique_y) < 2:
        raise ValueError("No hay suficientes coordenadas únicas para inferir resolución.")

    dxs = np.diff(unique_x)
    dys = np.diff(unique_y)
    dxs = dxs[dxs > 0]
    dys = dys[dys > 0]

    if dxs.size == 0 or dys.size == 0:
        raise ValueError("No fue posible inferir resolución positiva.")

    dx = float(dxs.min())
    dy = float(dys.min())

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
    return height, width, transform


def build_source_array(
        xs: np.ndarray,
        ys: np.ndarray,
        values: np.ndarray,
        nodata: float,
) -> tuple[np.ndarray, rasterio.Affine]:
    height, width, transform = infer_source_grid_geometry(xs, ys)

    sums = np.zeros((height, width), dtype="float64")
    counts = np.zeros((height, width), dtype="uint32")

    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    finite_values = np.isfinite(values)
    valid = (
        finite_values
        & (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )

    np.add.at(sums, (rows[valid], cols[valid]), values[valid])
    np.add.at(counts, (rows[valid], cols[valid]), 1)

    arr = np.full((height, width), nodata, dtype="float32")
    mask = counts > 0
    arr[mask] = (sums[mask] / counts[mask]).astype("float32")
    return arr, transform


def warp_band_to_template(
        source_array: np.ndarray,
        source_transform,
        template_path: Path,
        source_crs: str,
        nodata: float,
) -> tuple[np.ndarray, dict]:
    with rasterio.open(template_path) as tmpl:
        if tmpl.crs is None:
            raise ValueError(f"La plantilla no tiene CRS: {template_path}")

        dest = np.full((tmpl.height, tmpl.width), nodata, dtype="float32")
        reproject(
            source=source_array,
            destination=dest,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=nodata,
            dst_transform=tmpl.transform,
            dst_crs=tmpl.crs,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )

        profile = tmpl.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            nodata=nodata,
            compress="lzw",
        )
        return dest, profile


def export_region_geotiff(
        df: pd.DataFrame,
        feature_name: str,
        region: str,
        template_path: Path,
        output_path: Path,
        forced_cats: set[str],
        source_crs: str,
        nodata: float,
        verbose: bool,
) -> list[dict]:
    cols = value_columns(df)
    xs = pd.to_numeric(df["x"], errors="raise").to_numpy(dtype="float64")
    ys = pd.to_numeric(df["y"], errors="raise").to_numpy(dtype="float64")

    bands: list[np.ndarray] = []
    codebook_rows: list[dict] = []

    for col in cols:
        values, rows = prepare_band_values(
            df=df,
            col=col,
            force_categorical=(col in forced_cats),
            feature_name=feature_name,
            region=region,
        )
        source_array, source_transform = build_source_array(xs, ys, values, nodata=nodata)
        warped, profile = warp_band_to_template(
            source_array=source_array,
            source_transform=source_transform,
            template_path=template_path,
            source_crs=source_crs,
            nodata=nodata,
        )
        bands.append(warped)
        codebook_rows.extend(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=len(bands))

    if output_path.exists():
        output_path.unlink()

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, (col, band) in enumerate(zip(cols, bands), start=1):
            dst.write(band.astype("float32"), i)
            dst.set_band_description(i, col)
            valid = band[np.isfinite(band) & (band != nodata)]
            if valid.size:
                dst.update_tags(i, min=float(valid.min()), max=float(valid.max()))

    if verbose:
        valid_counts = [int(np.sum(np.isfinite(b) & (b != nodata))) for b in bands]
        print(f"{region}: {output_path.name}; bandas={cols}; valid={valid_counts}")

    return codebook_rows


def main() -> None:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    ref_grid_dir = Path(args.ref_grid_dir)
    output_dir = Path(args.output_dir)
    feature_name = args.feature_name or feature_dir.name
    forced_cats = parse_csv_list(args.categorical_columns)

    files = validate_feature_dir(feature_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_codebook_rows: list[dict] = []

    for fp in files:
        region = fp.stem
        df = read_feature_table(fp)
        template_path = ref_grid_for_region(ref_grid_dir, region)
        output_path = output_dir / f"{region}.tif"

        rows = export_region_geotiff(
            df=df,
            feature_name=feature_name,
            region=region,
            template_path=template_path,
            output_path=output_path,
            forced_cats=forced_cats,
            source_crs=args.source_crs,
            nodata=args.nodata,
            verbose=args.verbose,
        )
        all_codebook_rows.extend(rows)

    if all_codebook_rows:
        codebook = pd.DataFrame(all_codebook_rows)
        codebook_path = output_dir / "_codebook.csv"
        codebook.to_csv(codebook_path, index=False)
        if args.verbose:
            print(f"codebook -> {codebook_path}")

    print(f"OK -> {output_dir}")


if __name__ == "__main__":
    main()
