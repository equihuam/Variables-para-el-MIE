#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 13_wf_z_v_holdridge.py

Propósito:
    Clasificar, para una región específica, la zona de vida Holdridge más
    cercana para cada píxel válido del raster de referencia y exportar el
    resultado como tabla congruente por píxel en formato Parquet.

Criterio canónico:
    `zvh` es una variable cualitativa. Si existe una tabla VAT/DBF asociada
    al raster, por ejemplo `zvh_mx3gw.tif.vat.dbf`, se usa para traducir los
    códigos raster a etiquetas descriptivas. Si no existe o no puede leerse,
    se intenta leer CategoryNames vía GDAL. Como último recurso se conservan
    los códigos como texto.

Lógica espacial validada:
    Reproyecta ref_grid y raster ZVH al CRS de manglares, recorta ZVH a la
    región, convierte las celdas categóricas a puntos de entrenamiento y
    clasifica por 1-NN categórico emulando kknn(scale=TRUE):

      kknn(layer ~ x + y, zvh_points, region_points,
           distance = 2, k = 1, kernel = "rectangular")
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"Neither gdal\.UseExceptions\(\) nor gdal\.DontUseExceptions\(\).*",
    category=FutureWarning,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from sklearn.neighbors import NearestNeighbors

OUTPUT_FIELD = "zvh"
K_NEIGHBORS = 1

CODE_COLUMN_CANDIDATES = [
    "VALUE", "Value", "value", "VAL", "val", "gridcode", "GRIDCODE", "grid_code",
    "GRID_CODE", "Class", "CLASS", "class", "ID", "Id", "id",
]
COUNT_COLUMN_CANDIDATES = {"COUNT", "Count", "count", "AREA", "Area", "area"}
LABEL_COLUMN_HINTS = [
    "label", "LABEL", "Label",
    "name", "NAME", "Name", "nombre", "NOMBRE", "Nombre",
    "desc", "DESC", "Desc", "descrip", "DESCRIP", "Descripcion", "DESCRIPCIO",
    "zona", "ZONA", "zone", "ZONE",
    "vida", "VIDA", "life", "LIFE",
    "clase", "CLASE", "class_name", "CLASS_NAME",
    "formacion", "FORMACION", "ecosistema", "ECOSISTEMA",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica zona de vida Holdridge por pixel para una región específica."
    )
    parser.add_argument("--mangroves-shp", required=True, help="Shapefile usado para definir CRS de trabajo.")
    parser.add_argument("--zvh-raster", required=True, help="Raster de zonas de vida Holdridge.")
    parser.add_argument("--ref-grid", required=True, help="ref_grid.tif regional.")
    parser.add_argument("--region-id", required=True, help="Identificador regional, por ejemplo region_1.")
    parser.add_argument("--output", required=True, help="Salida .parquet.")
    parser.add_argument(
        "--zvh-vat-dbf",
        default=None,
        help="Tabla VAT/DBF opcional con códigos y etiquetas. Si se omite, se buscan sidecars *.vat.dbf.",
    )
    parser.add_argument(
        "--distance-mode",
        choices=["kknn_scaled", "raw"],
        default="kknn_scaled",
        help="Modo de distancia para 1-NN. kknn_scaled emula scale=TRUE de kknn.",
    )
    parser.add_argument("--debug-grid-output", default=None, help="CSV opcional con la grilla regional reproyectada.")
    parser.add_argument("--debug-zvh-points-output", default=None, help="CSV opcional con puntos de entrenamiento ZVH recortados.")
    parser.add_argument("--debug-metadata-output", default=None, help="CSV opcional con metadatos de la corrida.")
    parser.add_argument("--debug-codebook-output", default=None, help="CSV opcional con el diccionario código -> etiqueta usado.")
    parser.add_argument("--verbose", action="store_true", help="Imprime diagnósticos.")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_target_crs(path: Path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"El shapefile está vacío: {path}")
    if gdf.crs is None:
        raise ValueError(f"El shapefile no tiene CRS: {path}")
    return gdf.crs


def read_category_names(path: Path) -> dict[int, str]:
    """Intenta leer nombres categóricos GDAL; si no existen, regresa {}."""
    mapping_out: dict[int, str] = {}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Neither gdal\.UseExceptions\(\) nor gdal\.DontUseExceptions\(\).*",
                category=FutureWarning,
            )
            from osgeo import gdal  # type: ignore
            try:
                gdal.UseExceptions()
            except Exception:
                pass

        ds = gdal.Open(str(path))
        if ds is None:
            return {}
        band = ds.GetRasterBand(1)
        cats = band.GetCategoryNames()
        if cats:
            for i, cat in enumerate(cats):
                if cat is not None and str(cat).strip() != "":
                    mapping_out[int(i)] = str(cat).strip()
        ds = None
    except Exception:
        return {}
    return mapping_out


def candidate_vat_paths(raster_path: Path, explicit_path: str | None = None) -> list[Path]:
    if explicit_path:
        return [Path(explicit_path)]
    return [
        Path(str(raster_path) + ".vat.dbf"),       # zvh_mx3gw.tif.vat.dbf
        raster_path.with_suffix(raster_path.suffix + ".vat.dbf"),
        raster_path.with_suffix(".vat.dbf"),       # zvh_mx3gw.vat.dbf
        raster_path.parent / f"{raster_path.stem}.vat.dbf",
        raster_path.parent / f"{raster_path.name}.VAT.DBF",
        raster_path.parent / f"{raster_path.stem}.VAT.DBF",
    ]


def read_dbf_table(path: Path) -> pd.DataFrame:
    """Lee una DBF standalone con pyogrio/geopandas/fiona según disponibilidad."""
    last_error: Exception | None = None

    try:
        import pyogrio  # type: ignore
        df = pyogrio.read_dataframe(path)
        if "geometry" in df.columns:
            df = pd.DataFrame(df.drop(columns=["geometry"]))
        return pd.DataFrame(df)
    except Exception as exc:
        last_error = exc

    try:
        df = gpd.read_file(path)
        if "geometry" in df.columns:
            df = pd.DataFrame(df.drop(columns=["geometry"]))
        return pd.DataFrame(df)
    except Exception as exc:
        last_error = exc

    try:
        import fiona  # type: ignore
        records: list[dict[str, Any]] = []
        with fiona.open(path) as src:
            for feat in src:
                records.append(dict(feat.get("properties") or {}))
        return pd.DataFrame(records)
    except Exception as exc:
        last_error = exc

    raise RuntimeError(f"No pude leer la tabla DBF {path}: {last_error}")


def choose_code_column(df: pd.DataFrame) -> str:
    for col in CODE_COLUMN_CANDIDATES:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col

    numeric_candidates = []
    for col in df.columns:
        if col in COUNT_COLUMN_CANDIDATES:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            numeric_candidates.append(col)

    if not numeric_candidates:
        raise ValueError(f"No pude identificar columna de código en VAT. Columnas: {df.columns.tolist()}")
    return numeric_candidates[0]


def choose_label_column(df: pd.DataFrame, code_col: str) -> str | None:
    excluded = {code_col, *COUNT_COLUMN_CANDIDATES}
    cols = [c for c in df.columns if c not in excluded]

    # 1) Preferir nombres semánticos.
    for hint in LABEL_COLUMN_HINTS:
        for col in cols:
            if hint.lower() in col.lower():
                s = df[col].dropna().astype(str).str.strip()
                if not s.empty and (s != "").any():
                    return col

    # 2) Cualquier columna textual no vacía, prefiriendo la de mayor longitud media.
    text_candidates: list[tuple[float, str]] = []
    for col in cols:
        s = df[col].dropna().astype(str).str.strip()
        if s.empty or not (s != "").any():
            continue
        numeric_ratio = pd.to_numeric(s, errors="coerce").notna().mean()
        if numeric_ratio < 0.8:
            text_candidates.append((float(s.str.len().mean()), col))

    if text_candidates:
        return sorted(text_candidates, reverse=True)[0][1]

    return None


def load_vat_codebook(raster_path: Path, explicit_vat_path: str | None = None) -> tuple[dict[int, str], pd.DataFrame, Path | None]:
    """Lee código -> etiqueta desde VAT/DBF; si no hay etiqueta descriptiva, regresa {}."""
    for vat_path in dict.fromkeys(candidate_vat_paths(raster_path, explicit_vat_path)):
        if not vat_path.exists():
            continue
        df = read_dbf_table(vat_path)
        if df.empty:
            continue
        code_col = choose_code_column(df)
        label_col = choose_label_column(df, code_col)
        if label_col is None:
            continue

        codes = pd.to_numeric(df[code_col], errors="coerce")
        labels = df[label_col].astype("string")
        codebook = pd.DataFrame({
            "code": codes,
            "label": labels,
            "source": str(vat_path),
            "code_column": code_col,
            "label_column": label_col,
        }).dropna(subset=["code", "label"])
        codebook["label"] = codebook["label"].astype(str).str.strip()
        codebook = codebook[codebook["label"] != ""]
        if codebook.empty:
            continue
        codebook["code"] = codebook["code"].round().astype(int)
        mapping_out = dict(zip(codebook["code"], codebook["label"]))
        return mapping_out, codebook.drop_duplicates("code"), vat_path

    return {}, pd.DataFrame(columns=["code", "label", "source", "code_column", "label_column"]), None


def build_category_map(raster_path: Path, explicit_vat_path: str | None) -> tuple[dict[int, str], pd.DataFrame, str]:
    vat_map, vat_codebook, vat_path = load_vat_codebook(raster_path, explicit_vat_path)
    if vat_map:
        return vat_map, vat_codebook, f"vat_dbf:{vat_path}"

    gdal_map = read_category_names(raster_path)
    if gdal_map:
        codebook = pd.DataFrame({
            "code": list(gdal_map.keys()),
            "label": list(gdal_map.values()),
            "source": "gdal_category_names",
            "code_column": "GDAL_index",
            "label_column": "CategoryNames",
        })
        return gdal_map, codebook, "gdal_category_names"

    return {}, pd.DataFrame(columns=["code", "label", "source", "code_column", "label_column"]), "code_as_text"


def reproject_raster_to_crs(src: rasterio.io.DatasetReader, dst_crs) -> tuple[np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    dst = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return dst, transform


def crop_array_to_region_bbox(arr: np.ndarray, transform, crs, region_arr: np.ndarray, region_transform) -> tuple[np.ndarray, rasterio.Affine]:
    height, width = region_arr.shape
    left, top = region_transform * (0, 0)
    right, bottom = region_transform * (width, height)
    geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as src:
            src.write(arr.astype(np.float32), 1)
            cropped, cropped_transform = mask(src, [mapping(geom)], crop=True, filled=False)
            out = cropped[0].astype("float64")
            out = np.where(np.ma.getmaskarray(out), np.nan, np.asarray(out, dtype=float))
    return out, cropped_transform


def valid_raster_points_dataframe(arr: np.ndarray, transform, value_col: str = "value") -> pd.DataFrame:
    valid_mask = np.isfinite(arr)
    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", value_col])
    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame({
        "row": rows.astype(int),
        "col": cols.astype(int),
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        value_col: arr[rows, cols],
    })


def scale_xy(train_xy: np.ndarray, pred_xy: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "raw":
        return train_xy, pred_xy, np.array([1.0, 1.0], dtype=float)
    sd = np.nanstd(train_xy, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return train_xy / sd, pred_xy / sd, sd


def predict_1nn_labels(train_df: pd.DataFrame, pred_df: pd.DataFrame, distance_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if train_df.empty:
        return np.full(len(pred_df), None, dtype=object), np.array([np.nan, np.nan])
    train_xy = train_df[["x", "y"]].to_numpy(dtype=float)
    pred_xy = pred_df[["x", "y"]].to_numpy(dtype=float)
    train_scaled, pred_scaled, sd = scale_xy(train_xy, pred_xy, distance_mode)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(train_scaled)
    _, idx = nn.kneighbors(pred_scaled, return_distance=True)
    labels = train_df["label"].to_numpy(dtype=object)[idx[:, 0]]
    return labels.astype(object), sd


def label_from_code(value: Any, cat_map: dict[int, str]) -> str:
    if pd.isna(value):
        return ""
    code_float = float(value)
    code_int = int(round(code_float))
    if abs(code_float - code_int) < 1e-6 and code_int in cat_map:
        return str(cat_map[code_int])
    if abs(code_float - code_int) < 1e-6:
        return str(code_int)
    return str(code_float)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Extensión no soportada: {path.suffix}")


def main() -> None:
    args = parse_args()
    mangroves_path = Path(args.mangroves_shp)
    zvh_path = Path(args.zvh_raster)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(mangroves_path, zvh_path, ref_grid_path)
    target_crs = load_target_crs(mangroves_path)
    cat_map, codebook, category_source = build_category_map(zvh_path, args.zvh_vat_dbf)

    with rasterio.open(ref_grid_path) as ref_src:
        if ref_src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
        ref_masked = ref_src.read(1, masked=True)
        original_valid = int((~np.ma.getmaskarray(ref_masked) & np.isfinite(np.asarray(ref_masked, dtype=float))).sum())
        region_arr, region_transform = reproject_raster_to_crs(ref_src, target_crs)

    region_points = valid_raster_points_dataframe(region_arr, region_transform, value_col="ref_value")
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    with rasterio.open(zvh_path) as zvh_src:
        if zvh_src.crs is None:
            raise ValueError("El raster ZVH no tiene CRS definido.")
        zvh_arr, zvh_transform = reproject_raster_to_crs(zvh_src, target_crs)
        zvh_region_arr, zvh_region_transform = crop_array_to_region_bbox(
            zvh_arr, zvh_transform, target_crs, region_arr, region_transform
        )

    zvh_points = valid_raster_points_dataframe(zvh_region_arr, zvh_region_transform, value_col="layer")
    zvh_points["label"] = [label_from_code(v, cat_map) for v in zvh_points["layer"]]

    predictions, sd = predict_1nn_labels(zvh_points, region_points, args.distance_mode)

    out = pd.DataFrame({
        "regionid": region_points["regionid"].to_numpy(),
        "pixid": region_points["pixid"].to_numpy(),
        "x": region_points["x"].to_numpy(),
        "y": region_points["y"].to_numpy(),
        OUTPUT_FIELD: pd.Series(predictions, dtype="string"),
    })

    if args.debug_grid_output:
        save_table(region_points[["regionid", "pixid", "row", "col", "x", "y", "ref_value"]], Path(args.debug_grid_output))
    if args.debug_zvh_points_output:
        save_table(zvh_points[["row", "col", "x", "y", "layer", "label"]], Path(args.debug_zvh_points_output))
    if args.debug_codebook_output:
        if codebook.empty:
            observed_codes = sorted({int(round(float(v))) for v in zvh_points["layer"].dropna().unique()})
            codebook_out = pd.DataFrame({
                "code": observed_codes,
                "label": [str(c) for c in observed_codes],
                "source": category_source,
                "code_column": "layer",
                "label_column": "layer",
            })
        else:
            codebook_out = codebook.copy()
        save_table(codebook_out, Path(args.debug_codebook_output))
    if args.debug_metadata_output:
        vc = pd.Series(predictions).value_counts(dropna=False).to_dict()
        meta = {
            "regionid": region_id,
            "target_crs": str(target_crs),
            "original_valid_points": original_valid,
            "reprojected_valid_points": int(len(region_points)),
            "zvh_training_points": int(len(zvh_points)),
            "n_labels_training": int(zvh_points["label"].nunique(dropna=True)) if len(zvh_points) else 0,
            "distance_mode": args.distance_mode,
            "sd_x": float(sd[0]) if len(sd) else np.nan,
            "sd_y": float(sd[1]) if len(sd) else np.nan,
            "category_source": category_source,
            "category_map_found": bool(cat_map),
            "prediction_counts_json": json.dumps({str(k): int(v) for k, v in vc.items()}, ensure_ascii=False),
        }
        save_table(pd.DataFrame([meta]), Path(args.debug_metadata_output))

    log(f"puntos válidos originales: {original_valid}", args.verbose)
    log(f"puntos válidos reproyectados: {len(region_points)}", args.verbose)
    log(f"puntos entrenamiento ZVH: {len(zvh_points)}", args.verbose)
    log(f"fuente categorías: {category_source}", args.verbose)
    log(f"etiquetas entrenamiento: {sorted(zvh_points['label'].dropna().unique().tolist())[:20]}", args.verbose)
    log(f"modo distancia: {args.distance_mode}; sd=({sd[0]}, {sd[1]})", args.verbose)

    save_table(out, output_path)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
