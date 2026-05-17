#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_ie_maps_to_reference.py

Comparación raster-raster de mapas de integridad ecosistémica (IE) contra una
referencia base observada, típicamente ei_qnint exportada como GeoTIFF regional.

Uso rutinario:
    candidate IE GeoTIFFs  vs  reference GeoTIFFs

Uso extendido opcional:
    baseline IE GeoTIFFs   vs  candidate IE GeoTIFFs
    baseline IE GeoTIFFs   vs  reference GeoTIFFs
    candidate IE GeoTIFFs  vs  reference GeoTIFFs

Notas:
- La comparación se hace pixel a pixel sobre GeoTIFFs, no por orden de CSV.
- Se exige misma forma, CRS y transform para que las métricas sean comparables.
- La referencia puede normalizarse, por ejemplo de escala 1..5 a 0..1.
- Los NoData se detectan antes de normalizar para evitar contaminar métricas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


DEFAULT_NODATA = -9999.0
DEFAULT_REGIONS = ",".join([f"region_{i}" for i in range(1, 15)])


def parse_range(spec: str | None) -> tuple[float, float] | None:
    if spec is None or str(spec).strip() == "":
        return None
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Rango inválido: {spec}. Use formato 'min,max', por ejemplo '1,5'.")
    lo, hi = map(float, parts)
    if hi == lo:
        raise ValueError("El rango de normalización no puede tener extremos iguales.")
    return lo, hi


def normalize_values(arr: np.ndarray, valid: np.ndarray, norm_range: tuple[float, float] | None) -> np.ndarray:
    out = arr.astype("float64", copy=True)
    if norm_range is None:
        return out
    lo, hi = norm_range
    out[valid] = (out[valid] - lo) / (hi - lo)
    return out


def read_raster(path: Path, band: int = 1, nodata_override: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float64")
        src_nodata = src.nodata
        nodata = nodata_override if nodata_override is not None else src_nodata
        if nodata is None:
            nodata = DEFAULT_NODATA
        meta = {
            "path": str(path),
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "height": src.height,
            "width": src.width,
            "nodata": nodata,
            "src_nodata": src_nodata,
            "dtype": src.dtypes[band - 1],
            "band": band,
        }
    return arr, meta


def valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask


def metadata_status(a_meta: dict[str, Any], b_meta: dict[str, Any]) -> dict[str, Any]:
    same_shape = (a_meta["height"] == b_meta["height"]) and (a_meta["width"] == b_meta["width"])
    same_crs = a_meta["crs"] == b_meta["crs"]
    same_transform = np.allclose(a_meta["transform"], b_meta["transform"], rtol=0, atol=1e-12)
    return {
        "same_shape": bool(same_shape),
        "same_crs": bool(same_crs),
        "same_transform": bool(same_transform),
        "a_height": a_meta["height"],
        "a_width": a_meta["width"],
        "b_height": b_meta["height"],
        "b_width": b_meta["width"],
    }


def pair_metrics(
    a_arr: np.ndarray,
    a_valid: np.ndarray,
    a_meta: dict[str, Any],
    b_arr: np.ndarray,
    b_valid: np.ndarray,
    b_meta: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    meta = metadata_status(a_meta, b_meta)

    row[f"{prefix}_same_shape"] = meta["same_shape"]
    row[f"{prefix}_same_crs"] = meta["same_crs"]
    row[f"{prefix}_same_transform"] = meta["same_transform"]

    if not (meta["same_shape"] and meta["same_crs"] and meta["same_transform"]):
        row.update({
            f"{prefix}_status": "grid_mismatch",
            f"{prefix}_n_a_valid": int(a_valid.sum()) if a_valid.shape == a_arr.shape else np.nan,
            f"{prefix}_n_b_valid": int(b_valid.sum()) if b_valid.shape == b_arr.shape else np.nan,
            f"{prefix}_n_common_valid": np.nan,
            f"{prefix}_cor": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_max_abs_diff": np.nan,
        })
        return row

    both = a_valid & b_valid
    only_a = a_valid & ~b_valid
    only_b = b_valid & ~a_valid

    row.update({
        f"{prefix}_n_a_valid": int(a_valid.sum()),
        f"{prefix}_n_b_valid": int(b_valid.sum()),
        f"{prefix}_n_common_valid": int(both.sum()),
        f"{prefix}_n_only_a": int(only_a.sum()),
        f"{prefix}_n_only_b": int(only_b.sum()),
    })

    if both.sum() == 0:
        row.update({
            f"{prefix}_status": "no_common_valid_pixels",
            f"{prefix}_a_min": np.nan,
            f"{prefix}_a_max": np.nan,
            f"{prefix}_b_min": np.nan,
            f"{prefix}_b_max": np.nan,
            f"{prefix}_cor": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_max_abs_diff": np.nan,
        })
        return row

    a_vals = a_arr[both]
    b_vals = b_arr[both]
    diff = a_vals - b_vals
    abs_diff = np.abs(diff)

    row.update({
        f"{prefix}_status": "ok",
        f"{prefix}_a_min": float(np.min(a_vals)),
        f"{prefix}_a_max": float(np.max(a_vals)),
        f"{prefix}_b_min": float(np.min(b_vals)),
        f"{prefix}_b_max": float(np.max(b_vals)),
        f"{prefix}_cor": float(np.corrcoef(a_vals, b_vals)[0, 1]) if both.sum() > 1 else np.nan,
        f"{prefix}_mae": float(np.mean(abs_diff)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}_max_abs_diff": float(np.max(abs_diff)),
    })
    return row


def sort_regions(regions: list[str]) -> list[str]:
    def key_fn(s: str):
        try:
            return int(str(s).split("_")[-1])
        except Exception:
            return str(s)
    return sorted(regions, key=key_fn)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara mapas IE contra referencia base raster, con comparación opcional entre dos productos IE."
    )
    parser.add_argument("--candidate-dir", required=True, help="Carpeta con GeoTIFFs del producto IE rutinario/candidato.")
    parser.add_argument("--reference-dir", required=True, help="Carpeta con GeoTIFFs de referencia base, por ejemplo condicion_dunas/ei_qnint.")
    parser.add_argument("--output", required=True, help="CSV de resumen de comparación.")

    parser.add_argument("--baseline-dir", default=None, help="Carpeta opcional con segundo producto IE, por ejemplo mapas de R.")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="Lista de regiones separada por comas.")

    parser.add_argument("--candidate-pattern", default="eicoastal_{region}.tif")
    parser.add_argument("--baseline-pattern", default="eicoastal_{region}.tif")
    parser.add_argument("--reference-pattern", default="{region}.tif")

    parser.add_argument("--candidate-band", type=int, default=1)
    parser.add_argument("--baseline-band", type=int, default=1)
    parser.add_argument("--reference-band", type=int, default=1)

    parser.add_argument("--candidate-nodata", type=float, default=None)
    parser.add_argument("--baseline-nodata", type=float, default=None)
    parser.add_argument("--reference-nodata", type=float, default=None)

    parser.add_argument("--candidate-normalize-from", default=None, help="Rango opcional para normalizar candidato, formato min,max.")
    parser.add_argument("--baseline-normalize-from", default=None, help="Rango opcional para normalizar baseline, formato min,max.")
    parser.add_argument("--reference-normalize-from", default="1,5", help="Rango para normalizar referencia base. Default: 1,5.")

    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--reference-label", default="reference")

    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    reference_dir = Path(args.reference_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    candidate_norm = parse_range(args.candidate_normalize_from)
    baseline_norm = parse_range(args.baseline_normalize_from)
    reference_norm = parse_range(args.reference_normalize_from)

    regions = sort_regions([r.strip() for r in args.regions.split(",") if r.strip()])
    rows: list[dict[str, Any]] = []

    for region in regions:
        row: dict[str, Any] = {"region": region}

        candidate_path = candidate_dir / args.candidate_pattern.format(region=region)
        reference_path = reference_dir / args.reference_pattern.format(region=region)
        baseline_path = baseline_dir / args.baseline_pattern.format(region=region) if baseline_dir else None

        row["candidate_path"] = str(candidate_path)
        row["reference_path"] = str(reference_path)
        if baseline_path is not None:
            row["baseline_path"] = str(baseline_path)

        missing = []
        if not candidate_path.exists():
            missing.append("candidate")
        if not reference_path.exists():
            missing.append("reference")
        if baseline_path is not None and not baseline_path.exists():
            missing.append("baseline")

        if missing:
            row["status"] = "missing_file"
            row["missing"] = ",".join(missing)
            rows.append(row)
            continue

        cand_arr_raw, cand_meta = read_raster(candidate_path, args.candidate_band, args.candidate_nodata)
        ref_arr_raw, ref_meta = read_raster(reference_path, args.reference_band, args.reference_nodata)

        cand_valid_raw = valid_mask(cand_arr_raw, cand_meta["nodata"])
        ref_valid_raw = valid_mask(ref_arr_raw, ref_meta["nodata"])

        cand_arr = normalize_values(cand_arr_raw, cand_valid_raw, candidate_norm)
        ref_arr = normalize_values(ref_arr_raw, ref_valid_raw, reference_norm)

        cand_valid = cand_valid_raw & np.isfinite(cand_arr)
        ref_valid = ref_valid_raw & np.isfinite(ref_arr)

        row["candidate_nodata"] = cand_meta["nodata"]
        row["reference_nodata"] = ref_meta["nodata"]
        row["reference_normalize_from"] = args.reference_normalize_from

        row.update(pair_metrics(
            cand_arr, cand_valid, cand_meta,
            ref_arr, ref_valid, ref_meta,
            prefix="candidate_vs_reference",
        ))

        if baseline_path is not None:
            base_arr_raw, base_meta = read_raster(baseline_path, args.baseline_band, args.baseline_nodata)
            base_valid_raw = valid_mask(base_arr_raw, base_meta["nodata"])
            base_arr = normalize_values(base_arr_raw, base_valid_raw, baseline_norm)
            base_valid = base_valid_raw & np.isfinite(base_arr)
            row["baseline_nodata"] = base_meta["nodata"]

            row.update(pair_metrics(
                base_arr, base_valid, base_meta,
                ref_arr, ref_valid, ref_meta,
                prefix="baseline_vs_reference",
            ))

            row.update(pair_metrics(
                cand_arr, cand_valid, cand_meta,
                base_arr, base_valid, base_meta,
                prefix="candidate_vs_baseline",
            ))

        row["status"] = "ok"
        rows.append(row)

    out = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    # Resumen compacto en consola.
    print(out[[c for c in out.columns if c in {
        "region", "status",
        "candidate_vs_reference_cor",
        "baseline_vs_reference_cor",
        "candidate_vs_baseline_cor",
        "candidate_vs_reference_mae",
        "baseline_vs_reference_mae",
        "candidate_vs_baseline_mae",
    }]])
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
