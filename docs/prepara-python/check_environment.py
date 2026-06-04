#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_environment.py

Verificación mínima del entorno para workflow-iie.

Qué valida:
- imports clave del stack numérico, tabular y geoespacial
- lectura/escritura básica de Parquet
- escritura/lectura mínima de raster GeoTIFF
- creación simple de geometrías con GeoPandas/Shapely
- disponibilidad de Snakemake y pytest
- detección opcional de PyQGIS

Uso:
    python check_environment.py
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "pyarrow",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
    "pyogrio",
    "scipy",
    "sklearn",
    "matplotlib",
    "snakemake",
    "pytest",
]

OPTIONAL_MODULES = [
    "pgmpy",
]


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_imports() -> bool:
    print("\n== Verificación de imports ==")
    success = True

    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
            ok(f"Import correcto: {mod}")
        except Exception as e:
            success = False
            fail(f"No se pudo importar {mod}: {e}")

    for mod in OPTIONAL_MODULES:
        try:
            importlib.import_module(mod)
            ok(f"Import opcional disponible: {mod}")
        except Exception as e:
            warn(f"Módulo opcional no disponible {mod}: {e}")

    try:
        importlib.import_module("qgis.core")
        ok("PyQGIS disponible")
    except Exception as e:
        warn(f"PyQGIS no disponible en este intérprete: {e}")

    return success


def check_parquet() -> bool:
    print("\n== Verificación de Parquet ==")
    try:
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df = pd.DataFrame(
                {
                    "regionid": ["region_1", "region_1", "region_2"],
                    "pixid": [1, 2, 1],
                    "x": [1.0, 2.0, 3.0],
                    "y": [4.0, 5.0, 6.0],
                    "value": [0.1, 0.2, 0.3],
                }
            )
            df.to_parquet(path, index=False)
            df2 = pd.read_parquet(path)

            assert len(df2) == 3
            assert list(df2.columns) == ["regionid", "pixid", "x", "y", "value"]

        ok("Lectura/escritura Parquet correcta")
        return True
    except Exception as e:
        fail(f"Prueba de Parquet falló: {e}")
        return False


def check_rasterio() -> bool:
    print("\n== Verificación de rasterio ==")
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tif"
            arr = np.array([[1, 2], [3, 4]], dtype="float32")
            transform = from_origin(0, 2, 1, 1)

            meta = {
                "driver": "GTiff",
                "height": arr.shape[0],
                "width": arr.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:4326",
                "transform": transform,
                "nodata": -9999.0,
            }

            with rasterio.open(path, "w", **meta) as dst:
                dst.write(arr, 1)

            with rasterio.open(path) as src:
                out = src.read(1)
                assert out.shape == (2, 2)
                assert str(src.crs) == "EPSG:4326"

        ok("Lectura/escritura raster mínima correcta")
        return True
    except Exception as e:
        fail(f"Prueba de rasterio falló: {e}")
        return False


def check_geopandas() -> bool:
    print("\n== Verificación de geopandas/shapely ==")
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )

        gdf2 = gdf.to_crs("EPSG:3857")
        assert len(gdf2) == 2
        assert gdf2.crs is not None

        ok("Creación y reproyección vectorial mínima correcta")
        return True
    except Exception as e:
        fail(f"Prueba de geopandas/shapely falló: {e}")
        return False


def check_versions() -> bool:
    print("\n== Resumen de versiones ==")
    modules = [
        "numpy",
        "pandas",
        "pyarrow",
        "rasterio",
        "geopandas",
        "shapely",
        "pyproj",
        "scipy",
        "sklearn",
        "matplotlib",
        "snakemake",
        "pytest",
    ]
    try:
        for mod in modules:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "sin __version__")
            print(f" - {mod}: {version}")
        print(f" - python: {sys.version.split()[0]}")
        ok("Resumen de versiones generado")
        return True
    except Exception as e:
        fail(f"No se pudo generar el resumen de versiones: {e}")
        return False


def main() -> int:
    print("Verificación del entorno para workflow-iie")
    print("=" * 50)

    checks = [
        check_imports(),
        check_parquet(),
        check_rasterio(),
        check_geopandas(),
        check_versions(),
    ]

    if all(checks):
        print("\nResultado final: ENTORNO OK")
        return 0

    print("\nResultado final: HAY PROBLEMAS EN EL ENTORNO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
