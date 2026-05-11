# -*- coding: utf-8 -*-
"""
Construye un GeoPackage consolidado y un proyecto QGIS a partir de las salidas
del workflow headless de workflow-iie.

Objetivos:
- centralizar capas vectoriales/tabulares en un GPKG
- registrar rásteres y vectores en un proyecto QGIS .qgz
- dejar una base reproducible para producción cartográfica posterior

Este script sí usa qgis.core, pero solo en la fase final de ensamblaje.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from osgeo import ogr
from qgis.core import (
    QgsApplication,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsLayerTreeGroup,
)


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path("../../visualizar")
RESULTS_DIR = BASE_DIR / "results"

GPKG_PATH = RESULTS_DIR / "pkg" / "workflow_iie_outputs.gpkg"
QGIS_PROJECT_PATH = RESULTS_DIR / "qgis" / "workflow_iie.qgz"

QGIS_PREFIX_PATH = r"C:\QGis_env\Library"

# Capas esperadas
RASTER_LAYERS = {
    "zonas_vida": RESULTS_DIR / "rasters" / "zvh_mx3gw_regunidas_dunas.tif",
    "erosion": RESULTS_DIR / "rasters" / "TasasdeErosion2_regunidas.tif",
    "batimetria": RESULTS_DIR / "rasters" / "GEBCO_regionmarina_regunidas.tif",
}

VECTOR_LAYERS = {
    "wind_points": RESULTS_DIR / "vectors" / "reg_unidas_wind_points.gpkg",
}

TABLES = {
    "final_data": RESULTS_DIR / "tables" / "final_data_v9_3.csv",
}


# ---------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_optional_file(path: Path) -> bool:
    return path.exists()


def delete_gpkg_if_exists(path: Path) -> None:
    if path.exists():
        driver = ogr.GetDriverByName("GPKG")
        driver.DeleteDataSource(str(path))


def add_csv_to_gpkg(csv_path: Path, gpkg_path: Path, layer_name: str) -> None:
    df = pd.read_csv(csv_path)
    # TODO: Definir si