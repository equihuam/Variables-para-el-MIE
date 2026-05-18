# -*- coding: utf-8 -*-
"""
Reproyecta y recorta un raster de batimetría GEBCO usando una máscara vectorial
(región marina), y lo alinea exactamente a la malla de reg_unidas.

Versión robusta para ejecución no supervisada, sin GUI ni dependencias de QGIS Desktop.
"""

import os
import sys
from osgeo import gdal, ogr

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

GEBCO_SRC = r"C:/sig_costas_/batimetria/01_GEBCO2020_SIMAR.tif"
SHP_MASK = r"C:/sig_costas_/region_marina/regionmarinamx.shp"
REG_UNIDAS_PATH = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

OUT_DIR = r"C:/sig_costas_/batimetria"
GEBCO_FINAL = os.path.join(OUT_DIR, "GEBCO_regionmarina_regunidas.tif")

NODATA_VALUE = -9999.0
OUTPUT_TYPE = gdal.GDT_Float32
RESAMPLE_ALG = "bilinear"   # variable continua


# ---------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def validate_input_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        fail(f"No existe el archivo de entrada ({label}): {path}")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        fail(f"No se pudo crear o acceder al directorio de salida: {path}")


def dataset_info(ds: gdal.Dataset, label: str) -> dict:
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * xsize
    ymin = ymax + gt[5] * ysize

    return {
        "label": label,
        "geotransform": gt,
        "projection": proj,
        "xsize": xsize,
        "ysize": ysize,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "pixel_width": gt[1],
        "pixel_height": gt[5],
    }


def print_dataset_info(info: dict) -> None:
    log(f"\n--- {info['label']} ---")
    log(f"Dimensiones: {info['xsize']} x {info['ysize']}")
    log(f"Resolución: {info['pixel_width']} x {abs(info['pixel_height'])}")
    log(f"Extensión : xmin={info['xmin']}, ymin={info['ymin']}, xmax={info['xmax']}, ymax={info['ymax']}")
    log("CRS:")
    log(info["projection"] if info["projection"] else "  [vacío]")


def validate_vector(path: str) -> None:
    ds = ogr.Open(path)
    if ds is None:
        fail(f"No se pudo abrir el vector de máscara: {path}")
    layer = ds.GetLayer()
    if layer is None:
        ds = None
        fail(f"No se pudo acceder a la capa del vector de máscara: {path}")
    feature_count = layer.GetFeatureCount()
    ds = None
    if feature_count == 0:
        fail(f"El vector de máscara no contiene geometrías: {path}")


def remove_file_safely(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        log(f"No se pudo eliminar {path}: {e}")


# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def warp_bathymetry_to_template(
    src_raster: str,
    cutline_vector: str,
    dst_raster: str,
    template_info: dict,
    nodata_value: float = NODATA_VALUE,
    resample_alg: str = RESAMPLE_ALG,
) -> None:
    log("\nReproyectando GEBCO, recortando a región marina y alineando a reg_unidas...")

    out_ds = gdal.Warp(
        destNameOrDestDS=dst_raster,
        srcDSOrSrcDSTab=src_raster,
        dstSRS=template_info["projection"],
        xRes=template_info["pixel_width"],
        yRes=abs(template_info["pixel_height"]),
        outputBounds=(
            template_info["xmin"],
            template_info["ymin"],
            template_info["xmax"],
            template_info["ymax"],
        ),
        cutlineDSName=cutline_vector,
        cropToCutline=True,
        dstNodata=nodata_value,
        outputType=OUTPUT_TYPE,
        resampleAlg=resample_alg,
        targetAlignedPixels=False,
        creationOptions=["COMPRESS=LZW"]
    )

    if out_ds is None:
        fail("gdal.Warp falló al generar el raster final de batimetría.")

    out_ds = None

    if not os.path.isfile(dst_raster):
        fail(f"No se generó el raster de salida esperado: {dst_raster}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:
    gdal.UseExceptions()

    log("Validando entradas...")
    validate_input_file(GEBCO_SRC, "raster de batimetría")
    validate_input_file(SHP_MASK, "shapefile de máscara")
    validate_input_file(REG_UNIDAS_PATH, "raster plantilla")
    validate_vector(SHP_MASK)
    ensure_output_dir(OUT_DIR)

    remove_file_safely(GEBCO_FINAL)

    reg_ds = gdal.Open(REG_UNIDAS_PATH)
    if reg_ds is None:
        fail(f"No se pudo abrir reg_unidas: {REG_UNIDAS_PATH}")

    reg_info = dataset_info(reg_ds, "reg_unidas")
    print_dataset_info(reg_info)
    reg_ds = None

    warp_bathymetry_to_template(
        src_raster=GEBCO_SRC,
        cutline_vector=SHP_MASK,
        dst_raster=GEBCO_FINAL,
        template_info=reg_info,
        nodata_value=NODATA_VALUE,
        resample_alg=RESAMPLE_ALG,
    )

    log("\n------------------------------------------")
    log("Proceso completado.")
    log("Raster resultante (batimetría marina alineada a reg_unidas):")
    log(f"  {GEBCO_FINAL}")
    log("------------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)