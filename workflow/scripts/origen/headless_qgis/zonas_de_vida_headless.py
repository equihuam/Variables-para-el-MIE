# -*- coding: utf-8 -*-
"""
Alinea un raster categórico de zonas de vida a la malla de un raster plantilla
y aplica una máscara basada en reg_unidas > 0.

Versión robusta para ejecución no supervisada en qgis_env Python,
sin dependencia de QGIS ni de interfaz gráfica.

Requisitos:
- osgeo.gdal
- osgeo_utils.gdal_calc

Supuestos:
- reg_unidas define extensión, resolución y CRS de salida
- zonas de vida es categórico
- 0 se usa como NoData; no corresponde a ninguna clase de zona de vida.
"""

import os
import sys
from osgeo import gdal
from osgeo_utils.gdal_calc import Calc

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

REG_RASTER_PATH = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"
ZVH_SRC_PATH = r"C:/sig_costas_/zonas_vida_h/07_zvh_mx3gw/zvh_mx3gw.tif"

OUT_DIR = r"C:/sig_costas_/zonas_vida_h/07_zvh_mx3gw"
TMP_FILENAME = "zvh_mx3gw_regunidas_tmp.tif"
OUT_FILENAME = "zvh_mx3gw_regunidas_dunas.tif"

NODATA_VALUE = 0 # 0 no corresponde a ninguna clase de zona de vida;.
DEFAULT_SRC_SRS = "EPSG:4326"
DELETE_TEMP = True

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
    log(f"Resolución: {info['pixel_width']} x {info['pixel_height']}")
    log(f"Extensión : xmin={info['xmin']}, ymin={info['ymin']}, xmax={info['xmax']}, ymax={info['ymax']}")
    log("CRS:")
    log(info["projection"] if info["projection"] else "  [vacío]")


def remove_file_safely(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
            log(f"Archivo eliminado: {path}")
    except OSError as e:
        log(f"No se pudo eliminar el archivo {path}: {e}")

# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def warp_to_template(
    src_path: str,
    template_info: dict,
    dst_path: str,
    default_src_srs: str = DEFAULT_SRC_SRS,
    nodata_value: int = NODATA_VALUE,
) -> None:
    src_ds = gdal.Open(src_path)
    if src_ds is None:
        fail(f"No se pudo abrir el ráster de origen: {src_path}")

    src_proj = src_ds.GetProjection()
    if not src_proj or not src_proj.strip():
        log(f"Advertencia: el ráster de origen no tiene CRS; se asume {default_src_srs}.")
        src_proj = default_src_srs

    log("\nReproyectando/alineando ráster fuente a la malla de reg_unidas...")

    out_ds = gdal.Warp(
        destNameOrDestDS=dst_path,
        srcDSOrSrcDSTab=src_path,
        srcSRS=src_proj,
        dstSRS=template_info["projection"],
        xRes=template_info["pixel_width"],
        yRes=abs(template_info["pixel_height"]),
        outputBounds=(
            template_info["xmin"],
            template_info["ymin"],
            template_info["xmax"],
            template_info["ymax"],
        ),
        resampleAlg="near",
        dstNodata=nodata_value,
        outputType=gdal.GDT_Byte,
        targetAlignedPixels=False
    )

    #src_ds = None

    if out_ds is None:
        fail("gdal.Warp falló al generar el ráster temporal.")

    #out_ds = None

    if not os.path.isfile(dst_path):
        fail(f"No se generó el ráster temporal esperado: {dst_path}")


def apply_mask_with_calc(
    input_a: str,
    input_b: str,
    output_path: str,
    nodata_value: int = NODATA_VALUE,
) -> None:
    log("\nAplicando máscara: conservar A donde B > 0; asignar NoData fuera...")

    Calc(
        calc=f"((B>0)*A) + ((B<=0)*{nodata_value})",
        A=input_a,
        B=input_b,
        outfile=output_path,
        NoDataValue=nodata_value,
        type="Byte",
        format="GTiff",
        overwrite=True,
        quiet=False,
    )

    if not os.path.isfile(output_path):
        fail(f"No se generó el ráster de salida esperado: {output_path}")


def main() -> int:
    gdal.UseExceptions()

    tmp_path = os.path.join(OUT_DIR, TMP_FILENAME)
    out_path = os.path.join(OUT_DIR, OUT_FILENAME)

    log("Validando entradas...")
    validate_input_file(REG_RASTER_PATH, "raster máscara")
    validate_input_file(ZVH_SRC_PATH, "raster de zonas de vida")
    ensure_output_dir(OUT_DIR)

    if os.path.abspath(tmp_path) == os.path.abspath(out_path):
        fail("La ruta temporal y la ruta final no pueden ser la misma.")

    reg_ds = gdal.Open(REG_RASTER_PATH)
    if reg_ds is None:
        fail(f"No se pudo abrir la máscara: {REG_RASTER_PATH}")

    reg_info = dataset_info(reg_ds, "reg_unidas")
    print_dataset_info(reg_info)
    reg_ds = None

    remove_file_safely(tmp_path)
    remove_file_safely(out_path)

    warp_to_template(
        src_path=ZVH_SRC_PATH,
        template_info=reg_info,
        dst_path=tmp_path,
        default_src_srs=DEFAULT_SRC_SRS,
        nodata_value=NODATA_VALUE,
    )

    apply_mask_with_calc(
        input_a=tmp_path,
        input_b=REG_RASTER_PATH,
        output_path=out_path,
        nodata_value=NODATA_VALUE,
    )

    if DELETE_TEMP:
        log("\nEliminando temporal...")
        remove_file_safely(tmp_path)

    log("\n----------------------------------------")
    log("Proceso completado.")
    log(f"Salida final: {out_path}")
    log("Tipo de salida: Byte")
    log(f"NoData de salida: {NODATA_VALUE}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)