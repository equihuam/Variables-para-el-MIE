# -*- coding: utf-8 -*-
"""
Procesa shapefiles de huracanes para convertirlos a rásteres alineados
a la malla de reg_unidas.

Flujo:
1. Recorre recursivamente base_dir en busca de .shp
2. Reproyecta cada shapefile al CRS de reg_unidas
3. Detecta un campo numérico (o usa uno forzado)
4. Rasteriza en la malla exacta de reg_unidas
5. Guarda un GeoTIFF final por shapefile

Versión robusta para ejecución no supervisada, sin GUI ni QGIS Desktop.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from osgeo import gdal, ogr, osr

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = r"C:/sig_costas_/huracanes"
REG_UNIDAS_PATH = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

OUT_DIR = os.path.join(BASE_DIR, "huracanes_raster")
NODATA_VALUE = -9999.0
OUTPUT_TYPE = gdal.GDT_Float32

# Si quieres forzar un campo concreto, pon aquí su nombre.
# Si queda en None, se busca automáticamente el primer campo numérico.
FORCED_ATTR_FIELD = None

# Si True, elimina shapefiles reproyectados intermedios
DELETE_REPROJECTED = True


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


def remove_shapefile_set(shp_path: str) -> None:
    """
    Elimina el conjunto de archivos asociados a un shapefile.
    """
    base = os.path.splitext(shp_path)[0]
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]
    for ext in exts:
        p = base + ext
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError as e:
            log(f"No se pudo eliminar {p}: {e}")


def is_within_output_dir(path: str, out_dir: str) -> bool:
    """
    Evita reprocesar archivos que estén dentro del directorio de salida.
    """
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(out_dir)]) == os.path.abspath(out_dir)
    except ValueError:
        return False


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


# ---------------------------------------------------------------------
# VECTOR
# ---------------------------------------------------------------------

def find_numeric_field(shp_path: str) -> Optional[str]:
    ds = ogr.Open(shp_path)
    if ds is None:
        fail(f"No se pudo abrir el shapefile para inspección: {shp_path}")

    layer = ds.GetLayer()
    defn = layer.GetLayerDefn()

    for i in range(defn.GetFieldCount()):
        fd = defn.GetFieldDefn(i)
        if fd.GetType() in (ogr.OFTInteger, ogr.OFTInteger64, ogr.OFTReal):
            field_name = fd.GetName()
            ds = None
            return field_name

    ds = None
    return None


def reproject_vector_to_template(shp_in: str, shp_out: str, target_srs_wkt: str) -> None:
    """
    Reproyecta un shapefile al CRS de la plantilla usando GDAL.
    """
    remove_shapefile_set(shp_out)

    src_ds = ogr.Open(shp_in)
    if src_ds is None:
        fail(f"No se pudo abrir el shapefile de entrada: {shp_in}")

    out_ds = gdal.VectorTranslate(
        destNameOrDestDS=shp_out,
        srcDS=src_ds,
        format="ESRI Shapefile",
        reproject=True,
        dstSRS=target_srs_wkt
    )

    src_ds = None

    if out_ds is None:
        fail(f"Falló la reproyección del shapefile: {shp_in}")

    out_ds = None

    if not os.path.exists(shp_out):
        fail(f"No se generó el shapefile reproyectado: {shp_out}")


# ---------------------------------------------------------------------
# RASTERIZACIÓN
# ---------------------------------------------------------------------

def rasterize_to_template(
    shp_path: str,
    attr_field: str,
    raster_out: str,
    template_info: dict,
    nodata_value: float = NODATA_VALUE,
) -> None:
    """
    Rasteriza un shapefile usando exactamente la malla de reg_unidas.
    """
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        raster_out,
        template_info["xsize"],
        template_info["ysize"],
        1,
        OUTPUT_TYPE,
        options=["COMPRESS=LZW"]
    )
    if out_ds is None:
        fail(f"No se pudo crear el raster de salida: {raster_out}")

    out_ds.SetGeoTransform(template_info["geotransform"])
    out_ds.SetProjection(template_info["projection"])

    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    band.Fill(nodata_value)

    vec_ds = ogr.Open(shp_path)
    if vec_ds is None:
        out_ds = None
        fail(f"No se pudo abrir el shapefile reproyectado: {shp_path}")

    layer = vec_ds.GetLayer()

    err = gdal.RasterizeLayer(
        out_ds,
        [1],
        layer,
        options=[
            f"ATTRIBUTE={attr_field}",
            "ALL_TOUCHED=FALSE"
        ]
    )

    vec_ds = None
    band.FlushCache()
    out_ds.FlushCache()
    out_ds = None

    if err != 0:
        fail(f"gdal.RasterizeLayer falló para {shp_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:
    gdal.UseExceptions()

    log("Validando entradas...")
    validate_input_file(REG_UNIDAS_PATH, "raster plantilla")
    ensure_output_dir(OUT_DIR)

    reg_ds = gdal.Open(REG_UNIDAS_PATH)
    if reg_ds is None:
        fail(f"No se pudo abrir reg_unidas: {REG_UNIDAS_PATH}")

    reg_info = dataset_info(reg_ds, "reg_unidas")
    print_dataset_info(reg_info)
    reg_ds = None

    log(f"\nBuscando shapefiles en: {BASE_DIR}")

    processed = 0

    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if not f.lower().endswith(".shp"):
                continue

            shp_path = os.path.join(root, f)

            # saltar carpeta de salida
            if is_within_output_dir(shp_path, OUT_DIR):
                continue

            shp_name = os.path.splitext(f)[0]

            log("\n----------------------------------------")
            log(f"Procesando shapefile: {shp_path}")

            shp_reproj = os.path.join(OUT_DIR, f"{shp_name}_reproj.shp")
            raster_final = os.path.join(OUT_DIR, f"{shp_name}_regunidas.tif")

            # 1. Reproyección
            reproject_vector_to_template(
                shp_in=shp_path,
                shp_out=shp_reproj,
                target_srs_wkt=reg_info["projection"]
            )
            log(f"  → Reproyectado: {shp_reproj}")

            # 2. Campo numérico
            attr_field = FORCED_ATTR_FIELD if FORCED_ATTR_FIELD else find_numeric_field(shp_reproj)
            if not attr_field:
                fail(f"No se encontró campo numérico en: {shp_reproj}")

            log(f"  → Campo usado: {attr_field}")

            # 3. Rasterización directa a la malla final
            rasterize_to_template(
                shp_path=shp_reproj,
                attr_field=attr_field,
                raster_out=raster_final,
                template_info=reg_info,
                nodata_value=NODATA_VALUE
            )
            log(f"  → Raster final: {raster_final}")

            # 4. Limpieza opcional
            if DELETE_REPROJECTED:
                remove_shapefile_set(shp_reproj)

            processed += 1

    log("\n========================================")
    log("PROCESO COMPLETADO")
    log(f"Shapefiles procesados: {processed}")
    log(f"Todos los rásteres están en: {OUT_DIR}")
    log("========================================")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)