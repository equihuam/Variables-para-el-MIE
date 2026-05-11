# -*- coding: utf-8 -*-
"""
Reproyecta un shapefile de tasas de erosión/acreción al CRS de reg_unidas
y rasteriza el campo 'Tasa' sobre la misma malla de la plantilla.

Versión robusta para ejecución no supervisada, sin GUI ni QGIS Desktop.
"""

import os
import sys
from osgeo import gdal, ogr

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

VECTOR_IN = r"C:/sig_costas_/TasasErosiónAcreción/TasasdeErosion2.shp"
REG_UNIDAS_PATH = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

OUT_DIR = r"C:/sig_costas_/TasasErosiónAcreción"
VECTOR_REPROJ = os.path.join(OUT_DIR, "TasasdeErosion2_reproyectada.shp")
RASTER_OUT = os.path.join(OUT_DIR, "TasasdeErosion2_regunidas.tif")

ATTR_FIELD = "Tasa"
NODATA_VALUE = -9999.0
OUTPUT_TYPE = gdal.GDT_Float32

# Si True, elimina el shapefile reproyectado al terminar
DELETE_REPROJECTED = False


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


def remove_file_safely(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        log(f"No se pudo eliminar {path}: {e}")


def remove_shapefile_set(shp_path: str) -> None:
    base = os.path.splitext(shp_path)[0]
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]
    for ext in exts:
        remove_file_safely(base + ext)


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


# ---------------------------------------------------------------------
# VECTOR
# ---------------------------------------------------------------------

def field_exists(shp_path: str, field_name: str) -> bool:
    ds = ogr.Open(shp_path)
    if ds is None:
        fail(f"No se pudo abrir el vector para inspección: {shp_path}")

    layer = ds.GetLayer()
    defn = layer.GetLayerDefn()

    exists = False
    for i in range(defn.GetFieldCount()):
        if defn.GetFieldDefn(i).GetName() == field_name:
            exists = True
            break

    ds = None
    return exists


def reproject_vector_to_template(shp_in: str, shp_out: str, target_srs_wkt: str) -> None:
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

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata_value)
    out_band.Fill(nodata_value)

    vec_ds = ogr.Open(shp_path)
    if vec_ds is None:
        out_ds = None
        fail(f"No se pudo abrir el vector reproyectado: {shp_path}")

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

    out_band.FlushCache()
    out_ds.FlushCache()
    vec_ds = None
    out_ds = None

    if err != 0:
        fail(f"gdal.RasterizeLayer falló para {shp_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:
    gdal.UseExceptions()

    log("Validando entradas...")
    validate_input_file(VECTOR_IN, "vector de tasas")
    validate_input_file(REG_UNIDAS_PATH, "raster plantilla")
    ensure_output_dir(OUT_DIR)

    reg_ds = gdal.Open(REG_UNIDAS_PATH)
    if reg_ds is None:
        fail(f"No se pudo abrir reg_unidas: {REG_UNIDAS_PATH}")

    reg_info = dataset_info(reg_ds, "reg_unidas")
    print_dataset_info(reg_info)
    reg_ds = None

    log("\nReproyectando TasasdeErosion2 al CRS de reg_unidas...")
    reproject_vector_to_template(
        shp_in=VECTOR_IN,
        shp_out=VECTOR_REPROJ,
        target_srs_wkt=reg_info["projection"]
    )
    log(f"Vector reproyectado guardado en: {VECTOR_REPROJ}")

    if not field_exists(VECTOR_REPROJ, ATTR_FIELD):
        fail(f"El campo '{ATTR_FIELD}' no existe en el shapefile reproyectado.")

    log(f"\nRasterizando usando el atributo '{ATTR_FIELD}'...")
    rasterize_to_template(
        shp_path=VECTOR_REPROJ,
        attr_field=ATTR_FIELD,
        raster_out=RASTER_OUT,
        template_info=reg_info,
        nodata_value=NODATA_VALUE
    )

    if DELETE_REPROJECTED:
        remove_shapefile_set(VECTOR_REPROJ)

    log("\n----------------------------------------")
    log("Raster de tasas generado en:")
    log(f"  {RASTER_OUT}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)