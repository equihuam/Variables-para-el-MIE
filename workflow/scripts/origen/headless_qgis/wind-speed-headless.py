# -*- coding: utf-8 -*-
"""
Procesa variables de viento ERA5 para una malla plantilla (reg_unidas):

1. Promedia en el tiempo tres NetCDF:
   - velocidad del viento
   - componente meridional
   - componente zonal

2. Reproyecta / recorta / alinea los promedios a la malla de reg_unidas.

3. Genera puntos en los centros de píxel donde reg_unidas > 0.

4. Extrae en esos puntos los valores de los tres rásteres alineados.

5. Guarda:
   - un GeoPackage con puntos y atributos
   - un CSV tabular

Versión robusta para ejecución no supervisada en qgis_env Python,
sin dependencia de QGIS Desktop ni Processing.
"""

import os
import sys
import csv
from typing import Dict, Tuple

import numpy as np
from osgeo import gdal, ogr, osr

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

REG_RASTER_PATH = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

NC_DIR = r"C:/sig_costas_/dataset-sis-biodiversity-era5-global-7f8ab730-6cb3-450f-8b27-f94d9a678c3a"

NC_WSPEED = os.path.join(NC_DIR, "wind-speed_monthly-mean_era5_1979-2018_v1.0.nc")
NC_MERIDIONAL = os.path.join(NC_DIR, "meridional-wind-speed_monthly-mean_era5_1979-2018_v1.0.nc")
NC_ZONAL = os.path.join(NC_DIR, "zonal-wind-speed_monthly-mean_era5_1979-2018_v1.0.nc")

OUT_DIR = os.path.join(NC_DIR, "resultado_pyqgis")
os.makedirs(OUT_DIR, exist_ok=True)

# Promedios globales
WSPEED_MEAN_TIF = os.path.join(OUT_DIR, "ws_mean_global.tif")
MERID_MEAN_TIF = os.path.join(OUT_DIR, "merid_mean_global.tif")
ZONAL_MEAN_TIF = os.path.join(OUT_DIR, "zonal_mean_global.tif")

# Rásteres alineados a reg_unidas
WSPEED_ALIGN_TIF = os.path.join(OUT_DIR, "ws_mean_regunidas.tif")
MERID_ALIGN_TIF = os.path.join(OUT_DIR, "merid_mean_regunidas.tif")
ZONAL_ALIGN_TIF = os.path.join(OUT_DIR, "zonal_mean_regunidas.tif")

# Salidas vectoriales/tabulares
POINTS_GPKG = os.path.join(OUT_DIR, "reg_unidas_wind_points.gpkg")
CSV_OUT = os.path.join(OUT_DIR, "avg_windspeed_pyqgis.csv")

# Configuración general
FLOAT_NODATA = -9999.0
DEFAULT_SRC_SRS = "EPSG:4326"
DELETE_INTERMEDIATE_GLOBAL_MEANS = False

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


def remove_file_safely(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
            log(f"Archivo eliminado: {path}")
    except OSError as e:
        log(f"No se pudo eliminar {path}: {e}")


def remove_gpkg_safely(path: str) -> None:
    """
    Elimina un GeoPackage previo si existe.
    """
    try:
        driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(path):
            driver.DeleteDataSource(path)
            log(f"GeoPackage eliminado: {path}")
    except Exception as e:
        log(f"No se pudo eliminar el GeoPackage {path}: {e}")


def dataset_info(ds: gdal.Dataset, label: str) -> Dict:
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


def print_dataset_info(info: Dict) -> None:
    log(f"\n--- {info['label']} ---")
    log(f"Dimensiones: {info['xsize']} x {info['ysize']}")
    log(f"Resolución: {info['pixel_width']} x {info['pixel_height']}")
    log(f"Extensión : xmin={info['xmin']}, ymin={info['ymin']}, xmax={info['xmax']}, ymax={info['ymax']}")
    log("CRS:")
    log(info["projection"] if info["projection"] else "  [vacío]")


def pixel_center(gt: Tuple[float, ...], row: int, col: int) -> Tuple[float, float]:
    """
    Centro del píxel (col, row) usando la geotransform.
    """
    x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
    y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
    return x, y


# ---------------------------------------------------------------------
# 1. NETCDF -> PROMEDIO TEMPORAL -> GEOTIFF
# ---------------------------------------------------------------------

def nc_mean_to_tif(nc_path: str, tif_out: str, nodata_out: float = FLOAT_NODATA) -> None:
    """
    Promedia todas las bandas/tiempos de un NetCDF y guarda un GeoTIFF Float32.
    Convierte NoData / FillValue a NaN antes de promediar.
    """
    log(f"\nProcesando NetCDF: {nc_path}")
    ds = gdal.Open(nc_path)
    if ds is None:
        fail(f"No se pudo abrir {nc_path}")

    nb = ds.RasterCount
    if nb == 0:
        fail(f"{nc_path} no tiene bandas")

    arrays = []

    first_band = ds.GetRasterBand(1)
    nodata_val = first_band.GetNoDataValue()
    possible_fills = [nodata_val, 1e20, 9.96921e36]
    possible_fills = [v for v in possible_fills if v is not None]

    for b in range(1, nb + 1):
        band = ds.GetRasterBand(b)
        arr = band.ReadAsArray().astype(np.float32)

        for fv in possible_fills:
            arr[arr == fv] = np.nan

        # Limpieza defensiva para valores absurdos
        arr[np.abs(arr) > 1e10] = np.nan
        arrays.append(arr)

    stack = np.stack(arrays, axis=0)
    mean_arr = np.nanmean(stack, axis=0)
    mean_arr = np.where(np.isnan(mean_arr), nodata_out, mean_arr).astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        tif_out,
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW"]
    )
    if out is None:
        fail(f"No se pudo crear el GeoTIFF de salida: {tif_out}")

    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    out_band = out.GetRasterBand(1)
    out_band.WriteArray(mean_arr)
    out_band.SetNoDataValue(nodata_out)
    out_band.FlushCache()

    out = None
    ds = None

    log(f"  → Promedio guardado en {tif_out}")


# ---------------------------------------------------------------------
# 2. ALINEAR A reg_unidas
# ---------------------------------------------------------------------

def warp_to_template(
    src_tif: str,
    dst_tif: str,
    template_info: Dict,
    src_nodata: float = FLOAT_NODATA,
    dst_nodata: float = FLOAT_NODATA,
    default_src_srs: str = DEFAULT_SRC_SRS,
    resample_alg: str = "bilinear",
) -> None:
    """
    Reproyecta y recorta src_tif al CRS, resolución y extensión de reg_unidas.
    """
    log(f"\nReproyectando y recortando: {src_tif}")

    src_ds = gdal.Open(src_tif)
    if src_ds is None:
        fail(f"No se pudo abrir {src_tif}")

    src_proj = src_ds.GetProjection()
    if not src_proj or not src_proj.strip():
        log(f"  (No se encontró CRS en el raster de origen, usando {default_src_srs})")
        src_proj = default_src_srs

    out_ds = gdal.Warp(
        destNameOrDestDS=dst_tif,
        srcDSOrSrcDSTab=src_tif,
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
        resampleAlg=resample_alg,
        srcNodata=src_nodata,
        dstNodata=dst_nodata,
        outputType=gdal.GDT_Float32,
        targetAlignedPixels=False
    )

    src_ds = None

    if out_ds is None:
        fail(f"gdal.Warp falló para {src_tif}")

    out_ds = None
    log(f"  → guardado en {dst_tif}")


# ---------------------------------------------------------------------
# 3. LEER ARRAYS ALINEADOS
# ---------------------------------------------------------------------

def read_single_band_array(path: str) -> Tuple[np.ndarray, Dict]:
    ds = gdal.Open(path)
    if ds is None:
        fail(f"No se pudo abrir {path}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    info = dataset_info(ds, os.path.basename(path))
    ds = None

    return arr, {"nodata": nodata, "info": info}


# ---------------------------------------------------------------------
# 4. PUNTOS DE CENTRO DE PÍXEL + MUESTREO
# ---------------------------------------------------------------------

def create_points_and_csv(
    reg_raster_path: str,
    ws_path: str,
    merid_path: str,
    zonal_path: str,
    gpkg_out: str,
    csv_out: str,
) -> None:
    """
    Genera puntos en el centro de cada píxel con reg_unidas > 0 y extrae allí
    los valores de los tres rásteres alineados.
    """
    reg_ds = gdal.Open(reg_raster_path)
    if reg_ds is None:
        fail(f"No se pudo abrir el raster máscara: {reg_raster_path}")

    reg_band = reg_ds.GetRasterBand(1)
    reg_arr = reg_band.ReadAsArray()
    reg_gt = reg_ds.GetGeoTransform()
    reg_proj = reg_ds.GetProjection()
    reg_nodata = reg_band.GetNoDataValue()

    ws_arr, ws_meta = read_single_band_array(ws_path)
    merid_arr, merid_meta = read_single_band_array(merid_path)
    zonal_arr, zonal_meta = read_single_band_array(zonal_path)

    # Validación de forma
    if ws_arr.shape != reg_arr.shape or merid_arr.shape != reg_arr.shape or zonal_arr.shape != reg_arr.shape:
        fail("Los rásteres alineados no coinciden en dimensiones con reg_unidas.")

    remove_gpkg_safely(gpkg_out)
    remove_file_safely(csv_out)

    gpkg_driver = ogr.GetDriverByName("GPKG")
    ds_out = gpkg_driver.CreateDataSource(gpkg_out)
    if ds_out is None:
        fail(f"No se pudo crear el GeoPackage: {gpkg_out}")

    srs = osr.SpatialReference()
    if reg_proj:
        srs.ImportFromWkt(reg_proj)
    else:
        fail("reg_unidas no tiene proyección definida.")

    layer = ds_out.CreateLayer("reg_unidas_wind_points", srs, ogr.wkbPoint)
    if layer is None:
        fail("No se pudo crear la capa de puntos en el GeoPackage.")

    # Campos
    field_defs = [
        ("row", ogr.OFTInteger),
        ("col", ogr.OFTInteger),
        ("reg_val", ogr.OFTReal),
        ("wspeed", ogr.OFTReal),
        ("merid", ogr.OFTReal),
        ("zonal", ogr.OFTReal),
    ]
    for field_name, field_type in field_defs:
        fld = ogr.FieldDefn(field_name, field_type)
        layer.CreateField(fld)

    with open(csv_out, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["row", "col", "x", "y", "reg_val", "wspeed", "merid", "zonal"])

        nrows, ncols = reg_arr.shape
        feature_defn = layer.GetLayerDefn()
        total_points = 0

        for row in range(nrows):
            for col in range(ncols):
                reg_val = reg_arr[row, col]

                # Saltar NoData de máscara si existe
                if reg_nodata is not None and reg_val == reg_nodata:
                    continue

                # Mantener sólo celdas válidas de reg_unidas
                if reg_val <= 0:
                    continue

                ws_val = ws_arr[row, col]
                merid_val = merid_arr[row, col]
                zonal_val = zonal_arr[row, col]

                # Saltar celdas sin datos en cualquiera de las capas de viento
                if ws_meta["nodata"] is not None and ws_val == ws_meta["nodata"]:
                    continue
                if merid_meta["nodata"] is not None and merid_val == merid_meta["nodata"]:
                    continue
                if zonal_meta["nodata"] is not None and zonal_val == zonal_meta["nodata"]:
                    continue

                x, y = pixel_center(reg_gt, row, col)

                feat = ogr.Feature(feature_defn)
                feat.SetField("row", int(row))
                feat.SetField("col", int(col))
                feat.SetField("reg_val", float(reg_val))
                feat.SetField("wspeed", float(ws_val))
                feat.SetField("merid", float(merid_val))
                feat.SetField("zonal", float(zonal_val))

                geom = ogr.Geometry(ogr.wkbPoint)
                geom.AddPoint(float(x), float(y))
                feat.SetGeometry(geom)

                if layer.CreateFeature(feat) != 0:
                    fail("No se pudo escribir una entidad en el GeoPackage.")

                writer.writerow([row, col, x, y, reg_val, ws_val, merid_val, zonal_val])

                feat = None
                total_points += 1

    ds_out = None
    reg_ds = None

    log(f"\nPuntos guardados en: {gpkg_out}")
    log(f"CSV guardado en: {csv_out}")
    log(f"Total de puntos válidos: {total_points}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:
    gdal.UseExceptions()

    log("Validando entradas...")
    validate_input_file(REG_RASTER_PATH, "raster máscara")
    validate_input_file(NC_WSPEED, "NetCDF de velocidad del viento")
    validate_input_file(NC_MERIDIONAL, "NetCDF de componente meridional")
    validate_input_file(NC_ZONAL, "NetCDF de componente zonal")

    # Limpiar salidas intermedias/finales previas
    for path in [
        WSPEED_MEAN_TIF,
        MERID_MEAN_TIF,
        ZONAL_MEAN_TIF,
        WSPEED_ALIGN_TIF,
        MERID_ALIGN_TIF,
        ZONAL_ALIGN_TIF,
        CSV_OUT,
    ]:
        remove_file_safely(path)
    remove_gpkg_safely(POINTS_GPKG)

    # 1. Promedios globales
    nc_mean_to_tif(NC_WSPEED, WSPEED_MEAN_TIF)
    nc_mean_to_tif(NC_MERIDIONAL, MERID_MEAN_TIF)
    nc_mean_to_tif(NC_ZONAL, ZONAL_MEAN_TIF)

    # 2. Info plantilla
    reg_ds = gdal.Open(REG_RASTER_PATH)
    if reg_ds is None:
        fail(f"No se pudo abrir la máscara: {REG_RASTER_PATH}")
    reg_info = dataset_info(reg_ds, "reg_unidas")
    print_dataset_info(reg_info)
    reg_ds = None

    # 3. Alinear a plantilla
    warp_to_template(WSPEED_MEAN_TIF, WSPEED_ALIGN_TIF, reg_info, resample_alg="bilinear")
    warp_to_template(MERID_MEAN_TIF, MERID_ALIGN_TIF, reg_info, resample_alg="bilinear")
    warp_to_template(ZONAL_MEAN_TIF, ZONAL_ALIGN_TIF, reg_info, resample_alg="bilinear")

    # 4. Crear puntos y CSV
    create_points_and_csv(
        reg_raster_path=REG_RASTER_PATH,
        ws_path=WSPEED_ALIGN_TIF,
        merid_path=MERID_ALIGN_TIF,
        zonal_path=ZONAL_ALIGN_TIF,
        gpkg_out=POINTS_GPKG,
        csv_out=CSV_OUT,
    )

    # 5. Limpieza opcional
    if DELETE_INTERMEDIATE_GLOBAL_MEANS:
        log("\nEliminando promedios globales intermedios...")
        remove_file_safely(WSPEED_MEAN_TIF)
        remove_file_safely(MERID_MEAN_TIF)
        remove_file_safely(ZONAL_MEAN_TIF)

    log("\n----------------------------------------")
    log("Proceso completado.")
    log(f"Salida vectorial: {POINTS_GPKG}")
    log(f"Salida tabular : {CSV_OUT}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)