import os
from osgeo import gdal
import numpy as np
from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsProject
import processing
import csv


# RUTAS DE ENTRADA / SALIDA


# Raster máscara 
reg_raster_path = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

# Carpeta donde están los NetCDF de ERA5
nc_dir = r"C:/sig_costas_/dataset-sis-biodiversity-era5-global-7f8ab730-6cb3-450f-8b27-f94d9a678c3a"

# Archivos NetCDF (magnitud, componente meridional y zonal)
nc_wspeed      = os.path.join(nc_dir, "wind-speed_monthly-mean_era5_1979-2018_v1.0.nc")
nc_meridional  = os.path.join(nc_dir, "meridional-wind-speed_monthly-mean_era5_1979-2018_v1.0.nc")
nc_zonal_merid = os.path.join(nc_dir, "zonal-wind-speed_monthly-mean_era5_1979-2018_v1.0.nc")

# Carpeta de salida
out_dir = os.path.join(nc_dir, "resultado_pyqgis")
os.makedirs(out_dir, exist_ok=True)

# Promedios globales (antes de recortar)
wspeed_mean_tif      = os.path.join(out_dir, "ws_mean_global.tif")
merid_mean_tif       = os.path.join(out_dir, "merid_mean_global.tif")
zonal_merid_mean_tif = os.path.join(out_dir, "zonal_merid_mean_global.tif")

# Rásteres de viento ya alineados a reg_unidas 
wspeed_clip_tif      = os.path.join(out_dir, "ws_mean_regunidas.tif")
merid_clip_tif       = os.path.join(out_dir, "merid_mean_regunidas.tif")
zonal_merid_clip_tif = os.path.join(out_dir, "zonal_merid_mean_regunidas.tif")

# Puntos y CSV finales
points_path  = os.path.join(out_dir, "reg_unidas_points.gpkg")
sampled_path = os.path.join(out_dir, "reg_unidas_wind_points.gpkg")
csv_out      = os.path.join(out_dir, "avg_windspeed_pyqgis.csv")


# 1. Promedio NetCDF → GeoTIFF (tipo terra::app(..., mean))

def nc_mean_to_tif(nc_path, tif_out):
    """
    Promedia todas las bandas/tiempos de un NetCDF y guarda GeoTIFF.
    Convierte valores de relleno (_FillValue / NoData / 1e20 / 9.96921e36) a NaN
    antes de promediar.
    """
    print(f"\nProcesando NetCDF: {nc_path}")
    ds = gdal.Open(nc_path)
    if ds is None:
        raise RuntimeError(f"No se pudo abrir {nc_path}")

    nb = ds.RasterCount
    if nb == 0:
        raise RuntimeError(f"{nc_path} no tiene bandas")

    arrays = []

    # Detecta NoData de la primera banda
    first_band = ds.GetRasterBand(1)
    nodata_val = first_band.GetNoDataValue()
    # Valores típicos de relleno en ERA5
    possible_fills = [nodata_val, 1e20, 9.96921e36]
    possible_fills = [v for v in possible_fills if v is not None]

    for b in range(1, nb + 1):
        band = ds.GetRasterBand(b)
        arr = band.ReadAsArray().astype(float)

        # Relleno → NaN
        for fv in possible_fills:
            arr[arr == fv] = np.nan

    
        arr[np.abs(arr) > 1e10] = np.nan

        arrays.append(arr)

    mean_arr = np.nanmean(np.stack(arrays), axis=0)

    # NaN → -9999 para guardar a disco
    mean_arr = np.where(np.isnan(mean_arr), -9999, mean_arr)

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        tif_out,
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    out_band = out.GetRasterBand(1)
    out_band.WriteArray(mean_arr)
    out_band.SetNoDataValue(-9999)
    out.FlushCache()
    out = None

    print(f"  → Promedio guardado en {tif_out}")

# Calcula los promedios (equivalente a wspeed_mean <- app(wspeed, mean))
nc_mean_to_tif(nc_wspeed,      wspeed_mean_tif)
nc_mean_to_tif(nc_meridional,  merid_mean_tif)
nc_mean_to_tif(nc_zonal_merid, zonal_merid_mean_tif)


# 2. Raster plantilla: reg_unidas 

reg_layer = QgsRasterLayer(reg_raster_path, "reg_unidas")
if not reg_layer.isValid():
    raise RuntimeError("No se pudo cargar reg_unidas.tif")
QgsProject.instance().addMapLayer(reg_layer)

reg_ds = gdal.Open(reg_raster_path)
if reg_ds is None:
    raise RuntimeError(f"No se pudo abrir {reg_raster_path}")

reg_gt    = reg_ds.GetGeoTransform()
reg_proj  = reg_ds.GetProjection()
reg_xsize = reg_ds.RasterXSize
reg_ysize = reg_ds.RasterYSize

xmin = reg_gt[0]
ymax = reg_gt[3]
xmax = xmin + reg_gt[1] * reg_xsize
ymin = ymax + reg_gt[5] * reg_ysize   # reg_gt[5] < 0

print("\nCRS de reg_unidas:")
print(reg_proj)

# 3. Reproyectar + recortar promedios a la plantilla reg_unidas

def warp_to_reg_unidas(src_tif, dst_tif):
    """
    Reproyecta y recorta src_tif al CRS, resolución y extensión de reg_unidas.
    """
    print(f"\nReproyectando y recortando: {src_tif}")
    src_ds = gdal.Open(src_tif)
    if src_ds is None:
        raise RuntimeError(f"No se pudo abrir {src_tif}")

    src_proj = src_ds.GetProjection()
    if not src_proj or src_proj.strip() == "":
                print("  (No se encontró CRS en el raster de origen, usando EPSG:4326)")
        src_proj = "EPSG:4326"

    gdal.Warp(
        dst_tif,
        src_tif,
        srcSRS=src_proj,
        dstSRS=reg_proj,
        xRes=reg_gt[1],
        yRes=abs(reg_gt[5]),
        outputBounds=(xmin, ymin, xmax, ymax),
        resampleAlg='bilinear',
        srcNodata=-9999,
        dstNodata=-9999
    )
    print(f"  → guardado en {dst_tif}")

# Aplica a los tres rásteres de viento
warp_to_reg_unidas(wspeed_mean_tif,      wspeed_clip_tif)
warp_to_reg_unidas(merid_mean_tif,       merid_clip_tif)
warp_to_reg_unidas(zonal_merid_mean_tif, zonal_merid_clip_tif)


# 4. Crear puntos a partir de la malla de reg_unidas

print("\nConvirtiendo reg_unidas a puntos de centro de píxel...")

pts_result = processing.run(
    "native:pixelstopoints",
    {
        "INPUT_RASTER": reg_raster_path,
        "RASTER_BAND": 1,
        "FIELD_NAME": "reg_value",
        "OUTPUT": points_path
    }
)


points_layer = QgsVectorLayer(points_path, "reg_unidas_points", "ogr")
if not points_layer.isValid():
    raise RuntimeError("No se pudo cargar la capa de puntos")
QgsProject.instance().addMapLayer(points_layer)

# 5. Extraer valores de los rásteres de viento en esos puntos

print("Extrayendo valores de viento en cada píxel (punto)...")

sampled_result = processing.run(
    "qgis:rastersampling",
    {
        "INPUT": pts_result["OUTPUT"],
        "RASTERS": [wspeed_clip_tif, merid_clip_tif, zonal_merid_clip_tif],
        "COLUMN_PREFIX": "wind_",
        "OUTPUT": sampled_path
    }
)

final_points = QgsVectorLayer(sampled_path, "wind_points", "ogr")
if not final_points.isValid():
    raise RuntimeError("No se pudo cargar la capa de puntos con viento")
QgsProject.instance().addMapLayer(final_points)

# Las columnas nuevas típicamente serán:
#   wind_1  = wspeed_mean_regunidas
#   wind_2  = merid_mean_regunidas
#   wind_3  = zonal_merid_mean_regunidas


