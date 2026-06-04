import os
from osgeo import gdal
from qgis.core import QgsRasterLayer, QgsProject

# 1. RUTAS DE ENTRADA / SALIDA

gebco_src = r"C:/sig_costas_/batimetria/01_GEBCO2020_SIMAR.tif"
shp_mask  = r"C:/sig_costas_/region_marina/regionmarinamx.shp"
reg_unidas_path = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

# Carpeta y archivo de salida
out_dir = r"C:/sig_costas_/batimetria"
os.makedirs(out_dir, exist_ok=True)
gebco_final = os.path.join(out_dir, "GEBCO_regionmarina_regunidas.tif")


# 2. LEER RASTER PLANTILLA (reg_unidas) PARA OBTENER CRS, RESOLUCIÓN Y EXTENSIÓN

reg_ds = gdal.Open(reg_unidas_path)
if reg_ds is None:
    raise RuntimeError(f"No se pudo abrir reg_unidas: {reg_unidas_path}")

reg_gt    = reg_ds.GetGeoTransform()
reg_proj  = reg_ds.GetProjection()
reg_xsize = reg_ds.RasterXSize
reg_ysize = reg_ds.RasterYSize

# Extensión de reg_unidas (en su CRS)
xmin = reg_gt[0]
ymax = reg_gt[3]
xmax = xmin + reg_gt[1] * reg_xsize
ymin = ymax + reg_gt[5] * reg_ysize  # reg_gt[5] suele ser negativo

print("CRS de reg_unidas:")
print(reg_proj)
print("Resolución (x, y):", reg_gt[1], abs(reg_gt[5]))
print("Extensión (xmin, ymin, xmax, ymax):", xmin, ymin, xmax, ymax)


# 3. UN SOLO WARP: RECORTAR POR MÁSCARA + AJUSTAR A LA MALLA DE reg_unidas

if not os.path.exists(gebco_src):
    raise RuntimeError(f"No se encontró el ráster de batimetría: {gebco_src}")

print("\nReproyectando GEBCO, recortando a regionmarinamx y alineando a reg_unidas...")

gdal.Warp(
    gebco_final,        # salida
    gebco_src,          # entrada
    dstSRS=reg_proj,    # mismo CRS que reg_unidas
    xRes=reg_gt[1],     # misma resolución en X
    yRes=abs(reg_gt[5]),# misma resolución en Y
    outputBounds=(xmin, ymin, xmax, ymax),  # misma extensión que reg_unidas
    cutlineDSName=shp_mask,     # shapefile de máscara (región marina)
    cropToCutline=True,         # recortar a la envolvente del polígono
    dstNodata=-9999,            # valor NoData fuera de región marina
    resampleAlg="bilinear"      # batimetría = variable continua
)

print("\n------------------------------------------")
print("Proceso completado.")
print("Raster resultante (batimetría marina alineada a reg_unidas):")
print("  ", gebco_final)
print("------------------------------------------")
