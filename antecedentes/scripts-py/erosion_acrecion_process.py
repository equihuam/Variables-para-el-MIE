import os
from osgeo import gdal, ogr
from qgis.core import QgsRasterLayer, QgsProject
import processing


# 1. RUTAS


vector_in = r"C:/sig_costas_/TasasErosiónAcreción/TasasdeErosion2.shp"
reg_unidas_path = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

out_dir = r"C:/sig_costas_/TasasErosiónAcreción"
os.makedirs(out_dir, exist_ok=True)

vector_reproj = os.path.join(out_dir, "TasasdeErosion2_reproyectada.shp")
raster_out    = os.path.join(out_dir, "TasasdeErosion2_regunidas.tif")

attr_field = "Tasa"


# 2. CARGAR reg_unidas COMO PLANTILLA (CRS + RESOLUCIÓN)


reg_layer = QgsRasterLayer(reg_unidas_path, "reg_unidas")
if not reg_layer.isValid():
    raise RuntimeError(f"No se pudo cargar reg_unidas: {reg_unidas_path}")

QgsProject.instance().addMapLayer(reg_layer)

target_crs = reg_layer.crs()          # CRS objetivo
reg_ds = gdal.Open(reg_unidas_path)   # para malla

reg_gt    = reg_ds.GetGeoTransform()
reg_proj  = reg_ds.GetProjection()
reg_xsize = reg_ds.RasterXSize
reg_ysize = reg_ds.RasterYSize

print("Plantilla reg_unidas:")
print("  Tamaño:", reg_xsize, "x", reg_ysize)
print("  Resolución:", reg_gt[1], "x", abs(reg_gt[5]))
print("  CRS:", target_crs.toWkt()[0:80], "...")

# 3. REPROYECTAR TasasdeErosion2 AL CRS DE reg_unidas

print("\nReproyectando TasasdeErosion2 al CRS de reg_unidas...")

processing.run(
    "native:reprojectlayer",
    {
        "INPUT": vector_in,
        "TARGET_CRS": target_crs,   # CRS de reg_unidas
        "OUTPUT": vector_reproj
    }
)

print("Vector reproyectado guardado en:", vector_reproj)


# 4. CREAR RÁSTER VACÍO CON LA MISMA MALLA QUE reg_unidas


driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(
    raster_out,
    reg_xsize,
    reg_ysize,
    1,                 # una banda
    gdal.GDT_Float32   # tipo de dato
)

out_ds.SetGeoTransform(reg_gt)
out_ds.SetProjection(reg_proj)

out_band = out_ds.GetRasterBand(1)
out_band.SetNoDataValue(-9999)
out_band.Fill(-9999)   # inicializamos con NoData


# 5. RASTERIZAR USANDO EL CAMPO 'Tasa'


vec_ds = ogr.Open(vector_reproj)
if vec_ds is None:
    raise RuntimeError(f"No se pudo abrir el vector reproyectado: {vector_reproj}")

layer = vec_ds.GetLayer()

print("\nRasterizando usando el atributo 'Tasa'...")

gdal.RasterizeLayer(
    out_ds,
    [1],            # bandas a escribir
    layer,
    options=[
        "ATTRIBUTE=Tasa",   # campo a usar
        "ALL_TOUCHED=FALSE" # solo celdas cuyo centro cae en la geometría
    ]
)

out_ds.FlushCache()
out_ds = None
vec_ds = None

print("\n----------------------------------------")
print("Raster de tasas generado en:")
print("  ", raster_out)
print("----------------------------------------")
