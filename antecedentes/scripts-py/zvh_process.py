import os
from osgeo import gdal
import processing


# 1. RUTAS

# Raster máscara 
reg_raster_path = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

# Raster original de zonas de vida
zvh_src_path = r"C:/sig_costas_/zonas_vida_h/07_zvh_mx3gw/zvh_mx3gw.tif"

# Carpeta y archivos de salida
out_dir = r"C:/sig_costas_/zonas_vida_h/07_zvh_mx3gw"
os.makedirs(out_dir, exist_ok=True)

zvh_tmp_path    = os.path.join(out_dir, "zvh_mx3gw_regunidas_tmp.tif")
zvh_masked_path = os.path.join(out_dir, "zvh_mx3gw_regunidas_dunas.tif")


# 2. LEER RASTER PLANTILLA (reg_unidas)

reg_ds = gdal.Open(reg_raster_path)
if reg_ds is None:
    raise RuntimeError(f"No se pudo abrir la máscara: {reg_raster_path}")

reg_gt    = reg_ds.GetGeoTransform()
reg_proj  = reg_ds.GetProjection()
reg_xsize = reg_ds.RasterXSize
reg_ysize = reg_ds.RasterYSize

xmin = reg_gt[0]
ymax = reg_gt[3]
xmax = xmin + reg_gt[1] * reg_xsize
ymin = ymax + reg_gt[5] * reg_ysize  # reg_gt[5] suele ser negativo

print("CRS de reg_unidas:")
print(reg_proj)


# 3. WARP: AJUSTAR zvh_mx3gw A LA MALLA DE reg_unidas


zvh_src_ds = gdal.Open(zvh_src_path)
if zvh_src_ds is None:
    raise RuntimeError(f"No se pudo abrir el ráster de origen: {zvh_src_path}")

src_proj = zvh_src_ds.GetProjection()
if not src_proj or src_proj.strip() == "":
    # AJUSTA ESTO si tu zvh tiene otro CRS conocido
    print("Advertencia: el ráster de origen no tiene CRS; se asume EPSG:4326.")
    src_proj = "EPSG:4326"

print("\nCRS de zvh_mx3gw:")
print(src_proj)

print("\nReproyectando y recortando zvh_mx3gw a la malla de reg_unidas...")

gdal.Warp(
    zvh_tmp_path,
    zvh_src_path,
    srcSRS=src_proj,
    dstSRS=reg_proj,
    xRes=reg_gt[1],
    yRes=abs(reg_gt[5]),
    outputBounds=(xmin, ymin, xmax, ymax),
    resampleAlg="near",   # vecino más cercano (dato categórico)
    dstNodata=-9999
)


# 4. APLICAR MÁSCARA DE DUNAS USANDO gdal:rastercalculator
#    (evita cargar los ráster en NumPy)


print("\nAplicando máscara de dunas (reg_unidas > 0) al ráster warp...")

# Fórmula:
#   greater(B,0) = 1 donde reg_unidas>0 (dunas), 0 en otro caso
#   output = A*greater(B,0) + (-9999)*(1-greater(B,0))

alg_params = {
    'INPUT_A': zvh_tmp_path,
    'BAND_A': 1,
    'INPUT_B': reg_raster_path,
    'BAND_B': 1,
    'INPUT_C': None,
    'BAND_C': -1,
    'INPUT_D': None,
    'BAND_D': -1,
    'INPUT_E': None,
    'BAND_E': -1,
    'INPUT_F': None,
    'BAND_F': -1,
    'FORMULA': 'A*greater(B,0)+(-9999)*(1-greater(B,0))',
    'NO_DATA': -9999,
    'RTYPE': 5,           # Float32 (cambia si quieres entero)
    'EXTRA': '',
    'OUTPUT': zvh_masked_path
}

processing.run("gdal:rastercalculator", alg_params)

# (Opcional) borrar el intermedio
try:
    os.remove(zvh_tmp_path)
    print(f"\nRaster temporal eliminado: {zvh_tmp_path}")
except OSError:
    print(f"\nNo se pudo eliminar el raster temporal (puede no existir): {zvh_tmp_path}")

print("\n----------------------------------------")
print("Proceso completado.")
print("Ráster final (alineado a reg_unidas y solo en dunas) en:")
print("  ", zvh_masked_path)
print("----------------------------------------")
