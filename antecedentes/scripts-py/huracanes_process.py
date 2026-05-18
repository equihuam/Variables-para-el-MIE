import os
from osgeo import gdal, ogr
from qgis.core import QgsRasterLayer, QgsProject
import processing


# 1. RUTAS


base_dir = r"C:/sig_costas_/huracanes"
reg_unidas_path = r"C:/sig_costas_/regiones_unidas/reg_unidas.tif"

out_dir = os.path.join(base_dir, "huracanes_raster")
os.makedirs(out_dir, exist_ok=True)


# 2. LEER RASTER PLANTILLA (reg_unidas)

reg_layer = QgsRasterLayer(reg_unidas_path, "reg_unidas")
if not reg_layer.isValid():
    raise RuntimeError(f"No se pudo cargar reg_unidas: {reg_unidas_path}")

QgsProject.instance().addMapLayer(reg_layer)

target_crs = reg_layer.crs()
reg_ds = gdal.Open(reg_unidas_path)

reg_gt    = reg_ds.GetGeoTransform()
reg_proj  = reg_ds.GetProjection()
reg_xsize = reg_ds.RasterXSize
reg_ysize = reg_ds.RasterYSize

xmin = reg_gt[0]
ymax = reg_gt[3]
xmax = xmin + reg_gt[1] * reg_xsize
ymin = ymax + reg_gt[5] * reg_ysize

print("Plantilla reg_unidas cargada correctamente")


# 3. FUNCIÓN PARA DETECTAR CAMPO NUMÉRICO

def find_numeric_field(shp_path):
    ds = ogr.Open(shp_path)
    layer = ds.GetLayer()
    defn = layer.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        fd = defn.GetFieldDefn(i)
        if fd.GetType() in (ogr.OFTInteger, ogr.OFTInteger64, ogr.OFTReal):
            return fd.GetName()
    return None

FORCED_ATTR_FIELD = None



# 4. LOOP SOBRE TODAS LAS SUBCARPETAS

print("\nBuscando shapefiles en:", base_dir)

for root, dirs, files in os.walk(base_dir):
    for f in files:
        if not f.lower().endswith(".shp"):
            continue

        shp_path = os.path.join(root, f)

        # saltar carpeta de salida
        if out_dir in os.path.abspath(shp_path):
            continue

        shp_name = os.path.splitext(f)[0]

        print("\n----------------------------------------")
        print("Procesando shapefile:", shp_path)
    
        # 4.1 Reproyección al CRS de reg_unidas

        shp_reproj = os.path.join(out_dir, f"{shp_name}_reproj.shp")

        processing.run(
            "native:reprojectlayer",
            {
                "INPUT": shp_path,
                "TARGET_CRS": target_crs,
                "OUTPUT": shp_reproj
            }
        )

        print("  → Reproyectado:", shp_reproj)


        
        # 4.2 Campo numérico a rasterizar

        if FORCED_ATTR_FIELD:
            attr_field = FORCED_ATTR_FIELD
        else:
            attr_field = find_numeric_field(shp_reproj)

        print("  → Campo usado:", attr_field)


       
        # 4.3 Rasterizar en la malla de reg_unidas

        raster_tmp = os.path.join(out_dir, f"{shp_name}_tmp.tif")

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            raster_tmp,
            reg_xsize,
            reg_ysize,
            1,
            gdal.GDT_Float32
        )
        out_ds.SetGeoTransform(reg_gt)
        out_ds.SetProjection(reg_proj)

        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(-9999)
        out_band.Fill(-9999)

        vec_ds = ogr.Open(shp_reproj)
        layer = vec_ds.GetLayer()

        gdal.RasterizeLayer(
            out_ds,
            [1],
            layer,
            options=[
                f"ATTRIBUTE={attr_field}",
                "ALL_TOUCHED=FALSE"
            ]
        )

        out_ds.FlushCache()
        out_ds = None
        vec_ds = None

        print("  → Rasterizado inicial:", raster_tmp)


        
        # 4.4 Recortar a EXTENSIÓN EXACTA de reg_unidas

        raster_final = os.path.join(out_dir, f"{shp_name}_regunidas.tif")

        gdal.Warp(
            raster_final,
            raster_tmp,
            dstSRS=reg_proj,
            xRes=reg_gt[1],
            yRes=abs(reg_gt[5]),
            outputBounds=(xmin, ymin, xmax, ymax),
            dstNodata=-9999
        )

        print("  → Raster FINAL recortado:", raster_final)

        # eliminar el temporal
        try:
            os.remove(raster_tmp)
        except:
            pass

print("\n========================================")
print("PROCESO COMPLETADO")
print("Todos los rásteres están en:")
print(out_dir)
print("========================================")
