"""
=============================================================================
4_crear_raster_regional.py
-----------------------------------------------------------------------------
Propósito:
    Rasterización de datos vectoriales en una matriz ráster (GeoTIFF)
    optimizada para análisis regional.

Rol en el workflow:
    Procesamiento espacial. Conversión de vectores a matriz de píxeles.
=============================================================================
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import numpy as np

# =============================================================================
# Configuración y Parámetros
# =============================================================================
input_csv = snakemake.input["csv_region"]
output_tif = snakemake.output["tif_region"]
pixel_size_metros = snakemake.params["res"]
res_grados = pixel_size_metros / 111000.0

def main():
    # 1. Leer datos
    df = pd.read_csv(input_csv)
    geometrias = [Point(xy) for xy in zip(df['longitud'], df['latitud'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometrias, crs="EPSG:4326")

    # 2. Definir límites y transformación
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = int((xmax - xmin) / res_grados) + 1
    height = int((ymax - ymin) / res_grados) + 1
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # 3. Rasterización
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['valor_indice']))
    matriz = rasterize(shapes=shapes, out_shape=(height, width), 
                       transform=transform, fill=-9999, dtype=rasterio.float32)

    # 4. Exportar ráster
    with rasterio.open(output_tif, 'w', driver='GTiff', height=height, width=width,
                       count=1, dtype=rasterio.float32, crs='EPSG:4326',
                       transform=transform, nodata=-9999) as dst:
        dst.write(matriz, 1)

if __name__ == "__main__":
    main()
