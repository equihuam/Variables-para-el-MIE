"""
=============================================================================
3_crear_reticula.py
-----------------------------------------------------------------------------
Propósito:
    Generar un archivo vectorial (GeoPackage) de resolución variable 
    basado en puntos de entrada, normalizando la cuadrícula mediante 
    anclaje matemático.

Rol en el workflow:
    Procesamiento espacial. Transformación de datos crudos a retícula vectorial.
=============================================================================
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import shapely.wkt

# =============================================================================
# Configuración y Parámetros
# =============================================================================
input_csv = snakemake.input["csv_puntos"]
output_gpkg = snakemake.output["gpkg"]
pixel_size_metros = snakemake.params["res"]
res_grados = pixel_size_metros / 111000.0

def create_pixel_box(point, res):
    """Encaja el punto en una malla global fija"""
    x_snap = np.floor(point.x / res) * res
    y_snap = np.floor(point.y / res) * res
    return box(x_snap, y_snap, x_snap + res, y_snap + res)

def main():
    # 1. Lectura de datos
    df = pd.read_csv(input_csv)
    col_valor = 'valor_iie' if 'valor_iie' in df.columns else 'valor_indice'

    # 2. Transformación espacial
    geometrias = [Point(xy) for xy in zip(df['longitud'], df['latitud'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometrias, crs="EPSG:4326")

    # 3. Creación de retícula
    gdf['geometry'] = gdf.geometry.apply(lambda p: create_pixel_box(p, res_grados))

    # 4. Agregación y Exportación
    gdf['geom_wkt'] = gdf.geometry.apply(lambda g: g.wkt)
    grid = gdf.groupby('geom_wkt').agg({col_valor: 'mean'}).reset_index()
    
    grid['geometry'] = grid['geom_wkt'].apply(shapely.wkt.loads)
    grid = gpd.GeoDataFrame(grid, geometry='geometry', crs="EPSG:4326")
    
    grid.to_file(output_gpkg, driver="GPKG")

if __name__ == "__main__":
    main()
