"""
=============================================================================
3_crear_reticula.py
-----------------------------------------------------------------------------
Propósito:
    Generar un GeoPackage vectorial de retícula regular a partir de puntos,
    construyendo la grilla en un CRS métrico y exportando en EPSG:4326.

Rol en el workflow:
    Procesamiento espacial. Transformación de datos crudos a retícula vectorial.
=============================================================================
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np


input_csv = snakemake.input["csv_puntos"]
output_gpkg = snakemake.output["gpkg"]
pixel_size_metros = float(snakemake.params["res"])

LAYER_NAME = "reticula_variable"

# CRS de entrada y salida para publicación
CRS_LONLAT = "EPSG:4326"

# CRS métrico para construir la retícula.
# Si EPSG:6372 diera problemas en tu entorno, cambia temporalmente a EPSG:3857.
CRS_METRICO = "EPSG:6372"


def create_pixel_box_metric(point, res_m):
    """
    Encaja un punto en una malla regular definida en metros.
    """
    x_snap = np.floor(point.x / res_m) * res_m
    y_snap = np.floor(point.y / res_m) * res_m

    return box(
        x_snap,
        y_snap,
        x_snap + res_m,
        y_snap + res_m
    )


def main():
    df = pd.read_csv(input_csv)

    col_valor = "valor_iie" if "valor_iie" in df.columns else "valor_indice"

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitud"], df["latitud"])],
        crs=CRS_LONLAT
    )

    # 1. Pasar a CRS métrico
    gdf_m = gdf.to_crs(CRS_METRICO)

    # 2. Crear celdas en metros
    gdf_m["geometry"] = gdf_m.geometry.apply(
        lambda p: create_pixel_box_metric(p, pixel_size_metros)
    )

    # 3. Agregar puntos que caen en la misma celda
    gdf_m["cell_id"] = gdf_m.geometry.apply(lambda geom: geom.wkb_hex)

    grid = (
        gdf_m
        .groupby("cell_id", as_index=False)
        .agg({
            col_valor: "mean",
            "geometry": "first"
        })
    )

    grid = gpd.GeoDataFrame(
        grid,
        geometry="geometry",
        crs=CRS_METRICO
    )

    # 4. Reproyectar a EPSG:4326 para publicación
    grid = grid.to_crs(CRS_LONLAT)

    # 5. Esquema estable para reportes y cartografía
    grid["valor_indice"] = grid[col_valor]
    grid["integridad_simulada"] = grid[col_valor]

    # 6. Limpieza de columna técnica
    grid = grid.drop(columns=["cell_id"])

    # 7. Exportar GeoPackage con nombre de capa explícito
    grid.to_file(
        output_gpkg,
        layer=LAYER_NAME,
        driver="GPKG"
    )

    print(f"GeoPackage generado: {output_gpkg}")
    print(f"Capa: {LAYER_NAME}")
    print(f"CRS salida: {CRS_LONLAT}")
    print(f"Resolución usada en CRS métrico: {pixel_size_metros} m")
    print(f"Features: {len(grid)}")
    print(f"Bounds EPSG:4326: {grid.total_bounds}")


if __name__ == "__main__":
    main()