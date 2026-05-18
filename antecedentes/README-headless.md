# README-headless

## 1. Propósito

Breve explicación del subproyecto y del objetivo de migrar scripts geoespaciales a ejecución headless.

## 2. Alcance

Qué procesos cubre:
- raster alineado a plantilla
- reproyección
- rasterización
- procesamiento NetCDF
- recorte con máscara
- exportación tabular/vectorial

## 3. Principios de diseño

- headless-first
- reproducibilidad
- separación entre análisis y visualización
- alineación estricta a raster plantilla
- validación formal con pytest

## 4. Entorno computacional

- nombre del entorno: qgis_env
- bibliotecas disponibles
- versión recomendada de Python
- observaciones de compatibilidad

## 5. Bibliotecas y criterio de uso
 
- GDAL/OGR/OSR: raster, reproyección, rasterización
- xarray/rioxarray/netCDF4: NetCDF y cubos climáticos
- dask: procesamiento escalable
- geopandas: vectores cuando simplifique
- qgis.core: solo si aporta valor real, sin GUI

## 6. Convenciones metodológicas

- raster plantilla: reg_unidas
- resample categórico: near
- resample continuo: bilinear
- definición explícita de NoData
- uso de main() y manejo de errores
- limpieza opcional de temporales

## 7. Estructura del proyecto

Resumen del árbol de directorios.

## 8. Ejecución

- ejecución directa de scripts
- ejecución con Snakemake
- dry-run
- logs
- rerun de pasos

## 9. Validación

- pruebas unitarias
- pruebas de integración
- criterios mínimos de aceptación

## 10. Limitaciones y pendientes

- datos muy grandes
- supuestos de CRS
- campos esperados en vectores
- mejoras futuras