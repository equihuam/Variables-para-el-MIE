# Operación de Snakemake

## 1. Requisitos
- entorno qgis_env activo
- estructura de directorios creada
- archivo config/config.yaml revisado

## 2. Validación previa
- verificar rutas de entrada
- verificar que existe reg_unidas
- verificar permisos de escritura

## 3. Ejecución básica
snakemake -n
snakemake --cores 1

## 4. Ejecución con número de núcleos
snakemake --cores 4

## 5. Forzar reejecución
snakemake --forcerun batimetria --cores 1

## 6. Limpiar productos
explicar qué se limpia manualmente y qué no

## 7. Logs
ubicación de logs por regla

## 8. Integración con pytest
pytest -q
pytest tests/test_erosion_integration.py -q

## 9. Problemas frecuentes
- rutas Windows/WSL
- CRS faltante
- shapefile sin campo esperado
- bloqueo de archivos