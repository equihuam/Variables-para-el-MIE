**Fecha:** 2026-04-11
**Código sugerido:** DOC / WORKFLOW / ENTORNO
**Tema:** Adaptación de scripts geoespaciales a `qgis_env` para ejecución headless

### Objetivo

Establecer un criterio común para adaptar scripts geoespaciales originalmente escritos con componentes interactivos de PyQGIS a una modalidad de ejecución **sin interfaz gráfica**, apta para integrarse a un workflow no supervisado dentro de `qgis_env`.

### Cambio realizado

Se revisó y refactorizó un conjunto de scripts que dependían, en diverso grado, de elementos asociados al uso interactivo de QGIS Desktop, tales como carga de capas al proyecto, uso de `QgsProject`, `QgsRasterLayer`, `QgsVectorLayer` y llamadas a `processing.run(...)`.

La adaptación consistió en sustituir esas dependencias por herramientas disponibles dentro del mismo entorno virtual, priorizando implementaciones más directas y robustas con:

* `gdal`
* `ogr`
* `osr`
* `xarray`, `rioxarray`, `geopandas` y `dask` se consideran *disponibles*, como criterio general para scripts futuros,cuando el tipo de insumo o la escala del procesamiento lo justifiquen.

### Criterio técnico adoptado

El principio general definido fue trabajar con un enfoque **headless-first**:

* evitar dependencias de GUI o de una sesión abierta de QGIS Desktop;
* conservar el uso del entorno `qgis_env` como base unificada de ejecución;
* aprovechar las bibliotecas geoespaciales ya instaladas en el entorno;
* mantener la lógica metodológica original de cada script, modificando sólo la capa de implementación.

En términos operativos, se consolidó el siguiente patrón:

1. usar `reg_unidas` como raster plantilla para obtener:

   * CRS,
   * resolución,
   * extensión,
   * dimensiones de la malla;

2. reproyectar, recortar o alinear insumos directamente sobre esa plantilla;

3. usar el método de remuestreo según el tipo de variable:

   * `near` para datos categóricos,
   * `bilinear` para variables continuas;

4. definir explícitamente valores `NoData`;

5. estructurar los scripts para ejecución batch mediante:

   * validación de entradas,
   * escritura controlada de salidas,
   * limpieza opcional de temporales,
   * función `main()`,
   * manejo explícito de errores.

### Justificación

La refactorización responde a la necesidad de contar con scripts más robustos, reproducibles y fáciles de integrar a un workflow automatizado. En varios casos, las dependencias de PyQGIS interactivo no aportaban ventajas reales al procesamiento, pero sí introducían fragilidad o complejidad innecesaria para ejecución por lote.

El cambio también ayuda a separar con mayor claridad dos niveles de trabajo:

* **procesamiento analítico**, que puede ejecutarse de forma headless;
* **visualización y ensamblaje cartográfico**, que puede reservarse para scripts específicamente orientados a QGIS Desktop o al cierre cartográfico.

### Alcance

Este criterio quedó aplicado a una primera tanda de scripts correspondientes a:

* alineación y enmascaramiento de raster categórico;
* procesamiento de variables de viento derivadas de NetCDF;
* rasterización de shapefiles temáticos sobre la malla de `reg_unidas`;
* recorte y alineación de batimetría usando máscara vectorial.

### Implicaciones para el workflow

Este ajuste sienta una base metodológica para la migración progresiva de scripts heredados hacia un esquema más automatizable. También facilita una futura integración con herramientas de orquestación del flujo, por ejemplo Snakemake u otros sistemas equivalentes.

Como criterio de desarrollo futuro, se recomienda que los nuevos scripts se escriban desde el inicio en modalidad **headless**, usando PyQGIS sólo cuando exista una ventaja específica que no pueda resolverse de forma más simple con las bibliotecas ya disponibles en `qgis_env`.

### Resultado

Se definió un patrón reusable para adaptar scripts geoespaciales a un entorno de ejecución no supervisado, preservando consistencia metodológica y mejorando la robustez del workflow.

Si quieres, el siguiente paso puede ser que te la deje en formato todavía más compacto, como entrada breve de bitácora de 1 párrafo largo más viñetas mínimas.
