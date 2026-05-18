# Proyecto workflow-iie

Flujo de trabajo para el preprocesamiento, integración, validación y organización de variables empleadas en el **Modelo de Integridad Ecosistémica (3T-EII)**.

## 1. Propósito del proyecto

`workflow-iie` reúne scripts, configuraciones, documentación y utilidades para construir un flujo reproducible de preparación de variables espaciales y tabulares asociadas al Modelo de Integridad Ecosistémica.

El proyecto está orientado a:

- integrar insumos raster, vectoriales, tabulares y climáticos;
- estandarizar procesos de alineación espacial, extracción de variables y consolidación de salidas;
- traducir y consolidar scripts heredados de R y PyQGIS hacia una arquitectura coherente en Python;
- facilitar validación formal, trazabilidad y automatización;
- preparar una base técnica estable para análisis posteriores, modelado e integración cartográfica.

## 2. Enfoque general

La arquitectura del proyecto sigue un criterio **headless-first** para el procesamiento analítico. Esto significa que la transformación, extracción, rasterización, alineación y validación de datos se implementan preferentemente mediante scripts Python reproducibles, sin depender de una sesión interactiva de QGIS Desktop.

El proyecto distingue dos capas complementarias:

### 2.1. Capa analítica

Ubicada principalmente en `scripts/`, `workflow/`, `config/`, `tests/` y `results/`.

Esta capa se encarga de:

- procesamiento geoespacial reproducible;
- integración de variables;
- generación de salidas raster, vectoriales y tabulares;
- validación de resultados;
- ejecución por lotes y trazabilidad del flujo.

### 2.2. Capa de visualización

Ubicada en `visualizar/`.

Esta capa está destinada al ensamblaje cartográfico y a la preparación de productos visuales. No sustituye el procesamiento analítico, sino que consume sus salidas para:

- consolidar capas finales en contenedores como GeoPackage;
- construir proyectos QGIS (`.qgz`) listos para revisión visual;
- organizar capas, estilos y agrupaciones cartográficas;
- preparar, en etapas posteriores, mapas exportables en formatos como PNG, PDF o SVG.

La separación entre análisis y visualización permite mantener el procesamiento desacoplado de decisiones cartográficas o dependencias gráficas.

## 3. Principios de diseño

El desarrollo del proyecto sigue estos principios:

- **headless-first**: priorizar scripts que corran sin GUI;
- **reproducibilidad**: documentar entradas, salidas y supuestos metodológicos;
- **modularidad**: separar utilidades, scripts de proceso, reglas de workflow y visualización;
- **trazabilidad**: registrar cambios en bitácora y documentar decisiones técnicas;
- **validación formal**: incorporar pruebas con `pytest` y criterios explícitos de aceptación;
- **separación de responsabilidades**: distinguir entre procesamiento, validación, modelado y representación cartográfica.

## 4. Bibliotecas y criterio de uso

El proyecto se apoya en un entorno Python con bibliotecas geoespaciales y científicas. De forma general, se adopta el siguiente criterio:

- **GDAL / OGR / OSR**: reproyección, alineación raster, rasterización, lectura/escritura espacial base;
- **xarray / rioxarray / netCDF4**: manejo de NetCDF y cubos climáticos;
- **geopandas**: procesamiento vectorial cuando simplifica el flujo;
- **dask**: escalamiento y procesamiento por bloques paralelizados cuando el volumen de cómputo lo requiere;
- **pandas / numpy / scipy / statsmodels / scikit-learn**: manejo tabular, modelado auxiliar, imputación e integración analítica;
- **qgis.core**: solo cuando aporta una ventaja clara, preferentemente en la fase final de ensamblaje cartográfico;
- **Snakemake**: orquestación del workflow;
- **pytest**: validación formal del código y de productos intermedios.

## 5. Convenciones metodológicas

A nivel de procesamiento geoespacial, el proyecto adopta varias convenciones comunes:

- `reg_unidas` funciona como raster plantilla de referencia para:
  - CRS,
  - resolución,
  - extensión,
  - dimensiones de malla;

- los datos **categóricos** se reproyectan o remuestrean con `near`;
- las variables **continuas** se reproyectan o remuestrean con `bilinear`;
- los valores `NoData` deben declararse explícitamente;
- los scripts deben incluir:
  - funciones reutilizables,
  - `main()`,
  - validación de entradas y salidas,
  - manejo explícito de errores,
  - limpieza opcional de temporales;

- cuando existan supuestos metodológicos todavía no verificados, se registran como `TODO` en código y como pendiente en bitácora.

## 6. Estructura general del proyecto

La estructura actual del proyecto incluye componentes para configuración, datos, documentación, scripts, pruebas, workflow y visualización. Entre los elementos ya definidos se encuentran `config/`, `data/`, `docs/`, `envs/`, `results/`, `scripts/`, `tests/`, `visualizar/` y `workflow/`, además del `Snakefile` y archivos README complementarios.

De manera resumida:

- `config/`: parámetros y rutas configurables;
- `data/`: insumos `raw`, productos intermedios, procesados y datos de prueba;
- `docs/`: bitácora, criterios de validación y operación;
- `envs/`: definición del entorno computacional;
- `results/`: salidas raster, vectoriales, tabulares y de validación;
- `scripts/`: scripts de procesamiento, incluyendo refactorizaciones headless y traducciones R → Python;
- `tests/`: pruebas unitarias e integración;
- `workflow/`: reglas de Snakemake y logs;
- `visualizar/`: ensamblaje cartográfico y proyectos QGIS.

## 7. Organización actual de scripts

Actualmente el proyecto contiene, dentro de `scripts/`, dos subcarpetas de trabajo principales:

- `headless-qgis/`
- `headless-r2py/`
- `R2Py_julian/`

Estas carpetas responden a la historia reciente de refactorización del proyecto:

- `headless-qgis/` agrupa scripts originalmente asociados a procesos geoespaciales que dependían, en alguna medida, de componentes de PyQGIS o de flujos cercanos a QGIS, y que fueron adaptados a ejecución headless;
- `headless-r2py/` agrupa scripts traducidos desde R hacia Python, manteniendo su lógica analítica y adaptándolos a una arquitectura reproducible y sin GUI.
- `R2Py_julian/`  agrupa scripts de Julián traducidos desde R hacia Python, manteniendo su lógica analítica y adaptándolos a una arquitectura reproducible y sin GUI.

Esta separación **es transitoria y organizativa**, no conceptual. Aún debe discutirse con el equipo cómo articular ambos conjuntos, pero la meta es que formen parte de **un solo conjunto coherente de scripts Python**, orquestado mediante Snakemake y documentado bajo criterios comunes.

En consecuencia, la organización futura tenderá a:

- homogenizar nombres, dependencias y convenciones;
- reducir duplicación de utilidades;
- centralizar funciones comunes;
- y exponer los procesos como una sola capa analítica unificada del workflow.

## 8. Documentación del proyecto

La documentación se distribuye en distintos niveles:

- **README.md**: visión general, arquitectura y propósito del proyecto;
- **README-headless.md**: criterios técnicos de desarrollo y ejecución sin GUI;
- **docs/bitacora.csv**: registro cronológico compacto de cambios técnicos;
- **docs/criterios-validacion.md**: reglas y criterios de validación;
- **docs/operacion-snakemake.md**: instrucciones de operación del workflow.


## Producción de mapa raster de IIE de la salida de Netica

Se añadió un script en Python headless para generar rasters regionales de **IE** a partir de tablas de entrenamiento, predicciones y mallas de referencia. La implementación reemplaza una versión previa en R y utiliza `pandas`, `numpy` y `rasterio` para construir salidas GeoTIFF reproducibles y aptas para integración en flujos automatizados.

Durante la migración se corrigieron problemas de lectura de insumos, correspondencia entre regiones y mallas de referencia, desajustes de CRS y artefactos espaciales derivados de la asignación directa de puntos reproyectados a la grilla final. La solución adoptada consiste en construir primero una grilla fuente en el CRS original de los datos y reproyectarla posteriormente a la plantilla regional con remuestreo por vecino más cercano. El módulo cuenta además con pruebas unitarias en `pytest` para validar su comportamiento básico.


## 9. Bitácora técnica

El proyecto utiliza una bitácora compacta en formato CSV para registrar ajustes relevantes del workflow.

Campos base de la bitácora:

- `fecha`
- `codigo`
- `modulo`
- `tipo`
- `objetivo`
- `cambio_realizado`
- `justificacion`
- `resultado`
- `implicaciones`
- `pendientes`
- `estado`
- `referencia`

Categorías sugeridas para `tipo`:

- `refactor`: cambios en implementación o estructura sin alterar el objetivo analítico;
- `validacion`: incorporación o ajuste de criterios de verificación;
- `documentacion`: mejora de README, notas técnicas o instrucciones;
- `workflow`: cambios en la organización general del flujo;
- `snakemake`: reglas, configuración y operación del pipeline;
- `entorno`: ajustes de bibliotecas, compatibilidad y configuración del entorno;
- `pruebas`: desarrollo de pruebas con `pytest` u otros mecanismos formales.

## 10. Workflow y automatización

La ejecución reproducible del proyecto se articula mediante **Snakemake**, con apoyo de:

- `Snakefile`
- `workflow/rules/*.smk`
- `config/config.yaml`
- `docs/operacion-snakemake.md`

La meta es que cada etapa del procesamiento quede explícitamente definida en términos de:

- entradas,
- salidas,
- dependencias,
- logs,
- criterios de reejecución.

## 11. Validación y pruebas

El proyecto incorpora una capa de validación formal con `pytest`, pensada para cubrir:

- pruebas unitarias de funciones reutilizables;
- pruebas de integración sobre datos pequeños;
- verificaciones mínimas end-to-end de productos clave.

La carpeta `tests/` ya está contemplada para esta función, aunque su contenido aún está en fase de consolidación.

## 12. Ensamblaje cartográfico y salidas visuales

La carpeta `visualizar/` se concibe como la capa de ensamblaje final para revisión y comunicación cartográfica. Allí podrán concentrarse scripts y recursos para:

- consolidar salidas del workflow en GeoPackage;
- construir proyectos QGIS listos para abrir en GUI;
- aplicar estilos (`qml/`);
- definir plantillas de salida;
- preparar exportaciones presentables en PNG, PDF o SVG.

En este diseño, el procesamiento analítico sigue siendo headless y automatizable, mientras que la construcción de productos de presentación se resuelve en una capa separada y explícita.

## 13. Estado actual

El proyecto se encuentra en una fase de consolidación de arquitectura. Entre los avances recientes se incluyen:

- refactorización de scripts geoespaciales hacia modo headless;
- traducción progresiva de scripts R a Python;
- definición de una bitácora técnica estructurada;
- diseño inicial del workflow con Snakemake;
- incorporación de una capa diferenciada de visualización;
- preparación de una base más formal para pruebas y documentación.

## 14. Desarrollo futuro

Líneas de trabajo previstas:

- consolidar el conjunto de scripts Python en una estructura más homogénea;
- completar reglas de Snakemake y su documentación operativa;
- fortalecer la capa de pruebas con `pytest`;
- formalizar criterios de validación para entradas, salidas y supuestos metodológicos;
- enriquecer la carpeta `visualizar/` para generar proyectos QGIS y productos cartográficos exportables;
- preparar una capa futura para procesamiento basado en modelos bayesianos previamente desarrollados y entrenados.

## 15. Varios

- Existen archivos y componentes aún en estado inicial o como esqueleto de trabajo.
- Algunos nombres de archivos y carpetas todavía pueden ajustarse para mejorar consistencia.
- Parte del contenido actual refleja una transición desde scripts heredados en R y PyQGIS hacia una arquitectura unificada en Python.
- En etapas posteriores convendrá normalizar convenciones de nombres, dependencias y rutas internas del proyecto.
- Falta construir una capa específica para el uso operativo de **redes bayesianas** dentro del workflow. Esta capa no estará orientada principalmente al entrenamiento desde cero dentro de este proyecto, sino al **cómputo eficiente sobre modelos bayesianos previamente desarrollados y entrenados**, con el objetivo de tomar esos modelos y generar **capas geoespaciales derivadas** a partir de las variables procesadas y del modelo proporcionado.
