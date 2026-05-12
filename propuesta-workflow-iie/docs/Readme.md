---
title: "Definición de requerimientos y caso de uso"
subtitle: "Producción de un estimador del Índice de Integridad Ecosistémica a partir de datos geográficos vectoriales y raster"
format:
  html:
    toc: true
    number-sections: true
  pdf:
    toc: true
    number-sections: true
  docx: default
---

# Propósito

Este documento define el caso de uso, los requerimientos funcionales y no funcionales, y la arquitectura lógica de un flujo de trabajo para producir un **estimador espacial del Índice de Integridad Ecosistémica (IIE)** a partir de datos geográficos vectoriales y raster. El flujo está concebido como una tubería reproducible, escalable y preferentemente **headless-first**, con capacidad de integración en un sistema de orquestación como **Snakemake**.

El objetivo final del proceso es generar, por región de análisis, un conjunto de productos congruentes entre sí y trazables, culminando en la producción de **mapas raster del índice de integridad ecosistémica** y sus insumos tabulares intermedios.

# Caso de uso

## Nombre del caso de uso

**Producción de un estimador espacial del Índice de Integridad Ecosistémica**

## Descripción general

A partir de una colección heterogénea de insumos geográficos —capas vectoriales, coberturas raster, tablas derivadas y salidas de un modelo probabilístico o red bayesiana— se requiere construir una representación espacialmente congruente del Índice de Integridad Ecosistémica para un conjunto de regiones costeras.

El proceso debe:

1. definir una geometría de referencia común por región;
2. producir coberturas raster congruentes para múltiples variables ambientales y antrópicas;
3. convertir dichas coberturas a series tabulares congruentes por píxel;
4. ensamblar una tabla maestra de entrenamiento o inferencia;
5. usar un motor de modelado probabilístico para producir valores estimados del índice;
6. reconstruir los mapas raster finales del índice a partir de dichas predicciones.

## Actores

### Actor principal

- **Equipo técnico del proyecto**: desarrolla, ejecuta, valida y ajusta el flujo de trabajo.

### Actores secundarios

- **Motor de inferencia bayesiana**: Netica u otro sistema equivalente que procese la tabla preparada y devuelva predicciones por observación/píxel.
- **Sistema de orquestación**: Snakemake u otro administrador de flujos que ejecute reglas, dependencias y paralelización.
- **Entorno SIG de validación**: QGIS u otra herramienta para inspección cartográfica de resultados.

## Disparador

El proceso se ejecuta cuando se dispone de:

- insumos geográficos base;
- definición de regiones de análisis;
- reglas para generar variables derivadas;
- una especificación del modelo probabilístico o una salida de inferencia ya producida.

## Resultado esperado

El flujo produce:

- una colección de **rasters de referencia** por región;
- una colección de **features tabulares congruentes** por región o globales;
- una **tabla final integrada** de variables por píxel;
- una **salida de inferencia** del índice;
- un conjunto de **GeoTIFF finales del Índice de Integridad Ecosistémica**;
- opcionalmente, productos auxiliares como histogramas, tablas CSV y proyectos QGIS.

# Alcance

## Incluye

- preparación de geometrías de referencia;
- generación de variables espaciales a partir de insumos vectoriales y raster;
- integración tabular de variables congruentes;
- preparación de tablas para modelado;
- reconstrucción cartográfica del índice final;
- validación estructural y pruebas automáticas del flujo.

## No incluye

- diseño conceptual del índice ecológico en sí;
- definición científica de cada variable o de sus umbrales;
- captura primaria de datos de campo;
- desarrollo de interfaces gráficas como requisito principal;
- operación manual interactiva como mecanismo central del flujo.

# Objetivo general

Implementar un flujo reproducible y escalable que permita estimar y mapear el Índice de Integridad Ecosistémica a partir de múltiples fuentes geoespaciales, preservando la congruencia espacial entre todas las variables y asegurando trazabilidad desde los insumos hasta el raster final.

# Objetivos específicos

1. Definir una **malla o plantilla raster de referencia** por región de análisis.
2. Generar variables espaciales derivadas a partir de insumos geográficos heterogéneos.
3. Convertir las variables espaciales en **series tabulares congruentes por píxel**.
4. Construir una tabla final apta para entrenamiento, inferencia o exportación a motores externos.
5. Integrar la salida del modelo probabilístico con la geometría espacial de referencia.
6. Reconstruir el mapa final del índice como raster por región.
7. Garantizar reproducibilidad, escalabilidad y validación automatizada.

# Supuestos del proceso

- Existe una partición territorial en regiones de análisis.
- Cada región cuenta o puede contar con un raster de referencia que fija CRS, resolución, extensión y alineación.
- Las variables derivadas pueden expresarse finalmente de forma raster o tabular por píxel.
- La congruencia espacial entre variables es un requisito central.
- La salida del modelo probabilístico preserva el orden o la clave necesaria para reasignar predicciones a los píxeles correctos.
- Los productos intermedios pueden almacenarse temporalmente en disco en formatos serializados como `PKL`, `CSV` o `GeoTIFF`.

# Requerimientos funcionales

## RF-01. Preparación de polígonos o regiones de referencia

El sistema debe permitir preparar una colección de regiones o polígonos de análisis a partir de insumos vectoriales, incluyendo operaciones como reproyección, corrección de CRS y buffering cuando el flujo lo requiera.

## RF-02. Generación de rasters de referencia por región

El sistema debe generar, para cada región, un raster de referencia que defina la malla espacial canónica del análisis, incluyendo:

- resolución;
- extensión;
- alineación;
- sistema de coordenadas;
- nodata.

## RF-03. Producción de coberturas raster congruentes

El sistema debe permitir generar coberturas raster congruentes por región a partir de distintos tipos de datos de entrada, incluyendo:

- datos vectoriales rasterizados;
- datos raster reproyectados y recortados;
- interpolaciones o transformaciones espaciales.

## RF-04. Conversión de coberturas a series numéricas congruentes

El sistema debe convertir las coberturas congruentes a tablas o series numéricas por píxel, preservando al menos:

- identificador regional;
- identificador de píxel;
- coordenadas `x`, `y`;
- una o más variables temáticas.

## RF-05. Almacenamiento serializado intermedio

Para las tablas intermedias y finales, se recomienda adoptar **Parquet** como formato base de almacenamiento tabular, por las siguientes razones:

- mayor eficiencia de lectura y escritura;
- mejor compresión;
- mejor soporte para tablas grandes;
- compatibilidad con particionamiento por región o variable;
- mejor integración con workflows escalables.

## RF-06. Integración de features en una tabla maestra

El sistema debe ensamblar una tabla maestra a partir de múltiples productos serializados congruentes, preservando la correspondencia espacial entre observaciones.

## RF-07. Preparación de datos para inferencia bayesiana

El sistema debe producir una tabla de entrada apta para ser consumida por un motor de inferencia bayesiana o un proceso equivalente, incluyendo discretización y codificación cuando sea necesario.

## RF-08. Integración de predicciones del modelo

El sistema debe poder leer una salida de inferencia generada por Netica o por otro motor compatible y asociarla correctamente con los píxeles de la tabla base.

## RF-09. Reconstrucción raster del índice final

El sistema debe reconstruir, por región, el raster final del Índice de Integridad Ecosistémica usando:

- la tabla base espacialmente congruente;
- las predicciones del modelo;
- el raster de referencia regional.

## RF-10. Exportación de productos cartográficos finales

El sistema debe generar productos finales al menos en formato:

- `GeoTIFF` por región;

y opcionalmente:

- histogramas PNG;
- tablas CSV auxiliares;
- proyectos QGIS para inspección.

## RF-11. Trazabilidad del flujo

El sistema debe permitir rastrear qué insumos y qué pasos generaron cada producto final o intermedio.

## RF-12. Ejecución automatizada por reglas

El sistema debe permitir la ejecución automatizada de los pasos mediante reglas y dependencias explícitas, preferentemente con Snakemake.

# Requerimientos no funcionales

## RNF-01. Reproducibilidad

El flujo debe producir resultados reproducibles a partir de la misma configuración, insumos y versión de scripts.

## RNF-02. Modularidad

Cada componente debe corresponder a una etapa funcional clara, con entradas y salidas explícitas.

## RNF-03. Escalabilidad

El flujo debe permitir crecer en número de variables y regiones sin rediseño estructural.

## RNF-04. Paralelización

Siempre que sea viable, las ramas por variable o por región deben poder ejecutarse en paralelo.

## RNF-05. Trazabilidad de errores

Las fallas deben poder localizarse a nivel de regla, script, región o variable.

## RNF-06. Validación automatizada

El proyecto debe contar con pruebas automatizadas que validen componentes críticos del flujo y contratos de datos.

## RNF-07. Portabilidad razonable

El flujo debe ejecutarse en entornos Python reproducibles, idealmente mediante entornos controlados y dependencias explícitas.

## RNF-08. Headless-first

El flujo debe funcionar sin depender de interfaces gráficas, salvo para validación o inspección posterior.

# Arquitectura lógica del proceso

## Etapa 1. Geometría de referencia

Se construyen las regiones y sus plantillas raster de referencia. Esta etapa fija la congruencia espacial del resto del proceso.

**Producto principal:** `ref_grid.tif` por región.

## Etapa 2. Generación de variables espaciales

Se generan capas raster o equivalentes para variables derivadas a partir de datos vectoriales o raster.

**Producto principal:** coberturas raster congruentes o datos preparados para su conversión tabular.

## Etapa 3. Conversión a series congruentes

Cada variable se transforma a una tabla de observaciones por píxel con las mismas llaves espaciales.

**Producto principal:** archivos serializados por variable, idealmente con columnas como `regionid`, `pixid`, `x`, `y` y la variable temática.

## Etapa 4. Integración tabular

Se ensamblan todas las variables en una sola tabla maestra.

**Producto principal:** tabla integrada de entrenamiento o inferencia.

## Etapa 5. Modelado e inferencia

Se prepara la tabla final para el motor bayesiano, se ejecuta la inferencia y se obtienen predicciones por observación.

**Producto principal:** tabla o archivo de predicciones por píxel.

## Etapa 6. Reconstrucción cartográfica final

Las predicciones del índice se reinsertan sobre la geometría raster de referencia y se generan los mapas finales por región.

**Producto principal:** raster final del Índice de Integridad Ecosistémica.

# Contrato mínimo de datos intermedios

## Contrato para tablas de features

Toda tabla serializada de features debería incluir como mínimo:

- `regionid`
- `pixid`
- `x`
- `y`

más una o varias columnas temáticas.

## Contrato para rasters congruentes

Todo raster congruente debe ser compatible con el raster de referencia regional en:

- CRS;
- resolución;
- `transform`;
- `shape`;
- nodata;
- extensión o máscara válida.

# Reglas de integridad del flujo

1. Ninguna variable debe perder su correspondencia espacial con la plantilla regional.
2. La unión entre variables debe preservar una llave espacial consistente.
3. La reconstrucción raster final debe usar la misma plantilla de referencia que originó la congruencia del resto del análisis.
4. Los productos intermedios deben ser suficientemente explícitos para permitir depuración y recomputación parcial.
5. Las salidas del modelo probabilístico deben poder vincularse sin ambigüedad con la tabla base.

# Estrategia de implementación recomendada

## Orquestación

Se recomienda implementar el flujo mediante **Snakemake**, organizando reglas por capas:

- `reference`
- `feature_tables`
- `training_table`
- `bayes_model`
- `final_maps`

## Desarrollo de scripts

Se recomienda que cada script:

- reciba rutas parametrizadas;
- pueda ejecutarse por región cuando sea razonable;
- tenga entradas y salidas explícitas;
- no dependa de rutas hardcodeadas como única opción operativa.

## Pruebas automatizadas

Se recomienda una batería de pruebas en **pytest** en tres niveles:

### Unitarias
Para funciones puras:
- normalización;
- discretización;
- construcción de llaves;
- ordenamiento de regiones;
- validación de contratos.

### Integración por script
Para mini-casos sintéticos:
- creación de `ref_grid`;
- rasterización simple;
- generación de `.pkl`;
- ensamblado tabular.

### Integración de workflow
Para un DAG mínimo:
- una región;
- un subconjunto pequeño de variables;
- ensamblado final;
- reconstrucción raster del índice.

# Riesgos y puntos críticos

## Riesgo 1. Desalineación espacial
Si una variable no respeta el raster de referencia, toda la integración posterior puede quedar sesgada.

## Riesgo 2. Dependencia implícita del orden de filas
El flujo requiere especial cuidado cuando un motor externo devuelve predicciones por fila. Debe preservarse una correspondencia inequívoca entre observación y píxel.

## Riesgo 3. Inconsistencia de nombres y columnas
El ensamblado final puede fallar si no existe una convención estable de nombres para archivos y columnas.

## Riesgo 4. Divergencias entre implementaciones R y Python
Algunas operaciones pueden no tener equivalencia exacta entre bibliotecas. Eso debe documentarse explícitamente y validarse empíricamente cuando haya datos disponibles.

# Criterios de aceptación

El flujo se considerará funcionalmente aceptable cuando cumpla al menos lo siguiente:

1. genera un raster de referencia válido por región;
2. genera variables intermedias congruentes y serializadas;
3. construye una tabla final con una fila por píxel y una columna por variable;
4. acepta o produce una salida de inferencia por píxel;
5. reconstruye correctamente un GeoTIFF final del índice por región;
6. permite reejecutar parcialmente etapas del flujo;
7. cuenta con pruebas automatizadas mínimas para validar componentes críticos.

# Resultado final esperado

El resultado final esperado del workflow es un conjunto de **mapas raster regionales del Índice de Integridad Ecosistémica**, derivados de una tubería reproducible que integra datos geográficos vectoriales y raster, variables espaciales congruentes, una tabla maestra por píxel y una etapa de inferencia probabilística.

Dicho resultado debe ser trazable, reproducible, escalable y suficientemente estable para sostener tanto análisis posteriores como validación cartográfica y modelado adicional.


# Hallazgos operativos de la fase de ensayos en laptop

La fase de ensayos sobre una laptop Windows con 16 GB de RAM permitió validar la arquitectura general del workflow, detectar defectos lógicos importantes en la traducción inicial de scripts R a Python y ajustar decisiones operativas relevantes para su ejecución estable.

## Hallazgo 1. La congruencia por píxel debe construirse solo sobre celdas válidas

El principal defecto lógico detectado en la traducción inicial de algunos scripts Python fue que, después de reproyectar el `ref_grid` regional, se estaban convirtiendo **todas las celdas** de la grilla reproyectada a observaciones tabulares, incluyendo celdas sin datos o fuera de la máscara útil.

Eso provocó una inflación artificial severa en las tablas de features. En una región de prueba se observaron del orden de más de cien millones de filas cuando la malla válida real contenía solo unos cientos de miles de celdas útiles.

### Decisión adoptada

A partir de esta observación, los scripts de features deben:

1. reproyectar el raster de referencia;
2. identificar primero las celdas válidas;
3. calcular coordenadas `x`, `y` solo para esas celdas válidas;
4. exportar únicamente esas observaciones.

### Implicación

La tabla tabular congruente por píxel no debe entenderse como “todas las celdas del raster reproyectado”, sino como “todas las celdas válidas de la malla útil regional”.

## Hallazgo 2. No debe materializarse la grilla completa antes del filtrado

En regiones grandes, incluso el paso intermedio de construir coordenadas para toda la grilla reproyectada resultó inviable por memoria.

### Decisión adoptada

Los scripts de features deben evitar patrones de este tipo:

- creación de `meshgrid` completo;
- transformación a coordenadas para toda la matriz;
- filtrado posterior.

En su lugar, deben trabajar así:

1. detectar índices de celdas válidas;
2. generar coordenadas solo para esas posiciones;
3. construir la tabla final directamente a partir de ese subconjunto.

Esto volvió viable la ejecución regional incluso en regiones con grillas reproyectadas muy grandes.

## Hallazgo 3. La organización de `raw/` como carpeta plana no escala bien

Durante los ensayos operativos se comprobó que una carpeta `raw/` plana se vuelve rápidamente confusa y poco manejable, tanto por mezcla de tipos de archivo como por dificultad para ubicar insumos base.

### Decisión adoptada

Se recomienda organizar el repositorio de datos por subcarpetas temáticas o por fuente, por ejemplo:

```text
raw/
  dunes_inegi/
  dunes_cost/
  coastal_regions/
  erosion/
  corals/
external/
  netica/
results/
  reference/
  features/
  training/
  final_maps/
```

Esta estructura debe reflejarse explícitamente en `config.yaml`, evitando rutas hardcodeadas dentro de los scripts.

## Hallazgo 4. En Windows conviene usar una raíz corta del repositorio de datos

Durante la integración con Snakemake en Windows se observaron fricciones asociadas a rutas largas, espacios en nombres y metadatos internos del workflow.

### Decisión adoptada

Se adoptó una raíz operativa corta para el repositorio de datos:

```text
C:/wf-ie-data
```

La ubicación física real de los datos puede estar en otra ruta, incluso sincronizada por nube, pero operativamente conviene usar una ruta corta y estable, idealmente mediante `symlink` cuando haga falta.

## Hallazgo 5. La tabla maestra consolidada sí es viable una vez corregidos los features

En una fase intermedia se ensayó una estrategia de ensamblado regional por temor a exceder memoria. Esa decisión fue útil para aislar el problema, pero una vez corregida la inflación de filas en los scripts de features, volvió a ser viable producir una tabla maestra consolidada.

### Decisión adoptada

Se conserva el enfoque:

- features regionales por variable;
- ensamblado final consolidado en `master_features.parquet`.

Internamente, el ensamblado sigue haciéndose por región para mayor robustez, pero el producto esperado del workflow vuelve a ser una tabla maestra global.

## Hallazgo 6. La salida del motor bayesiano debe incorporarse por compaginación posicional

La salida externa de Netica no necesariamente contiene llaves espaciales suficientes para hacer un `merge` relacional seguro.

### Decisión adoptada

Mientras la salida de inferencia consista en una secuencia de predicciones sin llaves explícitas, la integración debe hacerse por **correspondencia posicional**, bajo estas condiciones:

1. la tabla de entrada a Netica debe preservar un orden estable;
2. la salida de predicciones debe tener exactamente el mismo número de filas;
3. no deben introducirse reordenamientos intermedios entre exportación e integración.

Esto implica que la vinculación entre tabla base y predicciones no debe depender de un `merge` clásico, sino de congruencia en longitud y orden.

## Hallazgo 7. La validación del vertical mínimo fue exitosa y escaló bien a 14 regiones

Se realizó una fase de ensayo inicial con 3 regiones y posteriormente una corrida con las 14 regiones previstas. Una vez corregidos los scripts de features, la tabla `master_features.parquet` quedó en un orden de magnitud prácticamente idéntico al del flujo original en R.

### Implicación

Esto constituye una validación fuerte de que la traducción Python:

- ya preserva de forma razonable la lógica espacial del flujo original;
- ya produce un número de observaciones congruente con el flujo R;
- y ya puede sostenerse como base del workflow reproducible en Snakemake.

## Hallazgo 8. La laptop es útil para depuración, pero la plataforma objetivo debe ser más robusta

La laptop permitió encontrar errores lógicos importantes y validar la arquitectura del workflow, pero también mostró límites claros de memoria y rendimiento para ciertas regiones o pasos costosos.

### Decisión adoptada

Se distinguen dos contextos de operación:

#### Laptop Windows
Adecuada para:

- depuración;
- ensayos incrementales;
- validación de scripts;
- corridas parciales o regionales.

#### Equipo de escritorio con WSL
Adecuado para:

- corridas formales del workflow completo;
- mayor paralelización con Snakemake;
- validación final del pipeline;
- eventual crecimiento del número de variables o regiones.

## Recomendaciones derivadas para implementación

1. Mantener el enfoque **headless-first**.
2. Preservar rutas parametrizadas y evitar hardcodeo de ubicaciones.
3. Validar siempre el número de celdas válidas frente al tamaño total de la grilla reproyectada.
4. Favorecer productos regionales intermedios y consolidación final explícita.
5. Documentar claramente cuándo una integración es por llave y cuándo es por posición.
6. Usar la laptop como entorno de depuración y WSL como entorno preferente de ejecución completa.

# Bibliotecas requeridas

Para facilitar la operación se sugiere crear un ambiente virtual de trabajo con la especificación que se anota en seguida. Para mayires detalles vease el [texto guía](README_entorno_workflow_iie.qmd) que hemos preoarado para eso

``` yaml
name: workflow-iie
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - pyarrow
  - rasterio
  - geopandas
  - shapely
  - pyproj
  - pyogrio
  - scipy
  - scikit-learn
  - matplotlib
  - snakemake
  - pytest
  - pip
  - pip:
      - pgmpy
```