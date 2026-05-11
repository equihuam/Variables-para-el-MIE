# Instalación de Python y del ambiente `qgis_env`

Esta guía está orientada a usuarios cuyo sistema operativo principal es **Windows 11**. En ese mismo equipo puede coexistir además un entorno Linux mediante **WSL** (*Windows Subsystem for Linux*).

La guía explica el procedimiento para trabajar en ambos entornos. Se cubre desde la instalación inicial de los componentes necesarios hasta la creación del ambiente **`qgis_env`**, que es el que estamos usando para el desarrollo del *workflow*. Como base se adopta **Python 3.11**, que actualmente es una opción estable para trabajar con la API de **QGIS**.

La base técnica es adecuada para nuestro objetivo: `micromamba` es el instalador oficial recomendado, no necesita un ambiente `base` con Python preinstalado, y `qgis` en conda-forge tiene *builds* actuales para **win-64** y **linux-64**. Además, para asegurar estabilidad con paquetes de la colección **conda-forge**, conviene usar **channel priority strict** y evitar mezclar canales. ([Mamba][1])

## 1) Archivo de entorno recomendado

Aunque no es indispensable, en esta guía optamos por usar un archivo `environment.yml` que puede emplearse tanto en Windows como en WSL. Esto ayuda a mantener ambos entornos alineados.

Para mayor estabilidad de QGIS, fijamos explícitamente el canal **`conda-forge/label/qgis_ltr`**, donde existen *builds* recientes de **QGIS 3.40.11** para **Python 3.11** en Windows y Linux. ([Anaconda][2])

Copia las líneas siguientes en un archivo y guárdalo con el nombre:

`environment-qgis_env.yml`

```yaml
name: qgis_env
channels:
  - conda-forge/label/qgis_ltr
  - conda-forge
dependencies:
  - python=3.11
  - qgis
  - xarray
  - geopandas
  - dask
  - pandas
  - snakemake
  - pytest
  - numpy
  - scipy
  - statsmodels
  - scikit-learn
  - matplotlib
````

## 1.1) Ubicación recomendada del archivo `environment-qgis_env.yml`

Aunque el contenido del archivo `environment-qgis_env.yml` puede ser el mismo en Windows y en WSL, la ubicación recomendada no es la misma en ambos casos.

### En Windows

Si el ambiente `qgis_env` se va a crear y usar en **Windows nativo**, conviene guardar el archivo en una carpeta normal de Windows, por ejemplo:

```text
C:\Users\TU_USUARIO\proyectos\qgis-win\
```

En ese caso, desde PowerShell se puede entrar a esa carpeta y crear el ambiente con:

```powershell
cd C:\Users\TU_USUARIO\proyectos\qgis-win
micromamba create -f .\environment-qgis_env.yml
```

### En WSL

Si el ambiente `qgis_env` se va a crear y usar en **WSL**, conviene guardar tanto el archivo YAML como el proyecto dentro del sistema de archivos Linux, por ejemplo:

```bash
~/proyectos/qgis-wsl/
```

En WSL, el símbolo `~` representa la carpeta personal del usuario, normalmente:

```bash
/home/tu_usuario
```

Así, una ubicación típica del archivo sería:

```bash
/home/tu_usuario/proyectos/qgis-wsl/environment-qgis_env.yml
```

Desde la terminal de WSL, bastaría con:

```bash
cd ~/proyectos/qgis-wsl
micromamba create -f environment-qgis_env.yml
```

### Acceso desde WSL a carpetas de Windows

WSL también puede acceder a archivos ubicados en discos de Windows. Por ejemplo, el disco `C:` aparece montado bajo:

```bash
/mnt/c/
```

De este modo, una carpeta como:

```text
C:\Users\TU_USUARIO\proyectos\qgis-win\
```

se verá en WSL como:

```bash
/mnt/c/Users/TU_USUARIO/proyectos/qgis-win/
```

Esto permite leer o copiar archivos entre ambos sistemas. Sin embargo, para el trabajo cotidiano del workflow, especialmente si se usarán `snakemake`, `geopandas`, `xarray`, `dask` y scripts *headless* (sin recurrir a la interfaz gráfica de QGIS), conviene trabajar dentro del sistema de archivos Linux y no sobre `/mnt/c/...`. ([Microsoft Learn][13])

## 2) Instalación en Windows 11

La documentación oficial indica que en **Windows PowerShell** la instalación recomendada de `micromamba` se hace con el script oficial siguiente. ([Mamba][1])

### 2.1. Abrir PowerShell

Abre **PowerShell** normal, o mejor **PowerShell 7** si lo usas habitualmente.

### 2.2. Instalar micromamba

Ejecuta:

```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)
```

Eso instala `micromamba`. Después conviene reiniciar la terminal para que tome correctamente el **PATH** y la inicialización del *shell*. La propia documentación también indica que `micromamba` usa `MAMBA_ROOT_PREFIX` para ubicar ambientes y caché. ([Mamba][1])

### 2.3. Inicializar PowerShell para activación cómoda

Si al reabrir la terminal no reconoce `micromamba activate`, ejecuta:

```powershell
micromamba shell init -s powershell -r $env:USERPROFILE\micromamba
```

**Cuidado:** esto debe hacerse una sola vez y sólo si es necesario, pues la operación modifica archivos de configuración del *shell*.

Luego cierra y vuelve a abrir PowerShell. La activación en Mamba/Micromamba depende del *shell* y por eso esta inicialización es importante. ([Mamba][3])

### 2.4. Configurar conda-forge como canal principal

Para reducir conflictos, usa prioridad estricta y deja `conda-forge` arriba. Eso está alineado con la recomendación de conda-forge y la documentación de conda. Observa que en este punto `micromamba` ya debería existir como comando válido en PowerShell. ([Conda-Forge][4])

```powershell
micromamba config append channels conda-forge
micromamba config set channel_priority strict
```

### 2.5. Crear el ambiente `qgis_env`

En la ventana de PowerShell, ubícate en la carpeta donde guardaste `environment-qgis_env.yml`. Por ejemplo:

```powershell
cd C:\Users\TU_USUARIO\proyectos\qgis-win
micromamba create -f .\environment-qgis_env.yml
```

El sistema resolverá dependencias y creará el ambiente aislado `qgis_env`, que contendrá Python 3.11 y las bibliotecas indicadas en el archivo YAML. Cuando termine, lo indicará en la misma terminal.

El contenido de este ambiente no estará disponible de manera general en el sistema: sólo podrá usarse cuando el ambiente sea activado.

### 2.6. Activar el ambiente

```powershell
micromamba activate qgis_env
```

### 2.7. Verificación básica

Este comando lanza Python, intenta cargar las bibliotecas solicitadas y, si todo funciona correctamente, imprimirá `OK`.

```powershell
python --version
python -c "import qgis, xarray, geopandas, dask, pandas, snakemake, pytest, numpy, scipy, statsmodels, sklearn, matplotlib; print('OK')"
```

### 2.8. Lanzar QGIS

Si quieres comprobar que QGIS quedó correctamente instalado en este ambiente, ejecuta:

```powershell
qgis
```

Esto abrirá la instalación de **QGIS** asociada a ese ambiente, independiente de otras instalaciones de QGIS que pudiera haber en el equipo.

También puedes probar sólo el módulo Python:

```powershell
python -c "from qgis.core import QgsApplication; print('QGIS Python OK')"
```

## 3) Instalación en WSL

Microsoft indica que en Windows 11 puedes instalar WSL con `wsl --install`, que las instalaciones nuevas quedan en **WSL 2** por defecto, y que **WSLg** permite ejecutar aplicaciones GUI Linux integradas en Windows. Eso significa que, si quieres, también puedes lanzar `qgis` con interfaz desde WSL en Windows 11. ([Microsoft Learn][5])

### 3.1. Instalar WSL en Windows 11

Desde **PowerShell como administrador**:

```powershell
wsl --install
```

Reinicia si te lo pide. Luego confirma que estás en WSL 2:

```powershell
wsl -l -v
```

Si hiciera falta forzar una distribución a WSL 2, por ejemplo `Ubuntu`, usa:

```powershell
wsl --set-version Ubuntu 2
```

Todo esto está documentado en la guía oficial de Microsoft. ([Microsoft Learn][5])

### 3.2. Entrar a Ubuntu

Abre tu terminal WSL/Ubuntu.

### 3.3. Instalar micromamba en WSL

La forma recomendada en Linux es el script oficial, aunque seguramente ya estará ahora un ícono en inicio con el pingüino de Linux: ([Mamba][1])

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Después cierra y vuelve a abrir la terminal WSL.

### 3.4. Inicializar bash

Si `micromamba` no queda activado automáticamente, ejecuta lo siguiente **una sola vez**:

```bash
micromamba shell init -s bash -r ~/micromamba
```

Luego:

```bash
source ~/.bashrc
```

La documentación de Mamba/Micromamba deja claro que la activación depende del *shell* y que `micromamba` puede usarse con sus subcomandos de *shell* para ello. ([Mamba][3])

### 3.5. Configurar canales

```bash
micromamba config append channels conda-forge
micromamba config set channel_priority strict
```

Eso sigue la recomendación de conda-forge y conda para evitar mezclas problemáticas de canales. ([Conda-Forge][4])

### 3.6. Crear el ambiente `qgis_env`

Se asume que el archivo `environment-qgis_env.yml` fue guardado dentro del sistema de archivos Linux, por ejemplo en `~/proyectos/qgis-wsl/`.

```bash
cd ~/proyectos/qgis-wsl
micromamba create -f environment-qgis_env.yml
```

### 3.7. Activar el ambiente

```bash
micromamba activate qgis_env
```

### 3.8. Verificación básica

```bash
python --version
python -c "import qgis, xarray, geopandas, dask, pandas, snakemake, pytest, numpy, scipy, statsmodels, sklearn, matplotlib; print('OK')"
```

### 3.9. Lanzar QGIS en WSL

Si tu Windows 11 tiene WSLg funcionando, puedes intentar:

```bash
qgis
```

Microsoft documenta que WSL 2 en Windows 11 soporta aplicaciones GUI Linux con integración de escritorio. ([Microsoft Learn][6])

## 4) Variante sin archivo YAML

Si prefieres crear el ambiente con un solo comando, tanto en Windows como en WSL puedes usar esto:

```bash
micromamba create -n qgis_env -c conda-forge/label/qgis_ltr -c conda-forge \
  python=3.11 qgis xarray geopandas dask pandas snakemake pytest numpy scipy \
  statsmodels scikit-learn matplotlib
```

Luego:

```bash
micromamba activate qgis_env
```

La creación y activación de ambientes con comandos de este tipo forma parte del flujo normal documentado para Mamba/Micromamba. ([Mamba][7])

## 5) Recomendaciones concretas para nuestro caso

Para nuestro flujo de trabajo sugerimos lo siguiente:

* **Windows 11**: usar `qgis_env` como entorno principal para pruebas con QGIS de escritorio y scripts Python/QGIS.
* **WSL**: usar otro `qgis_env` paralelo para trabajo *headless*, Snakemake, geopandas, xarray y procesamiento reproducible.
* Mantener ambos con el mismo `environment-qgis_env.yml` para reducir divergencias.
* Evitar mezclar `defaults` con `conda-forge`, porque conda-forge recomienda priorizar su propio canal y usar ambientes separados del entorno base. ([Conda-Forge][8])

## 6) Dos notas útiles

Primera: si quieres la rama más estable de QGIS, el uso de **`conda-forge/label/qgis_ltr`** tiene sentido aquí porque hay *builds* publicados de la línea LTR para `py311` tanto en `win-64` como en `linux-64`. ([Anaconda][2])

Segunda: `qgis` en conda-forge está soportado en **Windows 64-bit** y **Linux 64-bit**, así que nuestro objetivo es plenamente razonable tanto en Windows nativo como en WSL. ([Anaconda][9])

## 7) Conveniencia de usar WSL

La conveniencia de usar **WSL** sobre **Windows** aparece sobre todo cuando se quiere desarrollar un workflow **headless-first**, reproducible y cercano a un entorno Linux real. No lo proponemos como sustituto total de Windows para el uso interactivo de QGIS Desktop, sino como el entorno principal para procesamiento, automatización y ejecución de scripts.

En este esquema, Windows se aprovecha principalmente para prototipado, inspección visual y afinación cartográfica interactiva.

En una frase:

> **WSL** para producción; **Windows** para experimentar, inspeccionar y afinar.

| Aspecto                                     | Windows nativo                                                                                                                        | WSL                                                                                                                                                                                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **QGIS Desktop (GUI)**                      | **Mejor opción** para uso interactivo intensivo: abrir proyectos, revisar simbología, editar capas y afinar composición cartográfica. | Funciona con **WSLg** en Windows 11, pero Microsoft aclara que no ofrece una experiencia de escritorio Linux completa; sirve bien, pero no suele ser la primera elección para uso GUI pesado. ([Microsoft Learn][10])             |
| **PyQGIS standalone**                       | Muy viable, especialmente si quieres integrar scripts con el QGIS de escritorio de Windows y trabajar cerca de tus proyectos `.qgz`.  | **Muy atractivo** si quieres scripts reproducibles y un entorno Linux limpio. Encaja mejor con automatización y *shell scripting*. WSL permite correr herramientas Linux sin VM tradicional. ([Microsoft Learn][11])              |
| **`qgis_process` / procesamiento headless** | Funciona, pero el entorno Windows suele introducir más fricción en rutas, *shell* y automatización.                                   | **Ventaja clara de WSL**. QGIS documenta `qgis_process` explícitamente e incluso indica cómo usarlo en sistemas sin *window manager* con `QT_QPA_PLATFORM=offscreen`, lo que encaja muy bien con trabajo *headless*. ([QGIS][12]) |
| **Snakemake / bash / utilidades Linux**     | Se puede, pero con más fricción si tu flujo usa muchas herramientas POSIX.                                                            | **Mejor opción**. WSL está pensado para correr la mayor parte de herramientas, utilidades y aplicaciones GNU/Linux directamente en Windows. ([Microsoft Learn][11])                                                               |
| **Rendimiento de archivos**                 | Mejor cuando trabajas en el filesystem de Windows.                                                                                    | Mejor cuando trabajas en el filesystem Linux, por ejemplo en `~/proyecto`; Microsoft recomienda evitar trabajar “cruzando” sistemas si no hace falta. ([Microsoft Learn][13])                                                     |
| **Acceso cruzado a archivos y comandos**    | Muy cómodo para integrarte con apps Windows.                                                                                          | También muy fuerte: WSL permite mezclar comandos Windows y Linux, abrir Explorer desde Linux y ejecutar Linux desde PowerShell con `wsl <command>`. ([Microsoft Learn][13])                                                       |
| **Integración con flujos reproducibles**    | Buena, pero más propensa a detalles específicos de Windows.                                                                           | **Mejor opción** si quieres un pipeline más portable a servidores Linux o a otros entornos científicos. Esto se desprende del propio modelo de WSL como entorno GNU/Linux dentro de Windows. ([Microsoft Learn][11])              |
| **Micromamba**                              | Muy usable.                                                                                                                           | Muy usable también; micromamba es un ejecutable pequeño, sin `base` obligatorio, y resulta especialmente cómodo en CI, Docker y *shells* Linux. ([Mamba][14])                                                                     |
| **Curva de complejidad operativa**          | **Más simple** si tu prioridad es “instalar y usar QGIS Desktop”.                                                                     | Más potente, pero requiere más disciplina: filesystem Linux, *shell*, rutas y separación clara entre GUI y procesamiento. ([Microsoft Learn][13])                                                                                 |
| **Para tu flujo concreto**                  | Mejor para inspección visual, simbología, composición y revisión final.                                                               | **Mejor para producir**: scripts, validación, Snakemake, PyQGIS *headless*, `qgis_process`, `xarray`, `geopandas`, `dask`. ([QGIS][12])                                                                                           |

[1]: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html "Micromamba Installation — documentation"
[2]: https://anaconda.org/conda-forge/qgis/files?version= "qgis - conda-forge | Anaconda.org"
[3]: https://mamba.readthedocs.io/en/latest/user_guide/concepts.html "Concepts - Mamba's documentation! - Read the Docs"
[4]: https://conda-forge.org/docs/user/tipsandtricks/ "Tips & tricks | conda-forge"
[5]: https://learn.microsoft.com/en-us/windows/wsl/install "How to install Linux on Windows with WSL"
[6]: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps "Run Linux GUI apps with WSL"
[7]: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html "Mamba User Guide — documentation"
[8]: https://conda-forge.org/docs/user/transitioning_from_defaults/ "Transitioning from Anaconda's defaults channels"
[9]: https://anaconda.org/conda-forge/qgis "qgis - conda-forge | Anaconda.org"
[10]: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps "Run Linux GUI apps with WSL | Microsoft Learn"
[11]: https://learn.microsoft.com/en-us/windows/wsl/ "Windows Subsystem for Linux Documentation | Microsoft Learn"
[12]: https://docs.qgis.org/latest/en/docs/user_manual/processing/standalone.html "Using processing from the command line — QGIS Documentation"
[13]: https://learn.microsoft.com/en-us/windows/wsl/filesystems "Working across file systems | Microsoft Learn"
[14]: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html "Micromamba User Guide — documentation"



## PyCharm windows

### Configuración de PyCharm para usar QGIS en Windows

Para ejecutar scripts con `qgis.core` desde PyCharm usando el entorno `C:\QGis_env`, fue necesario complementar la configuración del intérprete con variables de entorno y una ruta adicional de módulos.

#### 1. Variables de entorno en `Run/Debug Configurations`

Definir las siguientes variables:

```text
GDAL_DATA=C:\QGis_env\Library\share\gdal
GDAL_DRIVER_PATH=C:\QGis_env\Library\lib\gdalplugins
PROJ_LIB=C:\QGis_env\Library\share\proj
````

Estas variables permiten que GDAL y PROJ localicen correctamente sus datos auxiliares y controladores.

#### 2. `Interpreter Paths` del intérprete en PyCharm

Agregar esta ruta:

```text
C:\QGis_env\Library\python
```

Esto permite que el intérprete encuentre el paquete `qgis` y resuelva correctamente importaciones como:

```python
from qgis.core import QgsApplication
```

