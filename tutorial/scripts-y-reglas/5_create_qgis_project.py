"""
=============================================================================
5_create_qgis_project.py
-----------------------------------------------------------------------------
Propósito:
    Actualizar las rutas de datos en una plantilla QGIS (.qgz) de forma
    dinámica mediante re-vinculación de capas, garantizando la persistencia
    de simbología y layouts.

Rol en el workflow:
    Auxiliar. Preparación del proyecto editorial para exportación final.
=============================================================================
"""

import argparse
import os
import sys
from qgis.core import *

# =============================================================================
# Configuración del entorno
# =============================================================================
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QGIS_NO_PLUGINS"] = "1"
os.environ["QGIS_PREFIX_PATH"] = "/srv/iie/envs/qgis_env"
container_prefix = "/data"
host_prefix = "/srv/iie-tutor"


# =============================================================================
# Funciones auxiliares
# =============================================================================
def to_container_path(host_path):
    abs_path = os.path.abspath(host_path)
    return abs_path.replace(host_prefix, container_prefix)

# =============================================================================
# Integración main
# =============================================================================
def main():
    # 1. Parseo de argumentos
    parser = argparse.ArgumentParser(description="Actualiza rutas de capas en un proyecto QGIS")
    parser.add_argument("--raster", required=True, help="Ruta al archivo raster de datos")
    parser.add_argument("--vector", required=True, help="Ruta al archivo vectorial de datos")
    parser.add_argument("--output", required=True, help="Ruta del proyecto de salida (.qgz)")
    parser.add_argument("--plantilla", required=True, help="Ruta de la plantilla base")
    args = parser.parse_args()

    # 2. Inicialización de QGIS Application
    qgs = QgsApplication([], False)
    qgs.initQgis()

    try:
        # 3. Cargar proyecto
        project = QgsProject.instance()
        if not project.read(args.plantilla):
            raise Exception(f"No se pudo leer la plantilla: {args.plantilla}")


        # Recolectar IDs de lo que queremos publicar o no
        wfs_layers = []
        wcs_layers = []
        restricted_wms_layers = [] # Nueva lista para capas prohibidas en WMS

        for layer in project.mapLayers().values():
            # Capas que SÍ queremos compartir como datos puros
            if layer.name() == "capa_vectores":
                wfs_layers.append(layer.id())
            elif layer.name() == "capa_raster":
                wcs_layers.append(layer.id())

            # Capas que NO queremos compartir por WMS (Mapas Base)
            # Asegúrate de que los nombres coincidan con los de la plantilla
            elif layer.name() in ["Google Satellite", "OpenStreetMap", "Mapa Base", "Google satélite"]: 
                restricted_wms_layers.append(layer.name())

        target_crs = project.crs()

        # 4. Vinculación inteligente y asignación de CRS
        for layer in project.mapLayers().values():
            raster_path = to_container_path(args.raster)
            vector_path = to_container_path(args.vector)

            if layer.name() == "capa_raster":
                layer.setDataSource(raster_path, "Mapa ejemplo raster", "gdal")
                layer.setCrs(target_crs)
                
            elif layer.name() == "capa_vectores":
                # Mantenemos el nombre "capa_vectores"
                layer.setDataSource(vector_path, "Mapa ejemplo vectorial", "ogr")
                layer.setCrs(target_crs)

        project.setCrs(target_crs)

        # 5. Garantía de escritura de directorios
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # =============================================================================
        # 6. Anotar Metadatos e Integridad para el Servidor QGIS
        # =============================================================================

        project.writeEntry("WMSServiceTitle", "/", "Análisis de Integridad Ecosistémica - IIE")
        project.writeEntry("WMSUrl", "/", "http://octavio-wsl:8085/cgi-bin/qgis_mapserv.fcgi")
        
        project.writeEntry("WMSUseCrsRestrictions", "/", True)
        project.writeEntry("WMSCrsList", "/", ["EPSG:3857", "EPSG:4326", "EPSG:6369", "EPSG:6362"])
        
        project.writeEntry("WMSCapabilitiesLayerMenu", "/", True)    

        # Forzar el orden de ejes estándar para evitar el problema de "Latitude/Longitude"
        # Esto le indica al servidor que siempre entregue X,Y (Longitud, Latitud)
        project.writeEntry("WMSAxisOrientation", "/", "xy")
        project.writeEntry("WFSAxisOrientation", "/", "xy")

        # Publicar WFS y WCS
        if wfs_layers:
            project.writeEntry("WFSLayers", "/", wfs_layers)
            
        if wcs_layers:
            project.writeEntry("WCSLayers", "/", wcs_layers)

        # Excluir explícitamente los mapas base del WMS
        if restricted_wms_layers:
            project.writeEntry("WMSRestrictedLayers", "/", restricted_wms_layers)

        # =============================================================================
        # 8. Forzar Extensión y Zoom (Calculando Extent)
        # =============================================================================
        # Calculamos la unión de todas las extensiones de las capas
        full_extent = QgsRectangle()
        for layer in project.mapLayers().values():
            if layer.isValid():
                # Actualizamos la extensión total con la de cada capa
                full_extent.combineExtentWith(layer.extent())
        
        # Si logramos calcular una extensión válida, la aplicamos al proyecto
        if not full_extent.isEmpty():
            project.writeEntry("ProjectView", "/fullExtent", 
                               f"{full_extent.xMinimum()},{full_extent.xMaximum()},{full_extent.yMinimum()},{full_extent.yMaximum()}")
            # Configuramos el canvas para que el servidor inicie en esa zona
            project.writeEntry("ProjectView", "/CanvasExtent", 
                               f"{full_extent.xMinimum()},{full_extent.xMaximum()},{full_extent.yMinimum()},{full_extent.yMaximum()}")
            print("Éxito: Extensión espacial calculada y aplicada al proyecto.")   
            
        # =============================================================================
        # 9. Configuración de Simbología y Atributos para Servidor
        # =============================================================================
        for layer in project.mapLayers().values():
            if layer.name() == "Vector_Resultante":
                # Fuerza a QGIS Server a usar una columna específica para el WFS (Display Field)
                # Reemplaza 'NOMBRE_DE_TU_ATRIBUTO' por la columna que quieres mostrar
                layer.setDisplayField("integridad_simulada") # o el nombre de tu columna clave
                
                # Opcional: Si quieres asegurar que siempre cargue un estilo específico
                # layer.loadNamedStyle("/srv/iie-tutor/plantillas/estilo_vector.qml")

            elif layer.name() == "Raster_Resultante":
                # Configuración específica para ráster si es necesario
                pass

        # 7. Guardar el archivo
        project.write(args.output)
        print(f"Éxito: Proyecto generado en {args.output}")

    except Exception as e:
        print(f"Error crítico: {e}")
        sys.exit(1)
    finally:
        qgs.exitQgis()

if __name__ == "__main__":
    main()