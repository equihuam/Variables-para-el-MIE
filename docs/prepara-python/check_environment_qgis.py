#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_environment_qgis.py

Verificación mínima de QGIS headless para workflow-iie.

Qué valida:
- import de qgis.core
- arranque y cierre de QgsApplication en modo headless
- disponibilidad básica de clases clave
- creación y carga mínima de raster
- creación y carga mínima de vector temporal
- reporte de versión de QGIS/PyQt si está disponible

Uso:
    python check_environment_qgis.py

Notas:
- Este script debe ejecutarse con un intérprete que tenga acceso a PyQGIS.
- Si se usa el Python del entorno de QGIS o un entorno configurado para PyQGIS,
  esta prueba ayuda a confirmar que el modo headless-first está operativo.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_imports():
    print("\n== Verificación de imports PyQGIS ==")
    try:
        from qgis.core import (
            Qgis,
            QgsApplication,
            QgsProject,
            QgsRasterLayer,
            QgsVectorLayer,
            QgsFeature,
            QgsGeometry,
            QgsPointXY,
            QgsField,
            QgsVectorFileWriter,
            QgsCoordinateReferenceSystem,
        )
        ok("Import de qgis.core correcto")
        return {
            "Qgis": Qgis,
            "QgsApplication": QgsApplication,
            "QgsProject": QgsProject,
            "QgsRasterLayer": QgsRasterLayer,
            "QgsVectorLayer": QgsVectorLayer,
            "QgsFeature": QgsFeature,
            "QgsGeometry": QgsGeometry,
            "QgsPointXY": QgsPointXY,
            "QgsField": QgsField,
            "QgsVectorFileWriter": QgsVectorFileWriter,
            "QgsCoordinateReferenceSystem": QgsCoordinateReferenceSystem,
        }
    except Exception as e:
        fail(f"No se pudo importar qgis.core: {e}")
        return None


def init_qgis(QgsApplication):
    print("\n== Arranque de QgsApplication ==")
    try:
        app = QgsApplication([], False)
        app.initQgis()
        ok("QgsApplication arrancó en modo headless")
        return app
    except Exception as e:
        fail(f"No se pudo iniciar QgsApplication: {e}")
        return None


def check_versions(Qgis):
    print("\n== Versión de QGIS ==")
    try:
        version_int = Qgis.QGIS_VERSION_INT
        version = Qgis.QGIS_VERSION
        ok(f"QGIS disponible: {version} ({version_int})")
        return True
    except Exception as e:
        warn(f"No se pudo obtener la versión de QGIS: {e}")
        return False


def check_raster_loading(QgsRasterLayer):
    print("\n== Carga mínima de raster en PyQGIS ==")
    try:
        import gc
        import os
        import tempfile

        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        tmpdir = Path(tempfile.mkdtemp())
        tif_path = tmpdir / "test_qgis_raster.tif"

        arr = np.array([[1, 2], [3, 4]], dtype="float32")
        transform = from_origin(0, 2, 1, 1)

        meta = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": -9999.0,
        }

        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(arr, 1)

        layer = QgsRasterLayer(str(tif_path), "test_raster")
        if not layer.isValid():
            fail("QgsRasterLayer no pudo cargar el raster temporal")
            return False

        ok("QgsRasterLayer cargó correctamente un raster temporal")

        # Liberar referencias antes de limpiar archivos en Windows
        del layer
        gc.collect()

        try:
            os.remove(tif_path)
            os.rmdir(tmpdir)
        except Exception as cleanup_err:
            warn(f"No se pudo limpiar el raster temporal, pero la prueba fue correcta: {cleanup_err}")

        return True

    except Exception as e:
        fail(f"Prueba de raster en PyQGIS falló: {e}")
        return False


def check_memory_vector_layer(
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
):
    print("\n== Capa vectorial en memoria ==")
    try:
        layer = QgsVectorLayer("Point?crs=EPSG:4326", "test_points", "memory")
        if not layer.isValid():
            fail("No se pudo crear una capa vectorial en memoria")
            return False

        provider = layer.dataProvider()
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(0, 0)))
        provider.addFeature(feat)
        layer.updateExtents()

        if layer.featureCount() != 1:
            fail("La capa vectorial en memoria no contiene el feature esperado")
            return False

        ok("Capa vectorial en memoria creada y poblada correctamente")
        return True
    except Exception as e:
        fail(f"Prueba de capa vectorial en memoria falló: {e}")
        return False


def check_project_write(QgsProject, QgsCoordinateReferenceSystem):
    print("\n== Escritura mínima de proyecto QGIS ==")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            qgz_path = Path(tmpdir) / "test_project.qgz"
            project = QgsProject.instance()
            project.clear()
            project.setTitle("test_project")
            crs = QgsCoordinateReferenceSystem("EPSG:4326")
            if crs.isValid():
                project.setCrs(crs)

            ok_write = project.write(str(qgz_path))
            if not ok_write or not qgz_path.exists():
                fail("No se pudo escribir un proyecto QGIS temporal")
                return False

            ok("Proyecto QGIS temporal escrito correctamente")
            return True
    except Exception as e:
        fail(f"Prueba de escritura de proyecto QGIS falló: {e}")
        return False


def main() -> int:
    print("Verificación de entorno QGIS headless para workflow-iie")
    print("=" * 60)

    imported = check_imports()
    if imported is None:
        print("\nResultado final: PyQGIS NO DISPONIBLE")
        return 1

    app = init_qgis(imported["QgsApplication"])
    if app is None:
        print("\nResultado final: NO SE PUDO INICIAR QGIS HEADLESS")
        return 1

    try:
        checks = [
            check_versions(imported["Qgis"]),
            check_raster_loading(imported["QgsRasterLayer"]),
            check_memory_vector_layer(
                imported["QgsVectorLayer"],
                imported["QgsFeature"],
                imported["QgsGeometry"],
                imported["QgsPointXY"],
            ),
            check_project_write(
                imported["QgsProject"],
                imported["QgsCoordinateReferenceSystem"],
            ),
        ]
    finally:
        try:
            app.exitQgis()
            ok("QgsApplication cerró correctamente")
        except Exception as e:
            warn(f"No se pudo cerrar QgsApplication limpiamente: {e}")

    if all(checks):
        print("\nResultado final: QGIS HEADLESS OK")
        return 0

    print("\nResultado final: HAY PROBLEMAS EN EL ENTORNO QGIS HEADLESS")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
