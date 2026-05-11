#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsRasterLayer,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsReferencedRectangle
)

from qgis.PyQt.QtGui import QColor


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
BN_MAPS_DIR = DROPBOX_DIR / "BN-results" / "BN_maps"

# Carpeta donde están los GeoTIFF generados
IE_RASTER_DIR = BN_MAPS_DIR / "cei_final_ie_expected_port_5_equal_2026" / "Python"

# Proyecto QGIS de salida
QGIS_PROJECT_PATH = BN_MAPS_DIR / "cei_final_ie_expected_port_5_equal_2026" / "ie_rasters_python.qgz"


def sort_rasters_numerically(tif_files: list[Path]) -> list[Path]:
    return sorted(tif_files, key=lambda p: int(p.stem.split("_")[-1]))


def find_geotiffs(input_dir: Path) -> list[Path]:
    tif_files = list(input_dir.rglob("*.tif"))
    tif_files = sort_rasters_numerically(tif_files)

    if not tif_files:
        raise FileNotFoundError(f"No se encontraron GeoTIFF en: {input_dir}")

    return tif_files


def init_qgis() -> QgsApplication:
    qgs = QgsApplication([], False)
    qgs.initQgis()
    return qgs


def apply_ie_style(layer: QgsRasterLayer) -> None:
    """
    Aplica una paleta fija y común para integridad ecosistémica:
    0 = degradado
    1 = prístino
    """
    shader = QgsRasterShader()
    color_ramp = QgsColorRampShader()
    color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
    color_ramp.setClip(True)

    items = [
        QgsColorRampShader.ColorRampItem(0.00, QColor("#8c510a"), "0.00"),
        QgsColorRampShader.ColorRampItem(0.25, QColor("#d8b365"), "0.25"),
        QgsColorRampShader.ColorRampItem(0.50, QColor("#f6e8c3"), "0.50"),
        QgsColorRampShader.ColorRampItem(0.75, QColor("#5ab4ac"), "0.75"),
        QgsColorRampShader.ColorRampItem(1.00, QColor("#01665e"), "1.00"),
    ]

    color_ramp.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp)

    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(),
        1,
        shader,
    )

    renderer.setClassificationMin(0.0)
    renderer.setClassificationMax(1.0)

    layer.setRenderer(renderer)
    layer.triggerRepaint()


def combined_extent(layers: list[QgsRasterLayer]):
    extent = None
    for layer in layers:
        if not layer.isValid():
            continue
        if extent is None:
            extent = layer.extent()
        else:
            extent.combineExtentWith(layer.extent())
    return extent


def add_rasters_to_group(
        project: QgsProject,
        tif_files: list[Path],
        group_name: str,
) -> list[QgsRasterLayer]:
    root = project.layerTreeRoot()
    group = root.addGroup(group_name)
    group.setExpanded(False)

    loaded_layers: list[QgsRasterLayer] = []

    for tif_path in tif_files:
        layer_name = tif_path.stem
        layer = QgsRasterLayer(str(tif_path), layer_name)

        if not layer.isValid():
            print(f"Aviso: no se pudo cargar {tif_path}")
            continue

        apply_ie_style(layer)

        project.addMapLayer(layer, False)
        layer_node = group.addLayer(layer)
        layer_node.setExpanded(False)

        loaded_layers.append(layer)

        print(f"Cargado: {tif_path.name}")

    return loaded_layers


def lambert_conformal_crs() -> QgsCoordinateReferenceSystem:
    crs = QgsCoordinateReferenceSystem.fromProj(
        'PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
        'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
        'AUTHORITY["EPSG","4326"]],PROJECTION["Lambert_Conformal_Conic_2SP"],'
        'PARAMETER["latitude_of_origin",12],PARAMETER["central_meridian",-102],'
        'PARAMETER["standard_parallel_1",17.5],PARAMETER["standard_parallel_2",29.5],'
        'PARAMETER["false_easting",2500000],PARAMETER["false_northing",0],'
        'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
        'AXIS["Northing",NORTH]]'
    )
    return crs


def create_project(
        tif_dir: Path,
        output_project_path: Path,
        project_title: str = "IE rasters",
        group_name: str = "IE rasters",
) -> Path:
    tif_files = find_geotiffs(tif_dir)

    project = QgsProject.instance()
    project.clear()
    project.setTitle(project_title)

    lambert_crs = lambert_conformal_crs()
    if lambert_crs.isValid():
        project.setCrs(lambert_crs)

    layers = add_rasters_to_group(project, tif_files, group_name=group_name)

    extent = combined_extent(layers)
    if extent is not None:
        project.viewSettings().setDefaultViewExtent(
            QgsReferencedRectangle(extent, project.crs())
        )

    output_project_path.parent.mkdir(parents=True, exist_ok=True)

    ok = project.write(str(output_project_path))
    if not ok:
        raise RuntimeError(f"No se pudo escribir el proyecto QGIS en: {output_project_path}")

    return output_project_path


def main() -> None:
    qgs = init_qgis()
    try:
        project_path = create_project(
            tif_dir=IE_RASTER_DIR,
            output_project_path=QGIS_PROJECT_PATH,
            project_title="Índice de integridad ecosistémica",
            group_name="IE Python",
        )
        print(f"Proyecto QGIS creado: {project_path}")
    finally:
        qgs.exitQgis()


if __name__ == "__main__":
    main()