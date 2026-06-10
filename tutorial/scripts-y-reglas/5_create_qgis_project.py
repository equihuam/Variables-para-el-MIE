#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
5_create_qgis_project.py
-----------------------------------------------------------------------------
Propósito:
    Construir proyectos QGIS reproducibles desde cero para QGIS Desktop
    y QGIS Server, usando únicamente una plantilla .qpt para los layouts.

Productos esperados:
    - target standalone:
        Proyecto QGIS con rutas relativas, pensado para abrirse en QGIS Desktop.

    - target server:
        Proyecto QGIS con rutas absolutas internas al contenedor Docker,
        pensado para QGIS Server.

Estrategia:
    - Cargar raster y vector desde salidas reales del workflow usando rutas del host.
    - Aplicar simbología raster tipo viridis.
    - Aplicar simbología vectorial graduada.
    - Para standalone:
        generar dos layouts editoriales independientes desde el mismo QPT.
    - Para server:
        evitar llamadas frágiles de PyQGIS Server.
        El proyecto se escribe primero con rutas válidas del host y luego se
        parchea el .qgs interno para:
            * reemplazar /srv/iie-tutor por /data;
            * publicar WMS/WFS/WCS;
            * registrar capas WFS/WCS.
=============================================================================
"""

import argparse
import math
import os
import re
import sys
import tempfile
import zipfile
from xml.sax.saxutils import escape as xml_escape

from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtXml import QDomDocument

from qgis.core import (
    QgsApplication,
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsGradientColorRamp,
    QgsGraduatedSymbolRenderer,
    QgsRendererRange,
    QgsSymbol,
    QgsPrintLayout,
    QgsReadWriteContext,
    QgsLayoutItemMap,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemPage,
    QgsLayoutSize,
    QgsLayoutPoint,
    QgsUnitTypes,
    QgsLayerTreeLayer,
    QgsRectangle,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsRasterBandStats,
)


# =============================================================================
# Entorno QGIS headless
# =============================================================================

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QGIS_NO_PLUGINS", "1")
os.environ.setdefault("QGIS_PREFIX_PATH", "/srv/iie/envs/qgis_env")


# =============================================================================
# Convenciones del proyecto
# =============================================================================

HOST_PREFIX = "/srv/iie-tutor"
CONTAINER_PREFIX = "/data"

PROJECT_CRS = "EPSG:4326"

OGC_WMS_CRS_LIST = ["EPSG:4326", "EPSG:3857", "EPSG:6372"]
OGC_WFS_CRS_LIST = ["EPSG:4326"]
#OGC_WFS_CRS_LIST = ["EPSG:6372"]
OGC_WCS_CRS_LIST = ["EPSG:4326"]

VECTOR_LAYER_INTERNAL = "reticula_variable"
VECTOR_LAYER_TITLE = "Retícula de integridad ecosistémica"
RASTER_LAYER_TITLE = "Integridad ecosistémica raster"

PREFERRED_VALUE_FIELDS = [
    "integridad_simulada",
    "valor_indice",
    "valor_iie",
]


# =============================================================================
# Layout imprimible
# =============================================================================

US_LETTER_W_MM = 279.4
US_LETTER_H_MM = 215.9

MAP_W_MM = 180.0
MAP_H_MM = 209.4
MAP_SCALE = 15_000_000

DEFAULT_MAP_X_MM = 10.0
DEFAULT_MAP_Y_MM = 3.25

DEFAULT_TITLE_X_MM = 10.0
DEFAULT_TITLE_Y_MM = 8.0
DEFAULT_TITLE_W_MM = 90.0
DEFAULT_TITLE_H_MM = 18.0

LAYOUT_RASTER_NAME = "iie-cartografia-raster"
LAYOUT_VECTOR_NAME = "iie-cartografia-vectorial"

TITLE_RASTER = "Mapa raster de\nintegridad ecosistémica"
TITLE_VECTOR = "Mapa vectorial de\nintegridad ecosistémica"


# =============================================================================
# QGIS Server / OGC
# =============================================================================

OGC_SERVICE_TITLE = "IIE - Integridad ecosistémica"
OGC_SERVICE_ABSTRACT = (
    "Servicio OGC generado automáticamente por el workflow IIE. "
    "Incluye variantes raster y vectorial de integridad ecosistémica."
)

OGC_RASTER_NAME = "iie_mapa_raster"
OGC_VECTOR_NAME = "iie_mapa_vectorial"

OGC_RASTER_TITLE = "Mapa raster de integridad ecosistémica"
OGC_VECTOR_TITLE = "Mapa vectorial de integridad ecosistémica"

OGC_CRS_LIST = ["EPSG:4326", "EPSG:3857"]


# =============================================================================
# Utilidades de ruta
# =============================================================================

def abs_path(path: str) -> str:
    """Devuelve una ruta absoluta normalizada."""
    return os.path.abspath(path)


def to_container_path(path: str) -> str:
    """
    Convierte una ruta absoluta del host a la ruta equivalente dentro
    del contenedor Docker.
    """
    path = abs_path(path)

    if not path.startswith(HOST_PREFIX):
        raise ValueError(
            f"La ruta no está dentro de {HOST_PREFIX}: {path}\n"
            "No puedo convertirla de forma segura a ruta de contenedor."
        )

    return path.replace(HOST_PREFIX, CONTAINER_PREFIX, 1)


def gpkg_uri(path: str, layername: str) -> str:
    """Construye una URI OGR explícita para una capa dentro de un GeoPackage."""
    return f"{path}|layername={layername}"


# =============================================================================
# Carga de capas
# =============================================================================

def load_vector_layer(host_gpkg: str) -> QgsVectorLayer:
    """Carga la capa vectorial desde el GeoPackage usando ruta del host."""
    uri = gpkg_uri(host_gpkg, VECTOR_LAYER_INTERNAL)
    layer = QgsVectorLayer(uri, VECTOR_LAYER_TITLE, "ogr")

    if not layer.isValid():
        raise RuntimeError(
            "No se pudo cargar la capa vectorial.\n"
            f"URI: {uri}"
        )

    return layer


def load_raster_layer(host_tif: str) -> QgsRasterLayer:
    """Carga el raster usando ruta del host."""
    layer = QgsRasterLayer(host_tif, RASTER_LAYER_TITLE, "gdal")

    if not layer.isValid():
        raise RuntimeError(
            "No se pudo cargar el raster.\n"
            f"Ruta: {host_tif}"
        )

    return layer


def find_value_field(layer: QgsVectorLayer) -> str:
    """Encuentra el campo numérico preferido para simbolizar el vector."""
    fields = [field.name() for field in layer.fields()]

    for candidate in PREFERRED_VALUE_FIELDS:
        if candidate in fields:
            return candidate

    raise RuntimeError(
        "No encontré un campo válido para simbolizar la capa vectorial.\n"
        f"Campos disponibles: {fields}\n"
        f"Campos esperados: {PREFERRED_VALUE_FIELDS}"
    )


# =============================================================================
# Simbología raster
# =============================================================================

def get_raster_minmax(layer: QgsRasterLayer) -> tuple[float, float]:
    """
    Calcula mínimo y máximo de la banda 1 usando la capa válida del host.
    """
    provider = layer.dataProvider()

    stats = provider.bandStatistics(
        1,
        QgsRasterBandStats.Min | QgsRasterBandStats.Max,
        layer.extent(),
        0,
    )

    vmin = stats.minimumValue
    vmax = stats.maximumValue

    if (
        vmin is None
        or vmax is None
        or not math.isfinite(vmin)
        or not math.isfinite(vmax)
        or vmin == vmax
    ):
        raise RuntimeError(
            f"Estadísticas raster inválidas: min={vmin}, max={vmax}"
        )

    return float(vmin), float(vmax)


def build_raster_viridis_renderer(
    layer: QgsRasterLayer,
    vmin: float,
    vmax: float,
) -> QgsSingleBandPseudoColorRenderer:
    """
    Construye un renderer raster tipo viridis.

    Se usa rampa discreta para hacerlo más robusto en QGIS Server WMS.
    """
    provider = layer.dataProvider()

    breaks = [
        float(vmin),
        float(vmin + 0.25 * (vmax - vmin)),
        float(vmin + 0.50 * (vmax - vmin)),
        float(vmin + 0.75 * (vmax - vmin)),
        float(vmax),
    ]

    colors = [
        "#440154",
        "#3b528b",
        "#21918c",
        "#5ec962",
        "#fde725",
    ]

    color_ramp_items = [
        QgsColorRampShader.ColorRampItem(
            value,
            QColor(color),
            f"{value:.2f}",
        )
        for value, color in zip(breaks, colors)
        if math.isfinite(value)
    ]

    color_ramp_shader = QgsColorRampShader(float(vmin), float(vmax))
    color_ramp_shader.setColorRampType(QgsColorRampShader.Discrete)
    color_ramp_shader.setColorRampItemList(color_ramp_items)
    color_ramp_shader.setClip(True)

    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(color_ramp_shader)

    renderer = QgsSingleBandPseudoColorRenderer(
        provider,
        1,
        raster_shader,
    )

    try:
        renderer.setClassificationMin(float(vmin))
        renderer.setClassificationMax(float(vmax))
    except Exception:
        pass

    return renderer


# =============================================================================
# Simbología vectorial
# =============================================================================

def build_vector_outline_renderer(layer: QgsVectorLayer) -> QgsSingleSymbolRenderer:
    """Renderer de respaldo: retícula sin relleno y con borde visible."""
    symbol = QgsFillSymbol.createSimple({
        "color": "255,255,255,0",
        "outline_color": "255,255,255,190",
        "outline_width": "0.25",
    })

    return QgsSingleSymbolRenderer(symbol)


def build_vector_graduated_renderer(
    layer: QgsVectorLayer,
    field_name: str,
) -> QgsGraduatedSymbolRenderer | QgsSingleSymbolRenderer:
    """Construye un renderer graduado simple usando el campo indicado."""
    values = []

    for feature in layer.getFeatures():
        value = feature[field_name]

        if value is None:
            continue

        try:
            value = float(value)
        except Exception:
            continue

        if math.isfinite(value):
            values.append(value)

    if len(values) == 0:
        raise RuntimeError(
            f"No hay valores numéricos válidos en el campo {field_name}"
        )

    vmin = min(values)
    vmax = max(values)

    if vmin == vmax:
        return build_vector_outline_renderer(layer)

    n_classes = 5
    step = (vmax - vmin) / n_classes

    ramp = QgsGradientColorRamp(
        QColor(247, 252, 245),
        QColor(0, 104, 55),
    )

    ranges = []

    for i in range(n_classes):
        lower = vmin + i * step
        upper = vmin + (i + 1) * step if i < n_classes - 1 else vmax

        if not math.isfinite(lower) or not math.isfinite(upper):
            continue

        color = ramp.color(float(i) / max(n_classes - 1, 1))

        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(color)
        symbol.setOpacity(0.55)

        label = f"{lower:.2f} – {upper:.2f}"

        ranges.append(
            QgsRendererRange(
                lower,
                upper,
                symbol,
                label,
            )
        )

    if len(ranges) == 0:
        return build_vector_outline_renderer(layer)

    return QgsGraduatedSymbolRenderer(field_name, ranges)


# =============================================================================
# Proyecto y árbol de capas
# =============================================================================

def set_layer_order(project: QgsProject, raster_layer, vector_layer) -> None:
    """
    Define orden visual en el árbol de capas:
        vector arriba
        raster abajo
    """
    root = project.layerTreeRoot()
    root.removeAllChildren()

    root.insertChildNode(0, QgsLayerTreeLayer(vector_layer))
    root.insertChildNode(1, QgsLayerTreeLayer(raster_layer))


def configure_server_layer_tree(project: QgsProject, raster_layer, vector_layer) -> None:
    """
    Organiza el árbol de capas para QGIS Server sin usar serverProperties().
    """
    root = project.layerTreeRoot()
    root.removeAllChildren()

    vector_node = root.insertLayer(0, vector_layer)
    vector_node.setItemVisibilityChecked(True)

    raster_node = root.insertLayer(1, raster_layer)
    raster_node.setItemVisibilityChecked(True)


# =============================================================================
# Layout QPT: dos layouts independientes desde la misma plantilla
# =============================================================================

def item_id(item) -> str:
    """Obtiene un identificador textual de un item de layout."""
    for attr in ("id", "uuid"):
        try:
            value = getattr(item, attr)()
            if value:
                return str(value)
        except Exception:
            pass

    try:
        return str(item.displayName())
    except Exception:
        return ""


def preferred_layout_item(layout: QgsPrintLayout, cls, preferred_tokens: list[str]):
    """
    Busca un item por tokens en id/displayName. Si no encuentra coincidencia,
    devuelve el primer item del tipo solicitado.
    """
    candidates = [item for item in layout.items() if isinstance(item, cls)]

    for item in candidates:
        ident = item_id(item).lower()
        if any(token.lower() in ident for token in preferred_tokens):
            return item

    return candidates[0] if candidates else None


def item_position_mm(item, default_x: float, default_y: float) -> tuple[float, float]:
    """Devuelve posición del item en mm, o fallback."""
    if item is None:
        return default_x, default_y

    pos = item.positionWithUnits()
    return float(pos.x()), float(pos.y())


def item_size_mm(item, default_w: float, default_h: float) -> tuple[float, float]:
    """Devuelve tamaño del item en mm, o fallback."""
    if item is None:
        return default_w, default_h

    size = item.sizeWithUnits()
    return float(size.width()), float(size.height())


def ensure_single_letter_landscape_page(layout: QgsPrintLayout) -> None:
    """Asegura una sola página US-letter en orientación landscape."""
    pages = layout.pageCollection()

    if pages.pageCount() == 0:
        page = QgsLayoutItemPage(layout)
        pages.addPage(page)

    while pages.pageCount() > 1:
        try:
            pages.deletePage(pages.page(pages.pageCount() - 1))
        except Exception:
            break

    page = pages.page(0)
    page.setPageSize(
        QgsLayoutSize(
            US_LETTER_W_MM,
            US_LETTER_H_MM,
            QgsUnitTypes.LayoutMillimeters,
        )
    )


def configure_map_item(layout: QgsPrintLayout, layer) -> None:
    """
    Configura el mapa del layout.

    No usa layer.extent(); conserva el encuadre del QPT y sólo fuerza tamaño
    y escala.
    """
    map_item = preferred_layout_item(
        layout,
        QgsLayoutItemMap,
        preferred_tokens=["mapa", "map", "main"],
    )

    if map_item is None:
        map_item = QgsLayoutItemMap(layout)
        layout.addLayoutItem(map_item)
        map_x, map_y = DEFAULT_MAP_X_MM, DEFAULT_MAP_Y_MM
        template_extent = QgsRectangle(layer.extent())
    else:
        map_x, map_y = item_position_mm(
            map_item,
            DEFAULT_MAP_X_MM,
            DEFAULT_MAP_Y_MM,
        )
        template_extent = QgsRectangle(map_item.extent())

    map_item.attemptMove(
        QgsLayoutPoint(
            map_x,
            map_y,
            QgsUnitTypes.LayoutMillimeters,
        )
    )

    map_item.attemptResize(
        QgsLayoutSize(
            MAP_W_MM,
            MAP_H_MM,
            QgsUnitTypes.LayoutMillimeters,
        )
    )

    map_item.setLayers([layer])

    if not template_extent.isEmpty():
        map_item.setExtent(template_extent)

    try:
        map_item.setScale(MAP_SCALE)
    except Exception:
        map_item.zoomScale(MAP_SCALE)

    map_item.attemptResize(
        QgsLayoutSize(
            MAP_W_MM,
            MAP_H_MM,
            QgsUnitTypes.LayoutMillimeters,
        )
    )

    try:
        map_item.setFollowVisibilityPreset(False)
    except Exception:
        pass

    map_item.refresh()


def configure_title_item(layout: QgsPrintLayout, title_text: str) -> None:
    """Actualiza el título del layout."""
    label_item = preferred_layout_item(
        layout,
        QgsLayoutItemLabel,
        preferred_tokens=["titulo", "title", "map_title"],
    )

    if label_item is None:
        label_item = QgsLayoutItemLabel(layout)
        layout.addLayoutItem(label_item)
        title_x, title_y = DEFAULT_TITLE_X_MM, DEFAULT_TITLE_Y_MM
        title_w, title_h = DEFAULT_TITLE_W_MM, DEFAULT_TITLE_H_MM
    else:
        title_x, title_y = item_position_mm(
            label_item,
            DEFAULT_TITLE_X_MM,
            DEFAULT_TITLE_Y_MM,
        )
        title_w, title_h = item_size_mm(
            label_item,
            DEFAULT_TITLE_W_MM,
            DEFAULT_TITLE_H_MM,
        )

    label_item.attemptMove(
        QgsLayoutPoint(
            title_x,
            title_y,
            QgsUnitTypes.LayoutMillimeters,
        )
    )

    label_item.attemptResize(
        QgsLayoutSize(
            title_w,
            title_h,
            QgsUnitTypes.LayoutMillimeters,
        )
    )

    label_item.setText(title_text)
    label_item.refresh()


def configure_legend_item(layout: QgsPrintLayout, layer) -> None:
    """Configura la leyenda existente del QPT para mostrar sólo la capa."""
    legend_item = preferred_layout_item(
        layout,
        QgsLayoutItemLegend,
        preferred_tokens=["leyenda", "legend"],
    )

    if legend_item is None:
        return

    legend_item.setAutoUpdateModel(False)

    root = legend_item.model().rootGroup()
    root.clear()
    root.addLayer(layer)

    try:
        legend_item.setTitle("")
    except Exception:
        pass

    legend_item.updateLegend()
    legend_item.refresh()


def load_layout_from_qpt(
    project: QgsProject,
    qpt_path: str,
    layout_name: str,
) -> QgsPrintLayout:
    """Carga una instancia independiente del QPT."""
    if not os.path.exists(qpt_path):
        raise RuntimeError(f"No existe la plantilla de layout: {qpt_path}")

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()

    with open(qpt_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    doc = QDomDocument()
    ok, error_msg, error_line, error_col = doc.setContent(template_content)

    if not ok:
        raise RuntimeError(
            "No se pudo leer el QPT como XML.\n"
            f"Archivo: {qpt_path}\n"
            f"Error: {error_msg}\n"
            f"Línea: {error_line}, columna: {error_col}"
        )

    context = QgsReadWriteContext()
    layout.loadFromTemplate(doc, context)
    layout.setName(layout_name)

    return layout


def add_or_replace_layout(project: QgsProject, layout: QgsPrintLayout) -> None:
    """Agrega un layout al proyecto, reemplazando uno previo con el mismo nombre."""
    manager = project.layoutManager()

    existing = manager.layoutByName(layout.name())
    if existing is not None:
        manager.removeLayout(existing)

    manager.addLayout(layout)


def configure_editorial_layout(
    layout: QgsPrintLayout,
    layer,
    title_text: str,
) -> None:
    """Configura una instancia de layout editorial de una sola página."""
    ensure_single_letter_landscape_page(layout)
    configure_map_item(layout, layer)
    configure_title_item(layout, title_text)
    configure_legend_item(layout, layer)


def import_layout_templates(
    project: QgsProject,
    qpt_path: str,
    raster_layer,
    vector_layer,
) -> None:
    """
    Crea dos layouts independientes a partir del mismo QPT:
        - iie-cartografia-raster
        - iie-cartografia-vectorial
    """
    raster_layout = load_layout_from_qpt(
        project=project,
        qpt_path=qpt_path,
        layout_name=LAYOUT_RASTER_NAME,
    )

    configure_editorial_layout(
        layout=raster_layout,
        layer=raster_layer,
        title_text=TITLE_RASTER,
    )

    add_or_replace_layout(project, raster_layout)

    vector_layout = load_layout_from_qpt(
        project=project,
        qpt_path=qpt_path,
        layout_name=LAYOUT_VECTOR_NAME,
    )

    configure_editorial_layout(
        layout=vector_layout,
        layer=vector_layer,
        title_text=TITLE_VECTOR,
    )

    add_or_replace_layout(project, vector_layout)


# =============================================================================
# Rutas finales según destino
# =============================================================================

def apply_target_datasources(
    project: QgsProject,
    host_vector: str,
    host_raster: str,
    target: str,
) -> tuple[str, str]:
    """
    Define las rutas finales esperadas.

    target == standalone:
        Las capas permanecen con rutas del host. QGIS escribe rutas relativas.

    target == server:
        Las capas permanecen con rutas del host durante PyQGIS.
        Las rutas /data/... se escriben después parcheando el .qgz.
    """
    if target == "standalone":
        project.writeEntry("Paths", "Absolute", False)

        vector_uri = gpkg_uri(host_vector, VECTOR_LAYER_INTERNAL)
        raster_uri = host_raster

    elif target == "server":
        # Usamos rutas finales sólo para reporte y parche XML.
        # No llamar setDataSource() aquí.
        server_vector = to_container_path(host_vector)
        server_raster = to_container_path(host_raster)

        vector_uri = gpkg_uri(server_vector, VECTOR_LAYER_INTERNAL)
        raster_uri = server_raster

    else:
        raise ValueError(f"Destino no reconocido: {target}")

    return vector_uri, raster_uri


# =============================================================================
# Parche XML del .qgz server
# =============================================================================

def _xml_text(value: str) -> str:
    """Escapa texto para XML."""
    return xml_escape(str(value), {'"': "&quot;"})


def _qstring_property(tag: str, value: str) -> str:
    return (
        f'    <{tag} type="QString">'
        f'{_xml_text(value)}'
        f'</{tag}>\n'
    )


def _bool_property(tag: str, value: bool) -> str:
    text = "true" if value else "false"
    return f'    <{tag} type="bool">{text}</{tag}>\n'


def _qstringlist_property(tag: str, values: list[str]) -> str:
    lines = [f'    <{tag} type="QStringList">\n']

    for value in values:
        lines.append(f'      <value>{_xml_text(value)}</value>\n')

    lines.append(f'    </{tag}>\n')

    return "".join(lines)


def _replace_or_insert_property(content: str, tag: str, xml_block: str) -> str:
    """
    Reemplaza una propiedad de proyecto si existe; si no existe, la inserta
    antes de </properties>.
    """
    pattern = re.compile(
        rf"\s*<{re.escape(tag)}\b[^>]*>.*?</{re.escape(tag)}>\s*",
        flags=re.DOTALL,
    )

    if pattern.search(content):
        return pattern.sub("\n" + xml_block, content)

    if "</properties>" not in content:
        # Fallback defensivo. Normalmente todo .qgs tiene <properties>.
        content = content.replace(
            "<qgis ",
            "<qgis ",
            1,
        )
        content = content.replace(
            ">\n",
            ">\n  <properties>\n  </properties>\n",
            1,
        )

    return content.replace("</properties>", xml_block + "  </properties>", 1)


def _patch_ogc_project_properties(
    content: str,
    raster_layer_id: str,
    vector_layer_id: str,
) -> str:
    """
    Inyecta propiedades OGC de QGIS Server directamente en el XML del proyecto.

    Esto evita usar project.writeEntry() para la configuración server, que en
    algunos entornos headless puede ser inestable.
    """
    replacements = {
        "WMSServiceTitle": _qstring_property("WMSServiceTitle", OGC_SERVICE_TITLE),
        "WMSServiceAbstract": _qstring_property(
            "WMSServiceAbstract",
            OGC_SERVICE_ABSTRACT,
        ),
        "WFSServiceTitle": _qstring_property("WFSServiceTitle", OGC_SERVICE_TITLE),
        "WFSServiceAbstract": _qstring_property(
            "WFSServiceAbstract",
            OGC_SERVICE_ABSTRACT,
        ),
        "WCSServiceTitle": _qstring_property("WCSServiceTitle", OGC_SERVICE_TITLE),
        "WCSServiceAbstract": _qstring_property(
            "WCSServiceAbstract",
            OGC_SERVICE_ABSTRACT,
        ),


        "WMSUseLayerIDs": _bool_property("WMSUseLayerIDs", False),
        "WMSCrsList": _qstringlist_property("WMSCrsList", OGC_WMS_CRS_LIST),
        "WFSCrsList": _qstringlist_property("WFSCrsList", OGC_WFS_CRS_LIST),
        "WCSCrsList": _qstringlist_property("WCSCrsList", OGC_WCS_CRS_LIST),
        "WFSLayers": _qstringlist_property("WFSLayers", [vector_layer_id]),
        "WCSLayers": _qstringlist_property("WCSLayers", [raster_layer_id]),
        "WFSTLayers": _qstringlist_property("WFSTLayers", []),
    }

    for tag, xml_block in replacements.items():
        content = _replace_or_insert_property(content, tag, xml_block)

    return content


def patch_qgz_for_server(
    input_qgz: str,
    output_qgz: str,
    raster_layer_id: str,
    vector_layer_id: str,
    host_prefix: str = HOST_PREFIX,
    container_prefix: str = CONTAINER_PREFIX,
) -> None:
    """
    Abre un .qgz, reemplaza rutas del host por rutas del contenedor e inyecta
    propiedades OGC en el .qgs interno.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = os.path.abspath(tmpdir)

        with zipfile.ZipFile(input_qgz, "r") as zin:
            zin.extractall(tmpdir)

        qgs_files = [
            os.path.join(tmpdir, name)
            for name in os.listdir(tmpdir)
            if name.endswith(".qgs")
        ]

        if len(qgs_files) != 1:
            raise RuntimeError(
                f"Esperaba exactamente un .qgs dentro de {input_qgz}, "
                f"pero encontré {len(qgs_files)}"
            )

        qgs_path = qgs_files[0]

        with open(qgs_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Convertir rutas host -> contenedor.
        content = content.replace(host_prefix, container_prefix)

        # Inyectar publicación OGC.
        content = _patch_ogc_project_properties(
            content=content,
            raster_layer_id=raster_layer_id,
            vector_layer_id=vector_layer_id,
        )

        with open(qgs_path, "w", encoding="utf-8") as f:
            f.write(content)

        with zipfile.ZipFile(output_qgz, "w", zipfile.ZIP_DEFLATED) as zout:
            for root, _, files in os.walk(tmpdir):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    arcname = os.path.relpath(full_path, tmpdir)
                    zout.write(full_path, arcname)


def write_project(
    project: QgsProject,
    output_project: str,
    target: str,
    raster_layer,
    vector_layer,
) -> None:
    """
    Guarda el proyecto.

    Para standalone:
        escribe directamente.

    Para server:
        escribe un .qgz temporal con rutas válidas del host y luego parchea
        el .qgs interno para rutas Docker y propiedades OGC.
    """
    if target == "standalone":
        if not project.write(output_project):
            raise RuntimeError(f"No se pudo guardar el proyecto: {output_project}")
        return

    if target == "server":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_qgz = os.path.join(tmpdir, os.path.basename(output_project))

            # Esta entrada es relativamente segura y ayuda a que el .qgs guarde
            # rutas absolutas que luego podamos reemplazar.
            try:
                project.writeEntry("Paths", "Absolute", True)
            except Exception:
                pass

            if not project.write(tmp_qgz):
                raise RuntimeError(
                    f"No se pudo guardar el proyecto temporal: {tmp_qgz}"
                )

            patch_qgz_for_server(
                input_qgz=tmp_qgz,
                output_qgz=output_project,
                raster_layer_id=raster_layer.id(),
                vector_layer_id=vector_layer.id(),
                host_prefix=HOST_PREFIX,
                container_prefix=CONTAINER_PREFIX,
            )

        return

    raise ValueError(f"Destino no reconocido: {target}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--raster", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout-template", required=False)

    parser.add_argument(
        "--target",
        choices=["standalone", "server"],
        required=True,
        help="Tipo de proyecto a generar: standalone o server",
    )

    args = parser.parse_args()

    if args.target == "standalone" and not args.layout_template:
        raise RuntimeError(
            "--layout-template es requerido para target standalone"
        )

    qgs = QgsApplication([], False)
    qgs.initQgis()

    try:
        project = QgsProject.instance()
        project.clear()

        target_crs = QgsCoordinateReferenceSystem(PROJECT_CRS)
        project.setCrs(target_crs)

        host_raster = abs_path(args.raster)
        host_vector = abs_path(args.vector)
        layout_template = (
            abs_path(args.layout_template)
            if args.layout_template
            else None
        )

        output_project = args.output

        # 1. Cargar capas desde rutas válidas del host.
        raster_layer = load_raster_layer(host_raster)
        vector_layer = load_vector_layer(host_vector)

        raster_layer.setCrs(target_crs)
        vector_layer.setCrs(target_crs)

        # 2. Nombres públicos estables.
        # Evitamos layer.serverProperties() para no disparar SIGSEGV.
        if args.target == "server":
            raster_layer.setName(OGC_RASTER_NAME)
            vector_layer.setName(OGC_VECTOR_NAME)

        # 3. Agregar capas al proyecto.
        project.addMapLayer(raster_layer, False)
        project.addMapLayer(vector_layer, False)

        # 4. Orden visual.
        if args.target == "server":
            configure_server_layer_tree(project, raster_layer, vector_layer)
        else:
            set_layer_order(project, raster_layer, vector_layer)

        # 5. Preparar simbología raster.
        raster_vmin, raster_vmax = get_raster_minmax(raster_layer)
        raster_renderer = build_raster_viridis_renderer(
            raster_layer,
            raster_vmin,
            raster_vmax,
        )

        # 6. Preparar simbología vectorial.
        value_field = find_value_field(vector_layer)

        try:
            vector_renderer = build_vector_graduated_renderer(
                vector_layer,
                value_field,
            )
            print(
                f"Simbología vectorial graduada preparada usando campo: "
                f"{value_field}",
                flush=True,
            )
        except Exception as e:
            print(
                "Advertencia: no se pudo preparar simbología graduada. "
                "Se usará retícula de contorno.\n"
                f"Detalle: {e}",
                file=sys.stderr,
                flush=True,
            )
            vector_renderer = build_vector_outline_renderer(vector_layer)

        # 7. Aplicar renderers mientras las capas siguen usando rutas válidas.
        raster_layer.setRenderer(raster_renderer)
        vector_layer.setRenderer(vector_renderer)

        print(
            f"Simbología raster tipo viridis clasificada y aplicada: "
            f"min={raster_vmin:.4f}, max={raster_vmax:.4f}",
            flush=True,
        )

        # 8. Resolver rutas finales esperadas.
        vector_uri_final, raster_uri_final = apply_target_datasources(
            project=project,
            host_vector=host_vector,
            host_raster=host_raster,
            target=args.target,
        )

        # 9. Configuración específica por destino.
        if args.target == "standalone":
            raster_layer.triggerRepaint()
            vector_layer.triggerRepaint()

            if layout_template is None:
                raise RuntimeError(
                    "--layout-template es requerido para target standalone"
                )

            import_layout_templates(
                project=project,
                qpt_path=layout_template,
                raster_layer=raster_layer,
                vector_layer=vector_layer,
            )

        elif args.target == "server":
            # No hacer nada adicional con PyQGIS Server aquí.
            # WMS/WFS/WCS se inyectan por parche XML después de project.write().
            pass

        else:
            raise ValueError(f"Destino no reconocido: {args.target}")

        # 10. Guardar proyecto.
        write_project(
            project=project,
            output_project=output_project,
            target=args.target,
            raster_layer=raster_layer,
            vector_layer=vector_layer,
        )

        print(f"Éxito: proyecto generado en {output_project}", flush=True)
        print(f"Destino:       {args.target}", flush=True)
        print(f"Raster host:   {host_raster}", flush=True)
        print(f"Vector host:   {gpkg_uri(host_vector, VECTOR_LAYER_INTERNAL)}", flush=True)
        print(f"Raster final:  {raster_uri_final}", flush=True)
        print(f"Vector final:  {vector_uri_final}", flush=True)

        if args.target == "standalone":
            print(f"Layout QPT:    {layout_template}", flush=True)
            print(f"Layouts:       {LAYOUT_RASTER_NAME}, {LAYOUT_VECTOR_NAME}", flush=True)
        else:
            print("Layouts:       no incluidos en variante server", flush=True)
            print(f"WMS layers:    {OGC_RASTER_NAME}, {OGC_VECTOR_NAME}", flush=True)
            print(f"WFS layers:    {OGC_VECTOR_NAME}", flush=True)
            print(f"WCS layers:    {OGC_RASTER_NAME}", flush=True)

    except Exception as e:
        print(f"Error crítico: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    finally:
        # En algunos entornos headless, qgs.exitQgis() puede provocar SIGSEGV
        # durante la destrucción de objetos C++ aunque el trabajo haya terminado.
        try:
            QgsProject.instance().clear()
        except Exception:
            pass


if __name__ == "__main__":
    main()