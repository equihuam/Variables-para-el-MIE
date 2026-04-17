#!/usr/bin/env python3
"""Consolida un manifest v0.6 de rutas/scripts en familias operativas globales
para apoyar el diseño de un workflow claro y comprensible.

Salidas:
- workflow_familias_globales_v1_0.json
- workflow_grafo_v1_0.json
- workflow_grafo_v1_0.mmd

Uso esperado:
    python workflow_familias_v1_0.py

Por defecto busca `r_script_paths_manifest_v0_6.json` en el mismo directorio
que este script.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

VERSION = "1.0"
DEFAULT_INPUT = "r_script_paths_manifest_v0_6.json"
DEFAULT_OUT_GLOBAL = "workflow_familias_globales_v1_0.json"
DEFAULT_OUT_GRAPH = "workflow_grafo_v1_0.json"
DEFAULT_OUT_MERMAID = "workflow_grafo_v1_0.mmd"


def norm_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().replace('\\', '/')
    if v.startswith('./'):
        v = v[2:]
    # conservar raíz Windows C:/... si existe
    if len(v) > 1 and v.endswith('/'):
        v = v[:-1]
    return v or None


@dataclass
class FamiliaGlobal:
    familia_id: str
    tipo_familia: str  # directorio_archivos | coleccion_llamada | coleccion_generada
    base: str
    patron: Optional[str] = None
    plantilla: Optional[str] = None
    relativa: Optional[str] = None
    modo_acceso: Optional[str] = None
    archivos_relativos: Set[str] = field(default_factory=set)
    scripts_consumen: Set[str] = field(default_factory=set)
    scripts_generan: Set[str] = field(default_factory=set)
    ocurrencias_consumo: Set[str] = field(default_factory=set)
    ocurrencias_generacion: Set[str] = field(default_factory=set)
    fuentes_consumo: Set[str] = field(default_factory=set)
    fuentes_generacion: Set[str] = field(default_factory=set)
    variables: Set[str] = field(default_factory=set)
    origenes_lista: Set[str] = field(default_factory=set)

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "familia_id": self.familia_id,
            "tipo_familia": self.tipo_familia,
            "base": self.base,
        }
        if self.patron:
            data["patron"] = self.patron
        if self.plantilla:
            data["plantilla"] = self.plantilla
        if self.relativa:
            data["relativa"] = self.relativa
        if self.modo_acceso:
            data["modo_acceso"] = self.modo_acceso
        if self.archivos_relativos:
            data["archivos_relativos"] = sorted(self.archivos_relativos)
        if self.variables:
            data["variables"] = sorted(self.variables)
        if self.origenes_lista:
            data["origenes_lista"] = sorted(self.origenes_lista)
        data["scripts_consumen"] = sorted(self.scripts_consumen)
        data["scripts_generan"] = sorted(self.scripts_generan)
        if self.ocurrencias_consumo:
            data["ocurrencias_consumo"] = sorted(self.ocurrencias_consumo)
        if self.ocurrencias_generacion:
            data["ocurrencias_generacion"] = sorted(self.ocurrencias_generacion)
        if self.fuentes_consumo:
            data["fuentes_consumo"] = sorted(self.fuentes_consumo)
        if self.fuentes_generacion:
            data["fuentes_generacion"] = sorted(self.fuentes_generacion)
        data["grado_consumo"] = len(self.scripts_consumen)
        data["grado_generacion"] = len(self.scripts_generan)
        return data


def family_key_from_collection(record: Dict[str, Any], generated: bool = False) -> Tuple[str, str, str, str]:
    base = norm_path(record.get("base")) or ""
    patron = record.get("patron") or ""
    plantilla = norm_path(record.get("plantilla")) or ""
    t = "coleccion_generada" if generated else "coleccion_llamada"
    return (t, base, patron, plantilla)


def family_key_from_single(ensamble: Dict[str, Any]) -> Tuple[str, str]:
    base = norm_path(ensamble.get("base")) or "(sin_base)"
    return ("directorio_archivos", base)


def ensure_family(families: Dict[Tuple[str, ...], FamiliaGlobal], key: Tuple[str, ...], **kwargs: Any) -> FamiliaGlobal:
    if key not in families:
        family_id = f"F{len(families)+1:03d}"
        families[key] = FamiliaGlobal(familia_id=family_id, **kwargs)
    return families[key]


def add_single_item(families: Dict[Tuple[str, ...], FamiliaGlobal], script: str, ref: str, role: str, item: Dict[str, Any]) -> None:
    ensamble = item.get("ensamble", {})
    base = norm_path(ensamble.get("base"))
    ruta = norm_path(ensamble.get("ruta"))
    relativa = ensamble.get("relativa")
    if not base:
        # intentar derivar base desde ruta literal
        if ruta and '/' in ruta:
            base = ruta.rsplit('/', 1)[0]
            relativa = ruta.rsplit('/', 1)[1]
        else:
            return
    key = family_key_from_single({"base": base})
    fam = ensure_family(
        families,
        key,
        tipo_familia="directorio_archivos",
        base=base,
        modo_acceso="archivo_unico",
    )
    if relativa:
        fam.archivos_relativos.add(relativa)
    if item.get("fuente"):
        if role == "entra":
            fam.fuentes_consumo.add(item["fuente"])
        else:
            fam.fuentes_generacion.add(item["fuente"])
    if role == "entra":
        fam.scripts_consumen.add(script)
        fam.ocurrencias_consumo.add(ref)
    else:
        fam.scripts_generan.add(script)
        fam.ocurrencias_generacion.add(ref)


def add_collection_item(
    families: Dict[Tuple[str, ...], FamiliaGlobal],
    script: str,
    collection: Dict[str, Any],
    generated: bool = False,
) -> str:
    key = family_key_from_collection(collection, generated=generated)
    fam = ensure_family(
        families,
        key,
        tipo_familia=("coleccion_generada" if generated else "coleccion_llamada"),
        base=norm_path(collection.get("base")) or "",
        patron=collection.get("patron"),
        plantilla=norm_path(collection.get("plantilla")),
        relativa=collection.get("relativa"),
        modo_acceso=collection.get("modo_acceso"),
    )
    variable = collection.get("variable") or collection.get("variable_fuente")
    if variable:
        fam.variables.add(variable)
    origen = norm_path(collection.get("origen_lista"))
    if origen:
        fam.origenes_lista.add(origen)
    for fuente in collection.get("fuentes_llamada", []):
        if generated:
            fam.fuentes_generacion.add(fuente)
        else:
            fam.fuentes_consumo.add(fuente)
    for occ in collection.get("ocurrencias", []):
        if generated:
            fam.ocurrencias_generacion.add(f"{script}:{occ}")
        else:
            fam.ocurrencias_consumo.add(f"{script}:{occ}")
    if generated:
        fam.scripts_generan.add(script)
    else:
        fam.scripts_consumen.add(script)
    return fam.familia_id


def build_global_view(manifest: Dict[str, Any]) -> Dict[str, Any]:
    families: Dict[Tuple[str, ...], FamiliaGlobal] = {}
    script_summary: Dict[str, Dict[str, List[str]]] = {}

    for script in sorted(manifest):
        node = manifest[script] or {}
        consumes: Set[str] = set()
        generates: Set[str] = set()

        # 1) colecciones resumidas del propio script
        local_collection_ids: Dict[str, str] = {}
        for cid, coll in (node.get("colecciones_llamadas") or {}).items():
            fid = add_collection_item(families, script, coll, generated=False)
            local_collection_ids[f"entra:{cid}"] = fid
            consumes.add(fid)
        for cid, coll in (node.get("colecciones_generadas") or {}).items():
            fid = add_collection_item(families, script, coll, generated=True)
            local_collection_ids[f"sale:{cid}"] = fid
            generates.add(fid)

        # 2) items individuales de entrada/salida
        for role_key, role in (("entra", "entra"), ("sale", "sale")):
            items = node.get(role_key) or {}
            for iid, item in items.items():
                ens = item.get("ensamble", {})
                ref = f"{script}:{role_key}:{iid}"
                modo = ens.get("modo_acceso")
                coll_id = ens.get("coleccion_id")
                if coll_id:
                    # vincular con colección ya resumida
                    fam_id = local_collection_ids.get(f"{role_key}:{coll_id}")
                    if fam_id:
                        if role == "entra":
                            consumes.add(fam_id)
                        else:
                            generates.add(fam_id)
                        continue
                if modo == "archivo_unico":
                    before = set(f.familia_id for f in families.values())
                    add_single_item(families, script, ref, role, item)
                    after = set(f.familia_id for f in families.values())
                    # localizar familia afectada más reciente o por base
                    ensamble = item.get("ensamble", {})
                    base = norm_path(ensamble.get("base"))
                    if not base:
                        ruta = norm_path(ensamble.get("ruta"))
                        if ruta and '/' in ruta:
                            base = ruta.rsplit('/',1)[0]
                    if base:
                        fam = families.get(("directorio_archivos", base))
                        if fam:
                            if role == "entra":
                                consumes.add(fam.familia_id)
                            else:
                                generates.add(fam.familia_id)
                elif modo == "plantilla_dinamica" and not coll_id:
                    # caso defensivo: salida dinámica no resumida en colecciones_generadas
                    synthetic = {
                        "base": ens.get("base"),
                        "plantilla": ens.get("plantilla"),
                        "relativa": ens.get("relativa"),
                        "modo_acceso": modo,
                        "fuentes_llamada": [item.get("fuente")] if item.get("fuente") else [],
                        "ocurrencias": [f"{role_key}:{iid}"],
                        "variable": item.get("expresion", {}).get("raw"),
                    }
                    fid = add_collection_item(families, script, synthetic, generated=(role == "sale"))
                    if role == "entra":
                        consumes.add(fid)
                    else:
                        generates.add(fid)

        script_summary[script] = {
            "consume_familias": sorted(consumes),
            "genera_familias": sorted(generates),
        }

    # 3) Ordenar y preparar resumen global
    families_json = [f.to_json() for f in sorted(families.values(), key=lambda x: (x.tipo_familia, x.base, x.familia_id))]

    # índices útiles
    by_role = {
        "familias_troncales_consumo": [],
        "familias_generadoras": [],
    }
    for fam in families_json:
        if fam["grado_consumo"] >= 2:
            by_role["familias_troncales_consumo"].append(fam["familia_id"])
        if fam["grado_generacion"] >= 1:
            by_role["familias_generadoras"].append(fam["familia_id"])

    return {
        "version": VERSION,
        "descripcion": "Vista consolidada global de familias operativas para diseño de workflow.",
        "familias_globales": families_json,
        "scripts": script_summary,
        "indices": by_role,
    }


def build_graph(global_view: Dict[str, Any]) -> Dict[str, Any]:
    families = {f["familia_id"]: f for f in global_view["familias_globales"]}
    scripts = global_view["scripts"]

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for fid, fam in sorted(families.items()):
        label = fam.get("base", fid)
        if fam["tipo_familia"] in {"coleccion_llamada", "coleccion_generada"} and fam.get("patron"):
            label = f"{label}\n[{fam['patron']}]"
        nodes.append({
            "id": fid,
            "tipo_nodo": "familia",
            "subtipo": fam["tipo_familia"],
            "label": label,
        })

    for script in sorted(scripts):
        sid = f"S::{script}"
        nodes.append({
            "id": sid,
            "tipo_nodo": "script",
            "label": script,
        })
        for fid in scripts[script].get("consume_familias", []):
            edges.append({
                "source": fid,
                "target": sid,
                "tipo": "consume",
            })
        for fid in scripts[script].get("genera_familias", []):
            edges.append({
                "source": sid,
                "target": fid,
                "tipo": "genera",
            })

    return {
        "version": VERSION,
        "descripcion": "Grafo simplificado script↔familia operativa.",
        "nodes": nodes,
        "edges": edges,
    }


def to_mermaid(graph: Dict[str, Any]) -> str:
    node_defs: List[str] = ["flowchart LR"]
    for node in graph["nodes"]:
        nid = node["id"].replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').replace(' ', '_')
        label = node["label"].replace('"', "'")
        if node["tipo_nodo"] == "script":
            node_defs.append(f'    {nid}["{label}"]')
        else:
            node_defs.append(f'    {nid}("{label}")')
    for edge in graph["edges"]:
        src = edge["source"].replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').replace(' ', '_')
        tgt = edge["target"].replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').replace(' ', '_')
        if edge["tipo"] == "consume":
            node_defs.append(f"    {src} --> {tgt}")
        else:
            node_defs.append(f"    {src} --> {tgt}")
    return "\n".join(node_defs) + "\n"


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    in_path = base_dir / DEFAULT_INPUT
    out_global = base_dir / DEFAULT_OUT_GLOBAL
    out_graph = base_dir / DEFAULT_OUT_GRAPH
    out_mermaid = base_dir / DEFAULT_OUT_MERMAID

    if not in_path.exists():
        raise FileNotFoundError(f"No se encontró el manifest de entrada: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    global_view = build_global_view(manifest)
    graph = build_graph(global_view)
    mermaid = to_mermaid(graph)

    with out_global.open("w", encoding="utf-8") as f:
        json.dump(global_view, f, indent=2, ensure_ascii=False)
    with out_graph.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    out_mermaid.write_text(mermaid, encoding="utf-8")

    print(json.dumps({
        "input": str(in_path),
        "output_global": str(out_global),
        "output_graph": str(out_graph),
        "output_mermaid": str(out_mermaid),
        "familias_globales": len(global_view["familias_globales"]),
        "scripts": len(global_view["scripts"]),
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
