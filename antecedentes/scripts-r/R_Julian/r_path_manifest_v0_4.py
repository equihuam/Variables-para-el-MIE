#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple


@dataclass
class ExpresionRuta:
    raw: str
    tipo: str
    variables_usadas: List[str] = field(default_factory=list)


@dataclass
class EnsambleRuta:
    ruta: Optional[str] = None
    plantilla: Optional[str] = None
    base: Optional[str] = None
    relativa: Optional[str] = None
    dinamica: bool = False
    derivada_de: Optional[str] = None
    origen_lista: Optional[str] = None
    patron: Optional[str] = None


@dataclass
class RegistroRuta:
    orden: int
    rol: str
    fuente: str
    expresion: ExpresionRuta
    ensamble: EnsambleRuta


@dataclass
class ManifiestoScript:
    nombre_script: str
    bases_entrada: Dict[str, str] = field(default_factory=dict)
    bases_salida: Dict[str, str] = field(default_factory=dict)
    componentes_recurrentes: Dict[str, str] = field(default_factory=dict)
    entradas: Dict[str, RegistroRuta] = field(default_factory=dict)
    salidas: Dict[str, RegistroRuta] = field(default_factory=dict)
    variables_ruta: Dict[str, str] = field(default_factory=dict)
    dependencias_script: Dict[str, RegistroRuta] = field(default_factory=dict)


ASSIGNMENT_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:<-|=)\s*(.+?)\s*$', re.MULTILINE)
FOR_ALIAS_RE = re.compile(r'for\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)')
FUNC_CALL_RE_TEMPLATE = r'\b{func}\s*\('
VAR_NAME_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
STRING_LITERAL_RE = re.compile(r'^["\']([^"\']+)["\']$')
INDEXED_VAR_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\[.*\]$')
PATH_LIKE_RE = re.compile(r'(/|\\|\.[Rr]$|\.[A-Za-z0-9]{2,5}$)')

INPUT_FUNCTIONS = [
    'rast', 'vect', 'st_read', 'read.csv', 'readRDS', 'load', 'list.files', 'read_excel', 'fread'
]
OUTPUT_FUNCTIONS = ['writeRaster', 'saveRDS', 'write.csv']
SOURCE_FUNCTIONS = ['source']

BASE_NAME_HINTS_INPUT = {'READPATH', 'INPUTPATH', 'INPATH', 'SRC', 'SOURCEPATH'}
BASE_NAME_HINTS_OUTPUT = {'WRITEPATH', 'OUTPATH', 'OUTPUTPATH', 'DST', 'DESTPATH'}
ROOT_BASE_HINTS = ('./data/', './data_crude/', './data_features/', './data_training_tables/')


def strip_comments(line: str) -> str:
    out = []
    in_str = False
    q = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            out.append(ch)
            if ch == q:
                in_str = False
                q = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            q = ch
            out.append(ch)
            i += 1
            continue
        if ch == '#':
            break
        out.append(ch)
        i += 1
    return ''.join(out)


def preprocess_text(text: str) -> str:
    return '\n'.join(strip_comments(line) for line in text.splitlines())


def split_top_level_args(s: str) -> List[str]:
    args: List[str] = []
    cur: List[str] = []
    depth_paren = depth_brack = depth_brace = 0
    in_str = False
    q = None
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            cur.append(ch)
            if ch == q:
                in_str = False
                q = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            q = ch
            cur.append(ch)
        elif ch == '(':
            depth_paren += 1
            cur.append(ch)
        elif ch == ')':
            depth_paren -= 1
            cur.append(ch)
        elif ch == '[':
            depth_brack += 1
            cur.append(ch)
        elif ch == ']':
            depth_brack -= 1
            cur.append(ch)
        elif ch == '{':
            depth_brace += 1
            cur.append(ch)
        elif ch == '}':
            depth_brace -= 1
            cur.append(ch)
        elif ch == ',' and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            args.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    tail = ''.join(cur).strip()
    if tail:
        args.append(tail)
    return args


def extract_balanced_call(text: str, start_idx: int) -> Tuple[str, int]:
    depth = 0
    in_str = False
    q = None
    i = start_idx
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == q:
                in_str = False
                q = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            q = ch
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1], i + 1
        i += 1
    raise ValueError('Unbalanced call expression')


def find_function_calls(text: str, func_name: str) -> List[str]:
    pattern = re.compile(FUNC_CALL_RE_TEMPLATE.format(func=re.escape(func_name)))
    calls: List[str] = []
    for m in pattern.finditer(text):
        open_paren_idx = m.end() - 1
        call_text, _ = extract_balanced_call(text, open_paren_idx)
        calls.append(f'{func_name}{call_text}')
    return calls


def parse_named_arg(call_text: str, arg_name: str) -> Optional[str]:
    start = call_text.find('(')
    if start == -1 or not call_text.endswith(')'):
        return None
    inner = call_text[start + 1:-1]
    for arg in split_top_level_args(inner):
        if re.match(rf'^{re.escape(arg_name)}\s*=', arg):
            return arg.split('=', 1)[1].strip()
    return None


def parse_first_arg(call_text: str) -> Optional[str]:
    start = call_text.find('(')
    if start == -1 or not call_text.endswith(')'):
        return None
    inner = call_text[start + 1:-1].strip()
    if not inner:
        return None
    args = split_top_level_args(inner)
    return args[0].strip() if args else None


def expr_type(expr: str) -> str:
    expr = expr.strip()
    if STRING_LITERAL_RE.match(expr):
        return 'literal'
    if expr.startswith('file.path('):
        return 'file.path'
    if expr.startswith('paste0('):
        return 'paste0'
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr):
        return 'variable'
    return 'mixta'


def unquote(s: str) -> Optional[str]:
    m = STRING_LITERAL_RE.match(s.strip())
    return m.group(1) if m else None


def extract_variables_used(expr: str) -> List[str]:
    if unquote(expr) is not None:
        return []
    tokens = VAR_NAME_RE.findall(expr)
    blacklist = {
        'file', 'path', 'paste0', 'c', 'TRUE', 'FALSE', 'NA',
        'showWarnings', 'overwrite', 'full', 'names', 'recursive',
        'pattern', 'filename', 'sep', 'header', 'gdal', 'datatype',
        'col_names', 'data', 'table', 'type', 'digits', 'full.names',
        'x', 'y', 'z', 'method', 'geom', 'keepgeom', 'split'
    }
    out: List[str] = []
    for tok in tokens:
        if tok not in blacklist and not tok.islower():
            out.append(tok)
    seen = set()
    result = []
    for x in out:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def normalize_join(parts: List[str]) -> str:
    clean = []
    for idx, part in enumerate(parts):
        if idx == 0:
            clean.append(part.rstrip('/'))
        else:
            clean.append(part.strip('/'))
    return '/'.join(x for x in clean if x != '')


def looks_like_path(value: str) -> bool:
    return bool(value and PATH_LIKE_RE.search(value))


def resolve_expr(expr: str, symbols: Dict[str, str], depth: int = 0) -> Tuple[Optional[str], bool]:
    expr = expr.strip()
    if depth > 12:
        return None, True

    lit = unquote(expr)
    if lit is not None:
        return lit, False

    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr):
        if expr in symbols:
            return resolve_expr(symbols[expr], symbols, depth + 1)
        return None, True

    m_idx = INDEXED_VAR_RE.match(expr)
    if m_idx:
        root = m_idx.group(1)
        if root in symbols:
            return resolve_expr(symbols[root], symbols, depth + 1)
        return None, True

    if expr.startswith('file.path(') and expr.endswith(')'):
        inner = expr[len('file.path('):-1]
        args = split_top_level_args(inner)
        parts = []
        any_dynamic = False
        for arg in args:
            resolved, dynamic = resolve_expr(arg, symbols, depth + 1)
            if resolved is None:
                resolved = '{expr}'
                dynamic = True
            parts.append(resolved)
            any_dynamic = any_dynamic or dynamic
        return normalize_join(parts), any_dynamic

    if expr.startswith('paste0(') and expr.endswith(')'):
        inner = expr[len('paste0('):-1]
        args = split_top_level_args(inner)
        pieces = []
        any_dynamic = False
        for arg in args:
            resolved, dynamic = resolve_expr(arg, symbols, depth + 1)
            if resolved is None:
                resolved = '{expr}'
                dynamic = True
            pieces.append(resolved)
            any_dynamic = any_dynamic or dynamic
        return ''.join(pieces), any_dynamic

    return None, True


def infer_base_candidate(resolved: str) -> str:
    clean = resolved.rstrip('/')
    name = PurePosixPath(clean).name
    if '.' in name:
        return str(PurePosixPath(clean).parent)
    return clean


def common_prefix_paths(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    parts_list = [PurePosixPath(p.rstrip('/')).parts for p in paths if p]
    if not parts_list:
        return None
    common = []
    for group in zip(*parts_list):
        if all(x == group[0] for x in group):
            common.append(group[0])
        else:
            break
    if not common:
        return None
    prefix = str(PurePosixPath(*common))
    return prefix if prefix else None


def path_dirname(path: str) -> str:
    return str(PurePosixPath(path.rstrip('/')).parent)


def split_base_relative(path: str, candidate_bases: List[str]) -> Tuple[Optional[str], Optional[str]]:
    best = None
    for base in candidate_bases:
        base_clean = base.rstrip('/')
        if path == base_clean or path.startswith(base_clean + '/'):
            if best is None or len(base_clean) > len(best):
                best = base_clean
    if best is None:
        return None, None
    rel = path[len(best):].lstrip('/')
    return best, rel


def collect_repeated_chunks(records: List[RegistroRuta]) -> Dict[str, str]:
    counts: Dict[str, int] = {}
    for rec in records:
        candidate = rec.ensamble.plantilla or rec.ensamble.ruta
        if not candidate:
            continue
        for part in PurePosixPath(candidate).parts:
            if part in ('/', '.', ''):
                continue
            counts[part] = counts.get(part, 0) + 1
    return {str(i): chunk for i, (chunk, n) in enumerate(counts.items(), start=1) if n >= 2}


def extract_all_assignments(text: str) -> Dict[str, str]:
    raw_symbols: Dict[str, str] = {}
    for m in ASSIGNMENT_RE.finditer(text):
        var = m.group(1)
        expr = m.group(2).strip()
        raw_symbols[var] = expr

    # complementa asignaciones multilínea de llamadas que suelen romper el regex por línea
    assign_call_pat = re.compile(r'(^|\n)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:<-|=)\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*\(', re.MULTILINE)
    for m in assign_call_pat.finditer(text):
        var = m.group(2)
        func = m.group(3)
        open_paren_idx = m.end() - 1
        try:
            call_text, _ = extract_balanced_call(text, open_paren_idx)
        except ValueError:
            continue
        raw_symbols[var] = f'{func}{call_text}'
    return raw_symbols


def extract_path_symbols(all_symbols: Dict[str, str]) -> Dict[str, str]:
    symbols: Dict[str, str] = {}
    for var, expr in all_symbols.items():
        if (
            unquote(expr) is not None
            or expr.startswith('file.path(')
            or expr.startswith('paste0(')
            or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr)
        ):
            resolved, dynamic = resolve_expr(expr, all_symbols)
            upper = var.upper()
            if resolved and looks_like_path(resolved):
                symbols[var] = expr
            elif upper in BASE_NAME_HINTS_INPUT or upper in BASE_NAME_HINTS_OUTPUT or upper.startswith(('READ', 'WRITE', 'OUT')) or upper.endswith('_DIR'):
                symbols[var] = expr
    return symbols


def extract_listfile_symbols(all_symbols: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for var, expr in all_symbols.items():
        e = expr.strip()
        if not e.startswith('list.files(') or not e.endswith(')'):
            continue
        first = parse_first_arg(e)
        if not first:
            continue
        dir_resolved, _ = resolve_expr(first, all_symbols)
        pattern = parse_named_arg(e, 'pattern')
        out[var] = {
            'raw': first,
            'directorio': dir_resolved or '',
            'patron': unquote(pattern) if pattern else ''
        }
    return out


def extract_for_aliases(text: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in FOR_ALIAS_RE.finditer(text)}


def indexed_root(expr: str) -> Optional[str]:
    m = INDEXED_VAR_RE.match(expr.strip())
    return m.group(1) if m else None


def list_origin_for_expr(expr: str, list_symbols: Dict[str, Dict[str, str]], aliases: Dict[str, str]) -> Optional[Dict[str, str]]:
    e = expr.strip()
    root = None
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', e):
        root = aliases.get(e) or (e if e in list_symbols else None)
    else:
        idx_root = indexed_root(e)
        if idx_root:
            root = aliases.get(idx_root) or idx_root
    if root and root in list_symbols:
        data = dict(list_symbols[root])
        data['variable_lista'] = root
        return data
    return None


def candidate_is_real_file_input(raw_expr: str, path_symbols: Dict[str, str], source: str,
                                 list_symbols: Dict[str, Dict[str, str]], aliases: Dict[str, str]) -> bool:
    resolved, dynamic = resolve_expr(raw_expr, path_symbols)
    if resolved and looks_like_path(resolved):
        return True
    if list_origin_for_expr(raw_expr, list_symbols, aliases):
        return True
    expr = raw_expr.strip()
    if source in ('list.files', 'source'):
        return True
    if expr.startswith(('file.path(', 'paste0(')):
        return True
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr):
        return expr in path_symbols
    return False


def build_record(orden: int, rol: str, fuente: str, raw_expr: str, path_symbols: Dict[str, str],
                 list_symbols: Dict[str, Dict[str, str]], aliases: Dict[str, str]) -> RegistroRuta:
    usadas = extract_variables_used(raw_expr)
    tipo = expr_type(raw_expr)
    resolved_str, dynamic = resolve_expr(raw_expr, path_symbols)
    origin = list_origin_for_expr(raw_expr, list_symbols, aliases)
    ensamble = EnsambleRuta(
        ruta=resolved_str if resolved_str and not dynamic else None,
        plantilla=resolved_str if resolved_str and dynamic else None,
        dinamica=dynamic,
    )
    if origin:
        ensamble.derivada_de = origin.get('variable_lista')
        ensamble.origen_lista = origin.get('directorio') or None
        ensamble.patron = origin.get('patron') or None
        if not ensamble.ruta and not ensamble.plantilla and ensamble.origen_lista:
            ensamble.plantilla = ensamble.origen_lista.rstrip('/') + '/{archivo_lista}'
            ensamble.dinamica = True
    return RegistroRuta(
        orden=orden,
        rol=rol,
        fuente=fuente,
        expresion=ExpresionRuta(raw=raw_expr, tipo=tipo, variables_usadas=usadas),
        ensamble=ensamble,
    )


def extract_input_records(text: str, path_symbols: Dict[str, str], list_symbols: Dict[str, Dict[str, str]],
                          aliases: Dict[str, str]) -> List[RegistroRuta]:
    records: List[RegistroRuta] = []
    order = 1
    for func in INPUT_FUNCTIONS:
        for call in find_function_calls(text, func):
            raw = parse_first_arg(call)
            if not raw:
                continue
            if not candidate_is_real_file_input(raw, path_symbols, func, list_symbols, aliases):
                continue
            records.append(build_record(order, 'entra', func, raw, path_symbols, list_symbols, aliases))
            order += 1
    return records


def extract_dependency_records(text: str, path_symbols: Dict[str, str], list_symbols: Dict[str, Dict[str, str]],
                               aliases: Dict[str, str]) -> List[RegistroRuta]:
    records: List[RegistroRuta] = []
    order = 1
    for func in SOURCE_FUNCTIONS:
        for call in find_function_calls(text, func):
            raw = parse_first_arg(call)
            if not raw:
                continue
            if not candidate_is_real_file_input(raw, path_symbols, func, list_symbols, aliases):
                continue
            records.append(build_record(order, 'apoyo', func, raw, path_symbols, list_symbols, aliases))
            order += 1
    return records


def extract_output_records(text: str, path_symbols: Dict[str, str], list_symbols: Dict[str, Dict[str, str]],
                           aliases: Dict[str, str]) -> List[RegistroRuta]:
    records: List[RegistroRuta] = []
    order = 1
    for call in find_function_calls(text, 'writeRaster'):
        raw = parse_named_arg(call, 'filename')
        if raw:
            records.append(build_record(order, 'sale', 'writeRaster', raw, path_symbols, list_symbols, aliases))
            order += 1
    for call in find_function_calls(text, 'saveRDS'):
        inner = call[call.find('(') + 1:-1]
        args = split_top_level_args(inner)
        raw = args[1].strip() if len(args) >= 2 else None
        if raw:
            records.append(build_record(order, 'sale', 'saveRDS', raw, path_symbols, list_symbols, aliases))
            order += 1
    for call in find_function_calls(text, 'write.csv'):
        raw = parse_named_arg(call, 'file')
        if raw is None:
            inner = call[call.find('(') + 1:-1]
            args = split_top_level_args(inner)
            raw = args[1].strip() if len(args) >= 2 else None
        if raw:
            records.append(build_record(order, 'sale', 'write.csv', raw, path_symbols, list_symbols, aliases))
            order += 1
    return records


def add_candidate_base(container: Dict[str, str], base: Optional[str]) -> None:
    if not base:
        return
    base = base.rstrip('/')
    if not base:
        return
    if base not in container.values():
        container[str(len(container) + 1)] = base


def root_hint_base(path: str) -> Optional[str]:
    for hint in ROOT_BASE_HINTS:
        if path.startswith(hint):
            return hint.strip('/').lstrip('./') if hint.startswith('./') else hint.rstrip('/')
    return None


def classify_bases(path_symbols: Dict[str, str], list_symbols: Dict[str, Dict[str, str]],
                   inputs: List[RegistroRuta], outputs: List[RegistroRuta]) -> Tuple[Dict[str, str], Dict[str, str]]:
    input_bases: Dict[str, str] = {}
    output_bases: Dict[str, str] = {}

    vars_entrada = {r.expresion.raw.strip() for r in inputs if r.expresion.tipo == 'variable'}
    vars_salida = {r.expresion.raw.strip() for r in outputs if r.expresion.tipo == 'variable'}

    for var, expr in path_symbols.items():
        resolved, dynamic = resolve_expr(expr, path_symbols)
        if not resolved or dynamic:
            continue
        upper = var.upper()
        inferred = infer_base_candidate(resolved)
        if var in vars_entrada or upper in BASE_NAME_HINTS_INPUT or upper.startswith(('READ', 'IN')):
            add_candidate_base(input_bases, inferred)
        elif var in vars_salida or upper in BASE_NAME_HINTS_OUTPUT or upper.startswith(('WRITE', 'OUT')):
            add_candidate_base(output_bases, inferred)

    for meta in list_symbols.values():
        add_candidate_base(input_bases, infer_base_candidate(meta.get('directorio', '')))

    for rec in inputs:
        candidate = rec.ensamble.ruta or rec.ensamble.plantilla or rec.ensamble.origen_lista
        if candidate:
            add_candidate_base(input_bases, root_hint_base(candidate) or infer_base_candidate(candidate))

    for rec in outputs:
        candidate = rec.ensamble.ruta or rec.ensamble.plantilla
        if candidate:
            add_candidate_base(output_bases, root_hint_base(candidate) or infer_base_candidate(candidate))

    return input_bases, output_bases


def enrich_base_relative(records: List[RegistroRuta], base_candidates: List[str], role: str) -> None:
    candidates = [b.rstrip('/') for b in base_candidates if b]
    for rec in records:
        candidate = rec.ensamble.ruta or rec.ensamble.plantilla or rec.ensamble.origen_lista
        if not candidate:
            continue
        base, rel = split_base_relative(candidate, candidates)
        if base is None and role == 'salida' and (rec.ensamble.ruta or rec.ensamble.plantilla):
            base = path_dirname(rec.ensamble.ruta or rec.ensamble.plantilla)
            rel = PurePosixPath((rec.ensamble.ruta or rec.ensamble.plantilla).rstrip('/')).name
        if rec.ensamble.origen_lista and rec.ensamble.derivada_de and rec.ensamble.origen_lista not in candidates:
            base, rel = split_base_relative(rec.ensamble.origen_lista, candidates)
        rec.ensamble.base = base
        if rec.ensamble.relativa is None:
            rec.ensamble.relativa = rel


def compact(obj):
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cv = compact(v)
            if cv is None:
                continue
            if cv == {} or cv == []:
                continue
            if k == 'dinamica' and cv is False:
                continue
            out[k] = cv
        return out
    if isinstance(obj, list):
        out = [compact(x) for x in obj]
        return [x for x in out if x is not None and x != {} and x != []]
    return obj


def manifest_for_script(script_path: Path) -> ManifiestoScript:
    raw_text = script_path.read_text(encoding='utf-8', errors='ignore')
    text = preprocess_text(raw_text)
    all_symbols = extract_all_assignments(text)
    path_symbols = extract_path_symbols(all_symbols)
    list_symbols = extract_listfile_symbols(all_symbols)
    aliases = extract_for_aliases(text)

    manifest = ManifiestoScript(nombre_script=script_path.name)
    manifest.variables_ruta = path_symbols.copy()

    inputs = extract_input_records(text, path_symbols, list_symbols, aliases)
    outputs = extract_output_records(text, path_symbols, list_symbols, aliases)
    deps = extract_dependency_records(text, path_symbols, list_symbols, aliases)

    manifest.bases_entrada, manifest.bases_salida = classify_bases(path_symbols, list_symbols, inputs, outputs)
    enrich_base_relative(inputs, list(manifest.bases_entrada.values()), 'entrada')
    enrich_base_relative(outputs, list(manifest.bases_salida.values()), 'salida')
    enrich_base_relative(deps, list(manifest.bases_entrada.values()), 'entrada')

    manifest.componentes_recurrentes = collect_repeated_chunks(inputs + outputs + deps)
    manifest.entradas = {str(i): rec for i, rec in enumerate(inputs, start=1)}
    manifest.salidas = {str(i): rec for i, rec in enumerate(outputs, start=1)}
    manifest.dependencias_script = {str(i): rec for i, rec in enumerate(deps, start=1)}
    return manifest


def manifest_to_dict(m: ManifiestoScript) -> Dict:
    out = {
        'bases': {
            'entrada': m.bases_entrada,
            'salida': m.bases_salida,
        },
        'componentes_recurrentes': m.componentes_recurrentes,
        'variables_ruta': m.variables_ruta,
        'entra': {k: compact(v) for k, v in m.entradas.items()},
        'sale': {k: compact(v) for k, v in m.salidas.items()},
    }
    if m.dependencias_script:
        out['dependencias_script'] = {k: compact(v) for k, v in m.dependencias_script.items()}
    return compact(out)


def build_project_manifest(paths: List[Path]) -> Dict[str, Dict]:
    return {p.name: manifest_to_dict(manifest_for_script(p)) for p in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extrae entradas, salidas y procedencia de rutas desde scripts R del directorio.')
    parser.add_argument('-d', '--directory', default=None,
                        help='Directorio a recorrer. Si no se indica, usa el directorio donde vive este script.')
    parser.add_argument('-o', '--output', default='r_script_paths_manifest_v0_4.json',
                        help='Nombre o ruta del JSON de salida.')
    parser.add_argument('--pretty', action='store_true', help='Imprime también el JSON en consola.')
    return parser.parse_args()


def collect_r_files(base_dir: Path, self_name: str) -> List[Path]:
    files = sorted(base_dir.glob('*.R')) + sorted(base_dir.glob('*.r'))
    out: List[Path] = []
    seen = set()
    for f in files:
        if f.name == self_name:
            continue
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base_dir = Path(args.directory).resolve() if args.directory else script_dir
    r_files = collect_r_files(base_dir, Path(__file__).name)
    if not r_files:
        raise SystemExit(f'No se encontraron archivos .R en: {base_dir}')
    manifest = build_project_manifest(r_files)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.pretty:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print(f'JSON escrito en: {out_path}')
        print(f'Scripts procesados: {len(r_files)}')


if __name__ == '__main__':
    main()
