#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple


@dataclass
class PathExpression:
    raw: str
    expr_type: str
    variables_used: List[str] = field(default_factory=list)


@dataclass
class ResolvedPath:
    path: Optional[str] = None
    template: Optional[str] = None
    base: Optional[str] = None
    relative: Optional[str] = None
    dynamic: bool = False


@dataclass
class PathRecord:
    order: int
    role: str
    source: str
    expression: PathExpression
    resolved: ResolvedPath


@dataclass
class ScriptManifest:
    script_name: str
    input_bases: Dict[str, str] = field(default_factory=dict)
    output_bases: Dict[str, str] = field(default_factory=dict)
    repeated_chunks: Dict[str, str] = field(default_factory=dict)
    inputs: Dict[str, PathRecord] = field(default_factory=dict)
    outputs: Dict[str, PathRecord] = field(default_factory=dict)
    assigned_paths: Dict[str, str] = field(default_factory=dict)


ASSIGNMENT_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:<-|=)\s*(.+?)\s*$', re.MULTILINE)
FUNC_CALL_RE_TEMPLATE = r'\b{func}\s*\('
VAR_NAME_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
STRING_LITERAL_RE = re.compile(r'^["\']([^"\']+)["\']$')

INPUT_FUNCTIONS = [
    'rast', 'vect', 'st_read', 'read.csv', 'readRDS', 'load', 'list.files', 'read_excel'
]

BASE_NAME_HINTS_INPUT = {'READPATH', 'INPUTPATH', 'INPATH', 'SRC', 'SOURCEPATH'}
BASE_NAME_HINTS_OUTPUT = {'WRITEPATH', 'OUTPATH', 'OUTPUTPATH', 'DST', 'DESTPATH'}


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
    depth = 0
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
            depth += 1
            cur.append(ch)
        elif ch == ')':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
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
    return 'mixed'


def extract_variables_used(expr: str) -> List[str]:
    if unquote(expr) is not None:
        return []
    tokens = VAR_NAME_RE.findall(expr)
    blacklist = {
        'file', 'path', 'paste0', 'c', 'TRUE', 'FALSE', 'NA',
        'showWarnings', 'overwrite', 'full', 'names', 'recursive',
        'pattern', 'filename', 'sep', 'header', 'gdal', 'datatype',
        'col_names'
    }
    out = []
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


def unquote(s: str) -> Optional[str]:
    m = STRING_LITERAL_RE.match(s.strip())
    return m.group(1) if m else None


def normalize_join(parts: List[str]) -> str:
    clean = []
    for idx, part in enumerate(parts):
        if idx == 0:
            clean.append(part.rstrip('/'))
        else:
            clean.append(part.strip('/'))
    return '/'.join(x for x in clean if x != '')


def resolve_expr(expr: str, symbols: Dict[str, str], depth: int = 0) -> Tuple[Optional[str], bool]:
    expr = expr.strip()
    if depth > 8:
        return None, True

    lit = unquote(expr)
    if lit is not None:
        return lit, False

    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr):
        if expr in symbols:
            return resolve_expr(symbols[expr], symbols, depth + 1)
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


def common_prefix_paths(paths: List[str]) -> Optional[str]:
    parts_list = []
    for p in paths:
        norm = p.rstrip('/')
        parts_list.append(PurePosixPath(norm).parts)
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


def collect_repeated_chunks(records: List[PathRecord]) -> Dict[str, str]:
    counts: Dict[str, int] = {}
    for rec in records:
        candidate = rec.resolved.template or rec.resolved.path
        if not candidate:
            continue
        for part in PurePosixPath(candidate).parts:
            if part in ('/', '.', ''):
                continue
            counts[part] = counts.get(part, 0) + 1
    return {str(i): chunk for i, (chunk, n) in enumerate(counts.items(), start=1) if n >= 2}


def extract_assignments(text: str) -> Dict[str, str]:
    symbols: Dict[str, str] = {}
    for m in ASSIGNMENT_RE.finditer(text):
        var = m.group(1)
        expr = m.group(2).strip()
        if (
            unquote(expr) is not None
            or expr.startswith('file.path(')
            or expr.startswith('paste0(')
            or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr)
        ):
            symbols[var] = expr
    return symbols


def build_record(order: int, role: str, source: str, raw_expr: str, symbols: Dict[str, str]) -> PathRecord:
    used_vars = extract_variables_used(raw_expr)
    ex_type = expr_type(raw_expr)
    resolved_str, dynamic = resolve_expr(raw_expr, symbols)
    return PathRecord(
        order=order,
        role=role,
        source=source,
        expression=PathExpression(raw=raw_expr, expr_type=ex_type, variables_used=used_vars),
        resolved=ResolvedPath(
            path=resolved_str if resolved_str and not dynamic else None,
            template=resolved_str if resolved_str and dynamic else None,
            dynamic=dynamic,
        ),
    )


def extract_input_records(text: str, symbols: Dict[str, str]) -> List[PathRecord]:
    records: List[PathRecord] = []
    order = 1
    for func in INPUT_FUNCTIONS:
        for call in find_function_calls(text, func):
            raw = parse_first_arg(call)
            if not raw:
                continue
            records.append(build_record(order, 'entra', func, raw, symbols))
            order += 1
    return records


def extract_output_records(text: str, symbols: Dict[str, str]) -> List[PathRecord]:
    records: List[PathRecord] = []
    order = 1

    for call in find_function_calls(text, 'writeRaster'):
        raw = parse_named_arg(call, 'filename')
        if raw:
            records.append(build_record(order, 'sale', 'writeRaster', raw, symbols))
            order += 1

    for call in find_function_calls(text, 'saveRDS'):
        inner = call[call.find('(') + 1:-1]
        args = split_top_level_args(inner)
        raw = args[1].strip() if len(args) >= 2 else None
        if raw:
            records.append(build_record(order, 'sale', 'saveRDS', raw, symbols))
            order += 1

    for call in find_function_calls(text, 'write.csv'):
        raw = parse_named_arg(call, 'file')
        if raw is None:
            inner = call[call.find('(') + 1:-1]
            args = split_top_level_args(inner)
            raw = args[1].strip() if len(args) >= 2 else None
        if raw:
            records.append(build_record(order, 'sale', 'write.csv', raw, symbols))
            order += 1

    return records


def classify_bases(symbols: Dict[str, str], inputs: List[PathRecord], outputs: List[PathRecord]) -> Tuple[Dict[str, str], Dict[str, str]]:
    input_bases: Dict[str, str] = {}
    output_bases: Dict[str, str] = {}
    in_idx = 1
    out_idx = 1

    for var, expr in symbols.items():
        resolved, dynamic = resolve_expr(expr, symbols)
        if not resolved or dynamic:
            continue
        upper = var.upper()
        if upper in BASE_NAME_HINTS_INPUT or upper.startswith('READ') or upper.startswith('IN'):
            input_bases[str(in_idx)] = resolved
            in_idx += 1
        elif upper in BASE_NAME_HINTS_OUTPUT or upper.startswith('WRITE') or upper.startswith('OUT'):
            output_bases[str(out_idx)] = resolved
            out_idx += 1

    # Autodeducción suave si no hubo variables-base explícitas
    if not input_bases:
        in_paths = [r.resolved.path for r in inputs if r.resolved.path]
        prefix = common_prefix_paths(in_paths)
        if prefix:
            if len(in_paths) == 1:
                prefix = str(PurePosixPath(in_paths[0]).parent)
            input_bases['1'] = prefix

    if not output_bases:
        out_paths = [(r.resolved.path or r.resolved.template) for r in outputs if (r.resolved.path or r.resolved.template)]
        prefix = common_prefix_paths(out_paths)
        if prefix:
            if len(out_paths) == 1:
                prefix = str(PurePosixPath(out_paths[0]).parent)
            output_bases['1'] = prefix

    return input_bases, output_bases


def enrich_base_relative(records: List[PathRecord], base_candidates: List[str]) -> None:
    resolved_paths = [r.resolved.path or r.resolved.template for r in records if (r.resolved.path or r.resolved.template)]
    auto_prefix = common_prefix_paths([p for p in resolved_paths if p])
    candidates = list(base_candidates)
    if auto_prefix and auto_prefix not in candidates:
        candidates.append(auto_prefix)

    for rec in records:
        candidate = rec.resolved.path or rec.resolved.template
        if not candidate:
            continue
        base, rel = split_base_relative(candidate, candidates)
        rec.resolved.base = base
        rec.resolved.relative = rel


def manifest_for_script(script_path: Path) -> ScriptManifest:
    raw_text = script_path.read_text(encoding='utf-8', errors='ignore')
    text = preprocess_text(raw_text)
    symbols = extract_assignments(text)

    manifest = ScriptManifest(script_name=script_path.name)
    manifest.assigned_paths = symbols.copy()

    inputs = extract_input_records(text, symbols)
    outputs = extract_output_records(text, symbols)

    manifest.input_bases, manifest.output_bases = classify_bases(symbols, inputs, outputs)

    enrich_base_relative(inputs, list(manifest.input_bases.values()))
    enrich_base_relative(outputs, list(manifest.output_bases.values()))

    manifest.repeated_chunks = collect_repeated_chunks(inputs + outputs)
    manifest.inputs = {str(i): rec for i, rec in enumerate(inputs, start=1)}
    manifest.outputs = {str(i): rec for i, rec in enumerate(outputs, start=1)}
    return manifest


def manifest_to_dict(m: ScriptManifest) -> Dict:
    return {
        'bases': {
            'entrada': m.input_bases,
            'salida': m.output_bases,
        },
        'componentes_recurrentes': m.repeated_chunks,
        'variables_ruta': m.assigned_paths,
        'entra': {k: asdict(v) for k, v in m.inputs.items()},
        'sale': {k: asdict(v) for k, v in m.outputs.items()},
    }


def build_project_manifest(paths: List[Path]) -> Dict[str, Dict]:
    return {p.name: manifest_to_dict(manifest_for_script(p)) for p in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extrae entradas, salidas, bases y componentes de rutas desde todos los scripts R del directorio del script.'
    )
    parser.add_argument(
        '-d', '--directory',
        default=None,
        help='Directorio a recorrer. Si no se indica, usa el directorio donde vive este script.'
    )
    parser.add_argument(
        '-o', '--output',
        default='r_script_paths_manifest_v0_2.json',
        help='Nombre o ruta del JSON de salida.'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Imprime también el JSON en consola.'
    )
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
