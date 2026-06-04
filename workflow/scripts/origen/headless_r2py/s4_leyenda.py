# -*- coding: utf-8 -*-
"""
Agrega columnas de leyenda a una tabla CSV a partir de códigos de:
- zonas de vida
- estado de conservación de dunas

Traducción headless del script R original.
"""

# TODO: Definir si valores sin correspondencia en las leyendas deben
# permanecer solo como advertencia o convertirse en error de validación.

import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(".")
INPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v5.csv"
OUTPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v6.csv"

ZVH_COL = "zonas_vida_h_aligned"
ZVH_LEGEND_COL = "zvh_legend"

CONSERV_COL = "dunas_2024_conserv_ed"
CONSERV_LEGEND_COL = "conserv_dun_legend"

ZONAS_DICT = {
    "4": "desierto templado calido",
    "5": "desierto subtropical",
    "10": "matorral desertico",
    "11": "matorral desertico premontano",
    "12": "matorral desertico montano bajo",
    "13": "bosque espinoso",
    "14": "bosque muy seco",
    "15": "bosque seco premontano",
    "17": "bosque subhumedo",
    "18": "bosque subhumedo premontano",
    "22": "bosque humedo premontano",
    "26": "bosque lluvioso",
    "27": "bosque lluvioso premontano",
}

CONSERV_DICT = {
    "1": "muy bueno",
    "2": "bueno",
    "3": "regular",
    "4": "malo",
    "5": "muy malo",
}


# ---------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def validate_input_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"No existe el archivo de entrada ({label}): {path}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def distinct_sorted_non_null(series: pd.Series) -> list[str]:
    vals = series.dropna().astype(str).unique().tolist()
    return sorted(vals, key=lambda x: (len(x), x))


def report_unmapped_values(df: pd.DataFrame, source_col: str, legend_col: str) -> list[str]:
    mask = df[source_col].notna() & df[legend_col].isna()
    vals = df.loc[mask, source_col].astype(str).drop_duplicates().tolist()
    return sorted(vals, key=lambda x: (len(x), x))


# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def add_legend_column(
    df: pd.DataFrame,
    source_col: str,
    legend_col: str,
    mapping: dict[str, str],
) -> pd.DataFrame:
    if source_col not in df.columns:
        fail(f"No existe la columna requerida: {source_col}")

    out = df.copy()
    out[source_col] = out[source_col].astype("string")
    out[legend_col] = out[source_col].map(mapping)

    return out


def main() -> int:
    validate_input_file(INPUT_CSV, "tabla final_data_v5")
    ensure_parent(OUTPUT_CSV)

    log("Leyendo tabla...")
    variables_ie = pd.read_csv(INPUT_CSV)

    # Verificar tipo y valores únicos como en el R original
    if ZVH_COL not in variables_ie.columns:
        fail(f"No existe la columna {ZVH_COL}")

    log(f"Tipo de dato de {ZVH_COL}: {variables_ie[ZVH_COL].dtype}")
    log(f"Valores únicos en {ZVH_COL}:")
    log(str(distinct_sorted_non_null(variables_ie[ZVH_COL])))

    log("\nAgregando leyenda de zonas de vida...")
    variables_ie = add_legend_column(
        df=variables_ie,
        source_col=ZVH_COL,
        legend_col=ZVH_LEGEND_COL,
        mapping=ZONAS_DICT,
    )

    unmapped_zvh = report_unmapped_values(variables_ie, ZVH_COL, ZVH_LEGEND_COL)
    
    # TODO: Convertir este chequeo en criterio formal de validación si se exige
    # cobertura completa del diccionario de clases.
    if unmapped_zvh:
        log("Valores de zonas de vida sin correspondencia:")
        log(str(unmapped_zvh))

    log("\nAgregando leyenda de conservación de dunas...")
    variables_ie = add_legend_column(
        df=variables_ie,
        source_col=CONSERV_COL,
        legend_col=CONSERV_LEGEND_COL,
        mapping=CONSERV_DICT,
    )

    unmapped_conserv = report_unmapped_values(variables_ie, CONSERV_COL, CONSERV_LEGEND_COL)
    if unmapped_conserv:
        log("Valores de conservación sin correspondencia:")
        log(str(unmapped_conserv))

    log("\nGuardando CSV enriquecido...")
    variables_ie.to_csv(OUTPUT_CSV, index=False, na_rep="")

    log("\n----------------------------------------")
    log("Proceso completado.")
    log(f"CSV guardado en: {OUTPUT_CSV}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)