# -*- coding: utf-8 -*-
"""
Traducción de un script R que:
1. Reescribe un CSV de entrada con otro nombre (sin eliminar NA).
2. Divide otro CSV en subconjuntos 70/30 y guarda ambos resultados.

Versión headless para workflow-iie.
"""

import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(".")

# Bloque 1: reescritura simple
INPUT_COPY_CSV = Path(r"C:/Users/Octavio/Dropbox/variables_ie_c/tablas_finales/final_data_v.csv")
OUTPUT_COPY_CSV = Path(r"C:/Users/Octavio/Dropbox/variables_ie_c/tablas_finales/final_data_v3_s_NA.csv")

# Bloque 2: partición 70/30
INPUT_SPLIT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v5.csv"
OUTPUT_TRAIN_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v5_70.csv"
OUTPUT_TEST_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v5_30.csv"

RANDOM_SEED = 123
TRAIN_FRACTION = 0.7


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


# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

# TODO: Confirmar si este bloque realmente debe eliminar filas con NA.
# El script R original solo reescribe el CSV con otro nombre, sin limpieza.
def copy_csv_without_modification(input_csv: Path, output_csv: Path) -> None:
    """
    Replica el comportamiento del bloque R original:
    lee un CSV y lo escribe con otro nombre, sin filtrar NA.
    """
    validate_input_file(input_csv, "CSV de copia")
    ensure_parent(output_csv)

    df = pd.read_csv(input_csv)
    df.to_csv(output_csv, index=False)

    log(f"CSV copiado sin modificación a: {output_csv}")

    # Posible altenativa
    #def copy_csv_without_modification(input_csv: Path, output_csv: Path) -> None:
    # TODO: Confirmar si este bloque debe filtrar NA.
    # El original en R no lo hace.
    #df = pd.read_csv(input_csv)
    #df.to_csv(output_csv, index=False)


def split_csv_train_test(
    input_csv: Path,
    output_train_csv: Path,
    output_test_csv: Path,
    train_fraction: float = TRAIN_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> None:
    validate_input_file(input_csv, "CSV para división 70/30")
    ensure_parent(output_train_csv)
    ensure_parent(output_test_csv)

    if not (0 < train_fraction < 1):
        fail("train_fraction debe estar entre 0 y 1.")

    df = pd.read_csv(input_csv)

    if df.empty:
        fail(f"El CSV está vacío: {input_csv}")

    n_train = int(len(df) * train_fraction)

    # Equivalente conceptual a sample(...) en R con semilla fija
    train_df = df.sample(n=n_train, random_state=random_seed)
    test_df = df.drop(train_df.index)

    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)

    log(f"CSV 70% guardado en: {output_train_csv}")
    log(f"CSV 30% guardado en: {output_test_csv}")
    log(f"Filas train: {len(train_df)}")
    log(f"Filas test : {len(test_df)}")


def main() -> int:
    log("Bloque 1: reescritura simple del CSV...")
    copy_csv_without_modification(INPUT_COPY_CSV, OUTPUT_COPY_CSV)

    log("\nBloque 2: división aleatoria 70/30...")
    split_csv_train_test(
        input_csv=INPUT_SPLIT_CSV,
        output_train_csv=OUTPUT_TRAIN_CSV,
        output_test_csv=OUTPUT_TEST_CSV,
        train_fraction=TRAIN_FRACTION,
        random_seed=RANDOM_SEED,
    )

    log("\n----------------------------------------")
    log("Proceso completado.")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)