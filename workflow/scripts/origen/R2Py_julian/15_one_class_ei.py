#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 15_one_class_ei.py

Propósito:
    Aprender una red bayesiana a partir de las dunas mejor conservadas,
    calcular un score de verosimilitud por observación y derivar versiones
    discretizadas del índice de integridad ecosistémica (EI) para entrenamiento.

Origen:
    Traducción inicial a Python del script R:
    15_one_class_ei.R

Resumen del flujo:
    1. Leer la tabla integrada de entrenamiento.
    2. Convertir variables categóricas a factor/categoría y discretizar variables continuas.
    3. Ajustar una red bayesiana sobre las observaciones con CONSERV_ED == 1.
    4. Calcular un score log-likelihood por observación sobre todas las filas.
    5. Generar discretizaciones del EI por intervalos iguales y por cuantiles.
    6. Construir las tablas finales de entrenamiento y exportarlas en PKL y CSV.

Insumos principales:
    - train_dat_2c.pkl

Salidas principales:
    - train_dat_2c_loglik.pkl
    - train_dat_2c_eieqint.pkl
    - train_dat_2c_eiqnint.pkl
    - cei_final_train_v1.pkl
    - cei_final_train_v1.csv
    - cei_final_train_v1d.csv
    - cei_final_train_v1ask.csv

Supuestos y notas:
    - Esta traducción usa pgmpy como aproximación a bnlearn.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - factorCols() se replica como coerción simple a categoría.
    - discretizeCols() se replica con 5 intervalos iguales por default, como en misc_functions.R.
    - Las discretizaciones de log-likelihood se reproducen con métodos interval y quantile.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, la equivalencia con bnlearn no es exacta: la estructura y los
    scores de una red bayesiana aprendida con pgmpy pueden diferir ligeramente.
    Además, los cortes de discretización se reconstruyen desde los valores
    calculados y no desde etiquetas de texto fijas del R, para preservar la
    lógica analítica y no una impresión puntual de intervalos.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, K2
from pgmpy.models import DiscreteBayesianNetwork


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
TRAIN_TABLE = DROPBOX_DIR / "data_training_tables" / "train_dat_2c.pkl"
OUTPUT_DIR = DROPBOX_DIR / "data_training_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_VARS = ["NESTB_EDO", "tipo_costa", "zvh", "CONSERV_ED"]
CON_VARS = [
    "erosion", "windspeed", "escollera", "espigon", "muro", "rompeolas",
    "puerto", "sp_inv_pot", "d_corales", "d_pastosmarinos", "bati_char",
    "d_grassland", "d_agriculture", "d_urban", "p_manglares",
]

# Equivalente a dat[,6:23] / final_train[,c(1:2,5:23)]
MODEL_COLS = [
    "NESTB_EDO", "CONSERV_ED", "erosion", "movdunas", "tipo_costa",
    "windspeed", "zvh", "escollera", "espigon", "muro", "rompeolas",
    "puerto", "sp_inv_pot", "d_corales", "d_pastosmarinos", "bati_char",
    "d_grassland", "d_agriculture", "d_urban", "p_manglares",
]

FINAL_COLS = [
    "x", "y", "regionid.x",
    "NESTB_EDO", "CONSERV_ED", "erosion", "movdunas", "tipo_costa",
    "windspeed", "zvh", "escollera", "espigon", "muro", "rompeolas",
    "puerto", "sp_inv_pot", "d_corales", "d_pastosmarinos", "bati_char",
    "d_grassland", "d_agriculture", "d_urban", "p_manglares",
]


def factor_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def discretize_cols(
        df: pd.DataFrame,
        cols: list[str],
        breaks: int = 5,
        method: str = "interval",
) -> pd.DataFrame:
    """
    Aproximación a misc_functions.R:
      discretizeCols(..., breaks_vec=rep(5,...), method="interval")
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        s = pd.to_numeric(out[col], errors="coerce")

        if s.nunique(dropna=True) <= 1:
            out[col] = pd.Series(["1"] * len(s), index=s.index, dtype="category")
            continue

        if method == "interval":
            disc = pd.cut(s, bins=breaks, labels=False, include_lowest=True)
        elif method == "quantile":
            disc = pd.qcut(s, q=breaks, labels=False, duplicates="drop")
        else:
            raise ValueError(f"Método de discretización no soportado: {method}")

        out[col] = (disc + 1).astype("Int64").astype(str).astype("category")

    return out


def learn_bn_structure(train_df: pd.DataFrame) -> DiscreteBayesianNetwork:
    hc = HillClimbSearch(train_df)
    best_model = hc.estimate(scoring_method=K2(train_df))
    model = DiscreteBayesianNetwork(best_model.edges())
    model.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu")
    return model


def row_loglik(model: DiscreteBayesianNetwork, df: pd.DataFrame) -> np.ndarray:
    """
    Aproximación a logLik(..., by.sample = TRUE) en bnlearn.
    """
    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}
    scores = np.zeros(len(df), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        ll = 0.0
        for var in model.nodes():
            cpd = cpds[var]
            state_names = cpd.state_names

            try:
                var_state = str(row[var])
                var_index = state_names[var].index(var_state)

                evidence = cpd.variables[1:]
                if evidence:
                    evidence_states = [str(row[e]) for e in evidence]
                    evidence_idx = tuple(
                        state_names[e].index(v) for e, v in zip(evidence, evidence_states)
                    )
                    p = cpd.values[(var_index,) + evidence_idx]
                else:
                    p = cpd.values[var_index]

                p = max(float(p), 1e-12)
                ll += np.log(p)
            except Exception:
                ll += np.log(1e-12)

        scores[i] = ll

    return scores


def assign_ei_from_ordered_bins(conserv_ed: pd.Series, bins_series: pd.Series) -> pd.Series:
    """
    Replica la lógica del R usando el orden ascendente de los intervalos.
    """
    out = pd.Series(0, index=conserv_ed.index, dtype=int)
    cats = list(bins_series.cat.categories)

    if len(cats) < 5:
        return out

    conserv_num = pd.to_numeric(conserv_ed, errors="coerce")
    mapping = {
        5: cats[0],
        4: cats[1],
        3: cats[2],
        2: cats[3],
        1: cats[4],
    }

    for conserv_class, interval in mapping.items():
        mask = (conserv_num == conserv_class) & (bins_series == interval)
        out.loc[mask] = 6 - conserv_class

    return out


def main() -> None:
    dat = pd.read_pickle(TRAIN_TABLE)

    dat = factor_cols(dat, CAT_VARS)
    dat = discretize_cols(dat, CON_VARS, breaks=5, method="interval")

    # Only best conserved dunes.
    dat_b = dat.loc[pd.to_numeric(dat["CONSERV_ED"], errors="coerce") == 1, MODEL_COLS].copy()
    dat_b = dat_b.astype(str)

    bn_model = learn_bn_structure(dat_b)

    model_data = dat[MODEL_COLS].copy().astype(str)
    loglike_scores = pd.DataFrame({"loglik": row_loglik(bn_model, model_data)})

    dat["loglik"] = loglike_scores["loglik"].to_numpy()
    histlik = -1.0 * loglike_scores["loglik"]
    dat["likelihood"] = histlik.to_numpy()

    datlik = dat.copy()
    datlik.to_pickle(OUTPUT_DIR / "train_dat_2c_loglik.pkl")

    # equal interval
    eq_int = pd.cut(loglike_scores["loglik"], bins=5, include_lowest=True)
    dat_eq_int = dat.copy()
    dat_eq_int["ei_eqint"] = assign_ei_from_ordered_bins(dat_eq_int["CONSERV_ED"], eq_int)
    dat_eq_int[dat_eq_int["ei_eqint"] != 0].to_pickle(OUTPUT_DIR / "train_dat_2c_eieqint.pkl")

    # quantile interval
    qn_int = pd.qcut(datlik["loglik"], q=5, duplicates="drop")
    dat_qn_int = datlik.copy()
    dat_qn_int["ei_qnint"] = assign_ei_from_ordered_bins(dat_qn_int["CONSERV_ED"], qn_int)
    dat_qn_int_nonzero = dat_qn_int[dat_qn_int["ei_qnint"] != 0].copy()
    dat_qn_int_nonzero.to_pickle(OUTPUT_DIR / "train_dat_2c_eiqnint.pkl")

    # Final training data without discretization.
    keep_cols = [c for c in FINAL_COLS if c in dat.columns]
    final_train = dat[keep_cols].copy()
    final_train["ei_qnint"] = dat_qn_int["ei_qnint"].to_numpy()

    final_train.to_pickle(OUTPUT_DIR / "cei_final_train_v1.pkl")
    final_train.to_csv(OUTPUT_DIR / "cei_final_train_v1.csv", index=False)

    # Discretize continuous variables with interval method
    datd = discretize_cols(final_train, CON_VARS, breaks=5, method="interval")
    datd.to_csv(OUTPUT_DIR / "cei_final_train_v1d.csv", index=False)

    # with * for Netica / ask
    datd_netica = datd.copy()
    datd_netica["ei_qnint"] = datd_netica["ei_qnint"].astype(str)
    datd_netica.loc[datd_netica["ei_qnint"] == "0", "ei_qnint"] = "*"
    datd_netica.to_csv(OUTPUT_DIR / "cei_final_train_v1d.csv", index=False)

    final_train_ask = final_train.copy()
    final_train_ask["ei_qnint"] = final_train_ask["ei_qnint"].astype(str)
    final_train_ask.loc[final_train_ask["ei_qnint"] == "0", "ei_qnint"] = "*"
    final_train_ask.to_csv(OUTPUT_DIR / "cei_final_train_v1ask.csv", index=False)

    print(f"OK -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()