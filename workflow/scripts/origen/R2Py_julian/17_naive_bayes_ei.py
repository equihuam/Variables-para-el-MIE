#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 17_naive_bayes_ei.py

Propósito:
    Entrenar un modelo naive Bayes para predecir la clase discretizada de
    integridad ecosistémica (ei_qnint) a partir de la tabla final de entrenamiento
    y exportar predicciones y probabilidades en formato PKL/CSV.

Origen:
    Traducción inicial a Python del script R:
    17_naive_bayes_ei.R

Resumen del flujo:
    1. Leer la tabla final de entrenamiento.
    2. Convertir variables categóricas a factor/categoría y discretizar variables continuas.
    3. Eliminar observaciones sin etiqueta ei_qnint.
    4. Construir y ajustar un modelo naive Bayes discreto.
    5. Generar predicciones de clase y probabilidades por observación.
    6. Exportar las salidas del modelo en PKL y CSV.

Insumos principales:
    - cei_final_train_v1.pkl

Salidas principales:
    - 17_naive_bayes_predictions.pkl
    - 17_naive_bayes_predictions.csv

Supuestos y notas:
    - Esta traducción usa pgmpy como aproximación a bnlearn.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - factorCols() se replica como coerción simple a categoría.
    - discretizeCols() se replica con 5 intervalos iguales por default, como en misc_functions.R.
    - La parte final del script R visible está incompleta/inconsistente; esta
      versión Python completa el flujo de predicción de manera explícita.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, la naive Bayes de bnlearn se aproxima con una red bayesiana
    discreta de estructura naive en pgmpy. Además, como el fragmento visible del
    script R deja incompleta la predicción final, esta traducción implementa una
    salida explícita de clase predicha y probabilidades por fila.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
TRAIN_TABLE = DROPBOX_DIR / "data_training_tables" / "cei_final_train_v1.pkl"
OUTPUT_DIR = DROPBOX_DIR / "data_training_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "ei_qnint"

CAT_VARS = ["NESTB_EDO", "tipo_costa", "zvh", "ei_qnint"]
CON_VARS = [
    "erosion", "windspeed", "escollera", "espigon", "muro", "rompeolas",
    "puerto", "sp_inv_pot", "d_corales", "d_pastosmarinos", "bati_char",
    "d_grassland", "d_agriculture", "d_urban", "p_manglares",
]

# En R: vars <- 4:22 sobre cei_final_train_v1
MODEL_COLS = [
    "NESTB_EDO", "CONSERV_ED", "erosion", "movdunas", "tipo_costa",
    "windspeed", "zvh", "escollera", "espigon", "muro", "rompeolas",
    "puerto", "sp_inv_pot", "d_corales", "d_pastosmarinos", "bati_char",
    "d_grassland", "d_agriculture", "d_urban", "p_manglares", "ei_qnint",
]


def factor_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def discretize_cols(
        df: pd.DataFrame,
        cols: List[str],
        breaks: int = 5,
        method: str = "interval",
) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        s = pd.to_numeric(out[col], errors="coerce")

        if s.nunique(dropna=True) <= 1:
            out[col] = pd.Series(["1"] * len(s), index=s.index, dtype="category")
            continue

        if method != "interval":
            raise ValueError("Esta traducción de 17 usa method='interval' como en R.")

        disc = pd.cut(s, bins=breaks, labels=False, include_lowest=True)
        out[col] = (disc + 1).astype("Int64").astype(str).astype("category")

    return out


def build_naive_bayes_structure(target: str, explanatory: List[str]) -> DiscreteBayesianNetwork:
    edges = [(target, var) for var in explanatory]
    return DiscreteBayesianNetwork(edges)


def fit_naive_bayes(train_df: pd.DataFrame, target: str, explanatory: List[str]) -> DiscreteBayesianNetwork:
    model = build_naive_bayes_structure(target, explanatory)
    fit_df = train_df[[target] + explanatory].copy().astype(str)
    model.fit(fit_df, estimator=MaximumLikelihoodEstimator)
    return model


def predict_naive_bayes(
        model: DiscreteBayesianNetwork,
        df: pd.DataFrame,
        target: str,
        explanatory: List[str],
) -> pd.DataFrame:
    infer = VariableElimination(model)
    x_df = df[explanatory].astype(str).copy()

    predictions = []
    prob_rows = []

    target_states = list(model.get_cpds(target).state_names[target])

    for _, row in x_df.iterrows():
        evidence = {col: row[col] for col in explanatory if pd.notna(row[col])}

        try:
            q = infer.query(variables=[target], evidence=evidence, show_progress=False)
            probs = q.values
            states = q.state_names[target]

            pred_state = states[int(np.argmax(probs))]
            prob_map = {f"prob_{state}": float(prob) for state, prob in zip(states, probs)}
        except Exception:
            pred_state = None
            prob_map = {f"prob_{state}": np.nan for state in target_states}

        predictions.append(pred_state)
        prob_rows.append(prob_map)

    pred_df = pd.DataFrame(prob_rows)
    pred_df["predicted_ei_qnint"] = predictions

    return pred_df


def main() -> None:
    if not TRAIN_TABLE.exists():
        raise FileNotFoundError(f"No existe la tabla de entrada: {TRAIN_TABLE}")

    dat = pd.read_pickle(TRAIN_TABLE)

    dat = factor_cols(dat, CAT_VARS)
    dat = discretize_cols(dat, CON_VARS, breaks=5, method="interval")

    existing_cols = [c for c in MODEL_COLS if c in dat.columns]
    if TARGET not in existing_cols:
        raise KeyError(f"No existe la columna objetivo requerida: {TARGET}")

    dat_model = dat[existing_cols].copy()

    # Drop unlabeled pixels
    dat_clean = dat_model[dat_model[TARGET].notna()].copy()

    # En el R: explanatory = names(dat_clean)[1:18]
    explanatory = [c for c in dat_clean.columns if c != TARGET]

    nb_model = fit_naive_bayes(dat_clean, TARGET, explanatory)
    pred = predict_naive_bayes(nb_model, dat_clean, TARGET, explanatory)

    out = dat_clean.reset_index(drop=True).copy()
    out = pd.concat([out, pred], axis=1)

    out_pkl = OUTPUT_DIR / "17_naive_bayes_predictions.pkl"
    out_csv = OUTPUT_DIR / "17_naive_bayes_predictions.csv"

    out.to_pickle(out_pkl)
    out.to_csv(out_csv, index=False)

    print(f"OK -> {out_pkl}")
    print(f"OK -> {out_csv}")


if __name__ == "__main__":
    main()
