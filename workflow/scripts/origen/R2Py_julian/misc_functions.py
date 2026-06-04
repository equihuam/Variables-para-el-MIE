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
      factorCols()
      discretizeCols(..., breaks_vec=rep(5,...), method="interval")

    method:
      - "interval": intervalos iguales
      - "quantile": cuantiles
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