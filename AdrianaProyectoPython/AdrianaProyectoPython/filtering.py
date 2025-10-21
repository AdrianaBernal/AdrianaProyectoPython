import pandas as pd
import numpy as np
from .statistics import entropy, get_ROC_AUC

# --------------------------------------
# Filtrado por varianza
# --------------------------------------
def filter_by_variance(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Filtra columnas numéricas de un DataFrame según un umbral mínimo de varianza.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    
    keep = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].var() > threshold]
    return df[keep]

# --------------------------------------
# Filtrado por entropía
# --------------------------------------
def filter_by_entropy(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Filtra columnas categóricas de un DataFrame según un umbral mínimo de entropía.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    
    keep = [col for col in df.columns 
            if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) 
            and entropy(df[col]) > threshold]
    return df[keep]

# --------------------------------------
# Filtrado por AUC
# --------------------------------------
def filter_by_auc(df: pd.DataFrame, y: pd.Series | np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Filtra columnas numéricas de un DataFrame según AUC mínimo respecto a y (binario 0/1).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    if len(y) != len(df):
        raise ValueError("Longitud de y no coincide con número de filas del DataFrame")
    
    keep = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                auc = get_ROC_AUC(pd.DataFrame({col: df[col], "y": y}))["auc"]
                if auc > threshold:
                    keep.append(col)
            except Exception:
                pass
    return df[keep]

# --------------------------------------
# Filtrado de variables altamente correlacionadas
# --------------------------------------
def filter_highly_correlated(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Elimina variables altamente correlacionadas, conservando una de cada par.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    
    cor_mat = df.corr().abs()
    remove_vars = set()
    
    cols = cor_mat.columns
    for i in range(len(cols)-1):
        for j in range(i+1, len(cols)):
            if cor_mat.iloc[i, j] > threshold:
                remove_vars.add(cols[j])
    
    keep = [col for col in df.columns if col not in remove_vars]
    return df[keep]
