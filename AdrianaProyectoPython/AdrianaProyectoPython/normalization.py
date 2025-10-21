import pandas as pd
import numpy as np

# --------------------------------------
# Normalización de un vector numérico
# --------------------------------------
def normalize(x: pd.Series | list | np.ndarray) -> np.ndarray:
    """
    Normaliza un vector numérico al rango [0, 1].
    
    Args:
        x: Vector numérico (list, np.ndarray o pd.Series)
    
    Returns:
        np.ndarray: Vector normalizado
    """
    x = np.array(x, dtype=float)
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("El argumento debe ser numérico")
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# --------------------------------------
# Estandarización de un vector numérico
# --------------------------------------
def standardize(x: pd.Series | list | np.ndarray) -> np.ndarray:
    """
    Estandariza un vector numérico a media 0 y desviación típica 1.
    
    Args:
        x: Vector numérico (list, np.ndarray o pd.Series)
    
    Returns:
        np.ndarray: Vector estandarizado
    """
    x = np.array(x, dtype=float)
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("El argumento debe ser numérico")
    return (x - np.mean(x)) / np.std(x)

# --------------------------------------
# Normalización de un DataFrame
# --------------------------------------
def get_normalized_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las columnas numéricas de un DataFrame al rango [0, 1].
    
    Args:
        df: pd.DataFrame
    
    Returns:
        pd.DataFrame: DataFrame con columnas numéricas normalizadas
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    df_norm = df.copy()
    for col in df_norm.select_dtypes(include=np.number).columns:
        df_norm[col] = normalize(df_norm[col])
    return df_norm

# --------------------------------------
# Estandarización de un DataFrame
# --------------------------------------
def get_standardized_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza las columnas numéricas de un DataFrame a media 0 y desviación típica 1.
    
    Args:
        df: pd.DataFrame
    
    Returns:
        pd.DataFrame: DataFrame con columnas numéricas estandarizadas
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    df_std = df.copy()
    for col in df_std.select_dtypes(include=np.number).columns:
        df_std[col] = standardize(df_std[col])
    return df_std
