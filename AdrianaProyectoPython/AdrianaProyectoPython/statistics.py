import numpy as np
import pandas as pd

# --------------------------------------
# Varianza por columna
# --------------------------------------
def get_vector_variances(matrix: pd.DataFrame | np.ndarray) -> pd.Series:
    """
    Calcula la varianza de cada columna de una matriz o DataFrame numérico.
    
    Args:
        matrix: pd.DataFrame o np.ndarray
    
    Returns:
        pd.Series: Varianza por columna
    """
    if not isinstance(matrix, (pd.DataFrame, np.ndarray)):
        raise TypeError("El argumento debe ser un DataFrame o ndarray numérico")
    
    df = pd.DataFrame(matrix)
    return df.var()

# --------------------------------------
# Entropía de un vector categórico o discretizado
# --------------------------------------
def entropy(x: pd.Series | np.ndarray) -> float:
    """
    Calcula la entropía (en bits) de un vector categórico o discretizado.
    
    Args:
        x: pd.Series o np.ndarray
    
    Returns:
        float: Entropía
    """
    x = pd.Series(x).astype("category")
    counts = x.value_counts(normalize=True)
    probs = counts[counts > 0]
    H = -np.sum(probs * np.log2(probs))
    return H

# --------------------------------------
# Entropía por columnas de un DataFrame
# --------------------------------------
def get_vector_entropy(df: pd.DataFrame) -> pd.Series:
    """
    Calcula la entropía de cada columna de un DataFrame.
    
    Args:
        df: pd.DataFrame
    
    Returns:
        pd.Series: Entropía por columna
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    
    return df.apply(entropy)

# --------------------------------------
# Curva ROC y AUC para un DataFrame con variable numérica y etiqueta lógica
# --------------------------------------
def get_ROC_AUC(df: pd.DataFrame) -> dict:
    """
    Calcula los puntos de la curva ROC y el AUC para una variable numérica y una etiqueta booleana.
    
    Args:
        df: pd.DataFrame con 2 columnas:
            - Primera columna: variable numérica
            - Segunda columna: etiqueta lógica (True/False)
    
    Returns:
        dict: {"roc_points": pd.DataFrame con FPR y TPR, "auc": float}
    """
    if not isinstance(df, pd.DataFrame) or df.shape[1] != 2:
        raise ValueError("El argumento debe ser un DataFrame con 2 columnas")
    
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.bool_):
        raise TypeError("Primera columna debe ser numérica y segunda columna booleana")
    
    # Ordenar por x descendente
    order = np.argsort(-x)
    x = x[order]
    y = y[order]
    
    cut_points = np.unique(x)
    roc_points = []
    P = np.sum(y)
    N = len(y) - P
    
    for c in cut_points:
        pred = x >= c
        TP = np.sum(pred & y)
        FP = np.sum(pred & ~y)
        FN = np.sum(~pred & y)
        TN = np.sum(~pred & ~y)
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        roc_points.append((FPR, TPR))
    
    # Añadir (0,0) y (1,1)
    roc_points = [(0,0)] + roc_points + [(1,1)]
    roc_df = pd.DataFrame(roc_points, columns=["FPR", "TPR"]).sort_values("FPR")
    
    # Cálculo del AUC por regla del trapecio
    auc = np.trapz(roc_df["TPR"], roc_df["FPR"])
    
    return {"roc_points": roc_df, "auc": auc}
