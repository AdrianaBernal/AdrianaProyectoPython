import pandas as pd

# --------------------------------------
# Calcular la matriz de correlación
# --------------------------------------
def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de un DataFrame con variables numéricas.
    
    Args:
        df: pd.DataFrame
    
    Returns:
        pd.DataFrame: matriz de correlación
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento debe ser un DataFrame")
    
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 variables numéricas")
    
    cor_mat = num_df.corr()
    return cor_mat

# --------------------------------------
# Obtener todas las correlaciones por pares
# --------------------------------------
def get_pairwise_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve todas las correlaciones por pares en formato tabla.
    
    Args:
        df: pd.DataFrame con variables numéricas
    
    Returns:
        pd.DataFrame con columnas: Var1, Var2, Correlation
    """
    cor_mat = get_correlation_matrix(df)
    
    data = []
    vars_ = cor_mat.columns
    for i in range(len(vars_)-1):
        for j in range(i+1, len(vars_)):
            data.append({"Var1": vars_[i], "Var2": vars_[j], "Correlation": cor_mat.iloc[i, j]})
    
    cor_df = pd.DataFrame(data)
    return cor_df
