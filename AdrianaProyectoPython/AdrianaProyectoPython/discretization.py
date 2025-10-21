import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# --------------------------------------
# Discretización por intervalos de igual anchura
# --------------------------------------
def discretize_EW(x: pd.Series | np.ndarray, num_bins: int):
    x = np.array(x, dtype=float)
    min_val = np.nanmin(x)
    max_val = np.nanmax(x)
    
    breaks = np.linspace(min_val, max_val, num_bins + 1)
    x_discretized = pd.cut(x, bins=breaks, labels=[f"Bin{i}" for i in range(1, num_bins+1)], include_lowest=True)
    
    return {"x_discretized": x_discretized, "cut_points": breaks}


# --------------------------------------
# Discretización por intervalos de igual frecuencia
# --------------------------------------
def discretize_EF(x: pd.Series | np.ndarray, num_bins: int):
    x = np.array(x, dtype=float)
    probs = np.linspace(0, 1, num_bins + 1)
    cut_points = np.unique(np.quantile(x, probs))
    x_discretized = pd.cut(x, bins=cut_points, include_lowest=True)
    return {"x_discretized": x_discretized, "cut_points": cut_points}

# --------------------------------------
# Discretización usando puntos de corte dados
# --------------------------------------
def discretize(x: pd.Series | np.ndarray, cut_points: np.ndarray):
    breaks = np.concatenate(([-np.inf], cut_points, [np.inf]))
    return pd.cut(x, bins=breaks, include_lowest=True)

# --------------------------------------
# Discretización basada en K-Means
# --------------------------------------
def discretize_KMeans(x: pd.Series | np.ndarray, num_bins: int):
    x = np.array(x, dtype=float).reshape(-1, 1)
    unique_vals = len(np.unique(x))
    if unique_vals < num_bins:
        num_bins = unique_vals

    kmeans = KMeans(n_clusters=num_bins, n_init=10, random_state=42)
    kmeans.fit(x)
    cut_points = np.sort(kmeans.cluster_centers_.flatten())

    cluster_assign = np.argmin(np.abs(x - cut_points[:, None].T), axis=1)
    labels = [f"Cluster{i+1}" for i in cluster_assign]
    x_discretized = pd.Categorical(labels, categories=[f"Cluster{i+1}" for i in range(num_bins)])
    
    return {"x_discretized": x_discretized, "cut_points": cut_points}

# --------------------------------------
# Discretización de un DataFrame completo
# --------------------------------------
def get_discretized_df(df: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    df_discrete = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                df_discrete[f"{col}_EW"] = discretize_EW(df[col], num_bins)["x_discretized"]
            except Exception as e:
                print(f"Warning: No se pudo discretizar EW la columna {col}: {e}")
                df_discrete[f"{col}_EW"] = pd.NA

            try:
                df_discrete[f"{col}_EF"] = discretize_EF(df[col], num_bins)["x_discretized"]
            except Exception as e:
                print(f"Warning: No se pudo discretizar EF la columna {col}: {e}")
                df_discrete[f"{col}_EF"] = pd.NA

            try:
                df_discrete[f"{col}_KM"] = discretize_KMeans(df[col], num_bins)["x_discretized"]
            except Exception as e:
                print(f"Warning: No se pudo discretizar KM la columna {col}: {e}")
                df_discrete[f"{col}_KM"] = pd.NA

    return df_discrete
