import pytest
import numpy as np
import pandas as pd
from AdrianaProyectoPython import (
    normalization,
    discretization,
    statistics,
    filtering,
    correlation,
    plotting,
    ManageDataset
)

# =======================================
# Datos de prueba comunes
# =======================================
np.random.seed(42)
df_numeric = pd.DataFrame({
    "a": np.random.normal(size=100),
    "b": np.random.uniform(size=100),
    "c": np.arange(100)
})

df_mixed = pd.DataFrame({
    "num": np.random.normal(size=50),
    "cat": ["a", "b", "c", "a", "b"] * 10
})

y_bin = np.random.choice([0,1], size=100)

# =======================================
# Normalización / Estandarización
# =======================================
def test_normalization_standardization_vector():
    x = np.array([2,4,6,8])
    norm = normalization.normalize(x)
    std = normalization.standardize(x)
    
    assert np.allclose(norm, np.array([0,0.3333,0.6667,1]), atol=1e-3)
    assert np.isclose(np.mean(std), 0)
    assert np.isclose(np.std(std), 1, atol=1e-3)

def test_normalization_standardization_df():
    df_norm = normalization.get_normalized_df(df_numeric)
    df_std = normalization.get_standardized_df(df_numeric)
    
    assert df_norm.min().min() >= 0
    assert df_norm.max().max() <= 1
    for col in df_std.columns:
        assert np.isclose(df_std[col].mean(), 0, atol=1e-3)
        assert np.isclose(df_std[col].std(ddof=1), 1, atol=1e-1)

# =======================================
# Discretización
# =======================================
def test_discretization_EW_EF_KMeans():
    x = np.arange(10)
    ew = discretization.discretize_EW(x, 3)['x_discretized']
    ef = discretization.discretize_EF(x, 3)['x_discretized']
    km = discretization.discretize_KMeans(x, 3)['x_discretized']
    
    assert len(np.unique(ew.astype(str))) == 3
    assert len(np.unique(ef.astype(str))) == 3
    assert len(np.unique(km.astype(str))) == 3

def test_discretized_df():
    df_disc = discretization.get_discretized_df(df_numeric, 3)
    for col in ["a_EW", "b_EF", "c_KM"]:
        assert col in df_disc.columns

# =======================================
# Estadísticas
# =======================================
def test_variance_entropy_auc():
    var = statistics.get_vector_variances(df_numeric)
    ent = statistics.get_vector_entropy(df_mixed[['cat']])
    auc = statistics.get_ROC_AUC(pd.DataFrame({"x": df_numeric["a"], "y": y_bin.astype(bool)}))['auc']
    
    assert all(var > 0)
    assert all(ent > 0)
    assert 0 <= auc <= 1

# =======================================
# Filtrado
# =======================================
def test_filtering():
    df_test = pd.DataFrame({
        "low_var": [1]*10,
        "high_var": np.arange(10)
    })
    df_var = filtering.filter_by_variance(df_test, 0.5)
    assert "low_var" not in df_var.columns
    assert "high_var" in df_var.columns

# =======================================
# Correlación
# =======================================
def test_correlation_matrix_pairwise():
    cor_mat = correlation.get_correlation_matrix(df_numeric)
    cor_df = correlation.get_pairwise_correlations(df_numeric)
    
    assert cor_mat.shape[0] == df_numeric.shape[1]
    assert not cor_df.empty

# =======================================
# ManageDataset
# =======================================
def test_managed_dataset():
    md = ManageDataset.ManagedDataset(df_numeric, name="Test", description="Prueba")
    assert md.name == "Test"
    assert md.description == "Prueba"
    md.add_transformation("Normalized")
    assert "Normalized" in md.transformations
    df_data = md.get_data()
    assert df_data.equals(df_numeric)

# =======================================
# Plotting (solo prueba de ejecución, no visual)
# =======================================
def test_plot_functions_execution():
    # solo comprobamos que no lanza errores
    plotting.plot_correlation_matrix(df_numeric)
    plotting.plotROC(pd.DataFrame({"x": df_numeric["a"], "y": y_bin.astype(bool)}))
    plotting.plot_normalized_entropy(df_mixed)
    plotting.plotCorrelationHeatmap(df_numeric)
    plotting.plotMutualInfoHeatmap(df_mixed)
    plotting.plotHistograms(df_numeric)
    plotting.plotBoxplots(df_numeric)
