import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .statistics import entropy, get_ROC_AUC
from .correlation import get_correlation_matrix

# --------------------------------------
# Heatmap de correlación
# --------------------------------------
def plot_correlation_matrix(df: pd.DataFrame):
    """
    Genera un mapa de calor (heatmap) con la matriz de correlación de un DataFrame.
    """
    cor_mat = get_correlation_matrix(df)
    plt.figure(figsize=(8,6))
    sns.heatmap(cor_mat, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Matriz de correlación")
    plt.show()

# --------------------------------------
# Curva ROC con AUC
# --------------------------------------
def plotROC(df: pd.DataFrame):
    """
    Dibuja la curva ROC (Receiver Operating Characteristic) y muestra el valor del AUC.

    Usa los resultados de la función get_ROC_AUC para graficar la relación entre
    la tasa de verdaderos positivos (TPR) y falsos positivos (FPR).
    """
    roc_result = get_ROC_AUC(df)
    roc_points = roc_result['roc_points']
    auc = roc_result['auc']
    
    plt.figure(figsize=(6,6))
    plt.plot(roc_points['FPR'], roc_points['TPR'], color='blue', lw=2)
    plt.plot([0,1], [0,1], color='red', lw=2, linestyle='--')
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC (AUC = {auc:.3f})")
    plt.grid(True)
    plt.show()

# --------------------------------------
# Entropía normalizada por variable
# --------------------------------------
def plot_normalized_entropy(df: pd.DataFrame):
    """
    Calcula y grafica la entropía normalizada de cada variable categórica del DataFrame.
    """
    entropies = {}
    for col in df.columns:
        col_data = df[col].astype('category')
        H = entropy(col_data)
        maxH = np.log2(len(col_data.cat.categories))
        entropies[col] = 0 if maxH == 0 else H / maxH
    
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(entropies.keys()), y=list(entropies.values()), palette="pastel")
    plt.ylim(0, 1)
    plt.ylabel("Entropía Normalizada")
    plt.xticks(rotation=45)
    plt.title("Entropía normalizada por variable")
    plt.show()

# --------------------------------------
# Heatmap de correlaciones (seaborn)
# --------------------------------------
def plotCorrelationHeatmap(df: pd.DataFrame):
    """
    Dibuja un mapa de calor de la matriz de correlación.
    """
    cor_mat = get_correlation_matrix(df)
    plt.figure(figsize=(8,6))
    sns.heatmap(cor_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Heatmap de correlaciones")
    plt.show()

# --------------------------------------
# Heatmap de información mutua
# --------------------------------------
def plotMutualInfoHeatmap(df: pd.DataFrame):
    """
    Calcula y visualiza un mapa de calor de la información mutua entre las variables.
    """
    df = df.apply(lambda x: x.astype('category'))
    n = df.shape[1]
    mi_mat = pd.DataFrame(np.zeros((n, n)), columns=df.columns, index=df.columns)
    
    def joint_entropy(x, y):
        joint_table = pd.crosstab(x, y)
        joint_probs = joint_table / joint_table.values.sum()
        joint_probs = joint_probs[joint_probs > 0]
        return -np.sum(joint_probs * np.log2(joint_probs))
    
    for i in df.columns:
        for j in df.columns:
            Hx = entropy(df[i])
            Hy = entropy(df[j])
            Hxy = joint_entropy(df[i], df[j])
            mi_mat.loc[i, j] = float(Hx.sum() + Hy.sum() - Hxy.sum())

    
    plt.figure(figsize=(8,6))
    sns.heatmap(mi_mat, annot=True, fmt=".2f", cmap="Greens")
    plt.title("Heatmap de Información Mutua")
    plt.show()

# --------------------------------------
# Histogramas para variables numéricas
# --------------------------------------
def plotHistograms(df: pd.DataFrame):
    """
    Genera histogramas individuales para todas las variables numéricas de un DataFrame.
    """

    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], bins=30, kde=False, color="pink")
        plt.title(f"Histograma de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.show()

# --------------------------------------
# Boxplots para variables numéricas
# --------------------------------------
def plotBoxplots(df: pd.DataFrame):
    """
    Genera boxplots individuales para todas las variables numéricas de un DataFrame.
    """
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        plt.figure()
        sns.boxplot(y=df[col], color="orange")
        plt.title(f"Boxplot de {col}")
        plt.ylabel(col)
        plt.show()
