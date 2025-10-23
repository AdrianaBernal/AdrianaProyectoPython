Mi Paquete : AdrianaProyectoPython
=================================================

Descripción
-----------
Este paquete ofrece herramientas completas para la normalización, estandarización,
discretización, cálculo de métricas, filtrado de variables, correlación y visualización
de datasets. Está pensado para trabajar con datasets numéricos, categóricos o mixtos,
y facilita la exploración y transformación de datos.

Instalación
-----------
1. Clona el repositorio:
```python
   git clone https://github.com/AdrianaBernal/AdrianaProyectoPython.git
   cd AdrianaProyectoPython/AdrianaProyectoPython
```

2. Instala el paquete localmente:
```python
   pip install -e .
```

Estructura del paquete
---------------------

```text
AdrianaProyectoPython/
|
├─ __init__.py
├─ normalization.py      # Normalización y estandarización de vectores y dataframes
├─ discretization.py     # Discretización por igual frecuencia, igual anchura y k-means
├─ statistics.py         # Varianza, entropía, curva ROC y AUC
├─ filtering.py          # Filtrado de variables por métricas y correlación
├─ correlation.py        # Matrices de correlación e información mutua
├─ plotting.py           # Gráficos: histogramas, boxplots, ROC, heatmaps
├─ ManageDataset.py      # Clase para gestionar datasets y registrar transformaciones
└─ tests/                # Tests unitarios con pytest

```


Ejemplos de uso
---------------
# Normalización y estandarización
```python
from AdrianaProyectoPython import normalization
import pandas as pd
df = pd.DataFrame({"a": [1,2,3], "b":[4,5,6]})
df_norm = normalization.get_normalized_df(df)
df_std = normalization.get_standardized_df(df)
```

# Discretización
```python
from AdrianaProyectoPython import discretization
x = [1, 2, 3, 4, 5]
discrete_ew = discretization.discretize_EW(x, 3)
```

# Cálculo de métricas
```python
from AdrianaProyectoPython import statistics
import numpy as np
var = statistics.get_vector_variances(pd.DataFrame({"a": np.arange(5)}))
ent = statistics.get_vector_entropy(pd.DataFrame({"b": ["x","y","x","z","y"]}))
```

# Filtrado de variables
```python
from AdrianaProyectoPython import filtering
df_filtered = filtering.filter_by_variance(df, threshold=0.5)
```

# Gestión de datasets
```python
from AdrianaProyectoPython import ManageDataset
md = ManageDataset.ManagedDataset(df, name="MiDataset")
md.add_transformation("Normalizado")
data = md.getData()
```


# Visualización
```python
from AdrianaProyectoPython import plotting
plotting.plot_correlation_matrix(df)
plotting.plotHistograms(df)
```


Tests
-----
Ejecutar todos los tests:
pytest AdrianaProyectoPython/test

Requisitos
----------
- Python >= 3.8
- pandas
- numpy
- matplotlib
