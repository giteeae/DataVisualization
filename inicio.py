#importamos la libreria pandas
import pandas as pd

# Importamos la lirberia seaborn
import seaborn as sns

Importamos la libreria matplotlib
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# importamos los datasets de ejemplo de la libreria sklearn
# podemos ver todos los datasets de ejemplo en https://scikit-learn.org/stable/datasets/toy_dataset.html
from sklearn.datasets import load_wine

# para ver mejor los datos importamos print de la libreria rich
from rich import print

#cargamos el dataset en una variable
ds_vinos = load_wine()

# vemos la descripcion del dataset
print(ds_vinos.DESCR)

# Usamos pandas para convertir este dataset en un dataframe
df_vinos = pd.DataFrame(ds_vinos.data, columns=ds_vinos.feature_names)
df_vinos['target'] = pd.Series(ds_vinos.target)

# vemos su cabecera
df_vinos.head()

# obtenemos las clases de salida y cuantas muestras tenemos de cada
df_vinos.target.value_counts()

# vemos una lista de muestras
df_vinos.info()

# pasamos a verla gráficamente
df_vinos.plot(kind="scatter", x="total_phenols", y="flavanoids")
plt.show()

sns.boxplot(x="target", y="total_phenols", data=df_vinos)
plt.show()

df_vinos.hist(bins=100,figsize=(10,10))
plt.show()

from pandas.plotting import scatter_matrix
df_vinos_plot=df_vinos.drop('target',axis=1)
scatter_matrix(df_vinos_plot,figsize=(10,10))
plt.show()

import seaborn as sns
sns.set(color_codes=True)
sns.pairplot(df_vinos, hue="target")
plt.show()

sns.pairplot(df_vinos.iloc[:,[5,6,11,13]],hue='target',diag_kind="kde")
plt.show()

#muestra el tamaño de dataframe
df_vinos.shape
