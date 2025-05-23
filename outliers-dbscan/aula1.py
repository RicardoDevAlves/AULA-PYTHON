import pandas as pd
df = pd.DataFrame(pd.read_pickle('x_scaled.pickle'))
df_original = df.copy()

print("Exibindo as primeiras 5 linhas do arquivo")
print(df.head())

from sklearn.decomposition import PCA

print("Aplicando PCA...")
# Escolha do número de componentes principais
num_components = 2
# Aplicar PCA
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(df)
# Criar um DataFrame com os componentes principais
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Verificar as primeiras linhas do DataFrame PCA
print("Exibindo DataFrame PCA em 2 Dimensões...")
print(df_pca.head())


# APLICANDO DBSCAN
from sklearn.cluster import DBSCAN
# Definir os parâmetros do DBSCAN
eps = 0.18  # valor de eps a ser ajustado
min_samples = 5  # valor de min_samples a ser ajustado
# Criar o objeto DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# Ajustar e predizer os clusters
clusters = dbscan.fit_predict(df_pca)
# Adicionar os clusters ao DataFrame PCA
df_pca['cluster'] = clusters

# Verificar como ficou o DataFrame PCA com os clusters
print("Exibindo DataFrame PCA com clusters")
print(df_pca.head())

import matplotlib.pyplot as plt
import seaborn as sns

# # Plotar o scatter plot dos clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='Set1', data=df_pca)
# plt.title('Clusters de Outliers Identificados pelo DBSCAN')
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.legend()
# plt.show()

# Teste com eps menor
from sklearn.cluster import DBSCAN

eps_test = 0.018 # Tente valores menores
min_samples_test = 5

dbscan_test = DBSCAN(eps=eps_test, min_samples=min_samples_test)
clusters_test = dbscan_test.fit_predict(df_pca)

df_pca['cluster'] = clusters_test

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='Set1', data=df_pca)
plt.title(f'Clusters com eps={eps_test} e min_samples={min_samples_test}')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

# Contagem de rótulos
print(f"Contagem de rótulos com valor de EPS: {eps_test}")
print(df_pca['cluster'].value_counts())

