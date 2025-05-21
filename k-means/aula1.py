with open('wine.csv', 'r') as f:
    lines = f.readlines()

    import pandas as pd
from io import StringIO

# Processando o conteúdo
clean_content = []
for line in lines:
    # Remover aspas e quebras de linha
    clean_line = line.replace('"', '').strip()
    # Substituir múltiplas vírgulas por um único separador
    clean_line = ','.join(clean_line.split(','))
    clean_content.append(clean_line)

# Juntar tudo em uma string formatada corretamente
clean_data = '\n'.join(clean_content)

# Agora ler com pandas
wine_df = pd.read_csv(StringIO(clean_data))

print("_______EXIBINDO AS PRIMEIRAS 5 LINHAS CORRETAS_______")
print(wine_df.head())

print("\n_______ESTRUTURA DO DATAFRAME_______")
print(wine_df.info())

# Verificando valores nulos
print("___VERIFICANDO VALORES NULOS___")
print(wine_df.isnull().sum())

# Preenchendo com a média (alternativa: remover linhas)
wine_df['alcohol'] = wine_df['alcohol'].fillna(wine_df['alcohol'].mean())

# Confirmando
print("\nValores nulos após tratamento:")
print(wine_df.isnull().sum())

print("\nEstatísticas descritivas:")
print(wine_df.describe())

# Verificando o balanceamento das classes
print("\nDistribuição das classes:")
print(wine_df['class_name'].value_counts())

import matplotlib.pyplot as plt

# Histograma para álcool
plt.figure(figsize=(10, 6))
wine_df['alcohol'].plot(kind='hist', bins=20)
plt.title('Distribuição do Teor Alcoólico')
plt.xlabel('Álcool')
plt.ylabel('Frequência')
plt.show()

# NORMALIZAÇÃO DOS DADOS
from sklearn.preprocessing import StandardScaler

# Separando features e target
X = wine_df.drop(['class_label', 'class_name'], axis=1)
y = wine_df['class_label']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# APLICANDO MÉTODO COTOVELO
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

# Treinando o K-Means com k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

wine_df['cluster'] = clusters

import pandas as pd

# Tabela cruzada entre classes reais e clusters
cross_tab = pd.crosstab(
    wine_df['class_name'],
    wine_df['cluster'],
    rownames=['Classe Real'],
    colnames=['Cluster']
)
print("_____________________________________________________________________")
print("Exibindo tabela cruzada entre as classes reais e clusters logo abaixo")
print(cross_tab)

# Métricas de validação
from sklearn.metrics import adjusted_rand_score
# Comparando clusters com classes reais
ari = adjusted_rand_score(wine_df['class_label'], clusters)
print(f"Índice Rand Ajustado: {ari:.2f}")

print("_____________________________________________________________________")
# Visualização dos Clusters(PCA). 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduzindo para 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotando e exibindo na tela
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Componente PCA 1')
plt.ylabel('Componente PCA 2')
plt.title('Clusters de Vinhos (K-Means com k=3)')
plt.show()

print("___________________________CENTRÓIDES________________________________")
# Criando DataFrame com os centróides
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns
)

# Adicionando rótulos (opcional)
centroids['cluster'] = ['Cluster 0', 'Cluster 1', 'Cluster 2']

print(centroids)

# Examinar as amostras "mal classificadas"
outliers = wine_df[
    (wine_df['class_name'] == 'Grignolino') & 
    (wine_df['cluster'] != 0)
]
print("_________Exibindo as amostras mal classificadas_________")
print(outliers.describe())