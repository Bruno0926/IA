import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados do notebook
data = pd.read_csv('/mnt/data/iris.csv')

# Visualizar os primeiros dados
data.head()

# Identificação de outliers usando o método IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
data_cleaned = data[~outliers]

# Normalização dos dados
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_cleaned.drop(columns=['species']))

# Convertendo de volta para DataFrame para facilitar a visualização
data_normalized_df = pd.DataFrame(data_normalized, columns=data_cleaned.columns[:-1])

# Encontrar o número ideal de clusters usando Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_normalized_df)
    wcss.append(kmeans.inertia_)

# Plotar o Elbow
plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Calcular o Silhouette Score para diferentes números de clusters
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_normalized_df)
    score = silhouette_score(data_normalized_df, kmeans.labels_)
    silhouette_scores.append(score)

# Plotar o Silhouette Scores
plt.figure(figsize=(10, 8))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores para Diferentes Números de Clusters')
plt.show()

# Aplicar K-means com o número ideal de clusters (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_normalized_df)

# Adicionar os clusters ao DataFrame original
data_cleaned['Cluster'] = clusters

# Calcular o Davies-Bouldin Score para os clusters obtidos
db_score = davies_bouldin_score(data_normalized_df, clusters)
print(f'Davies-Bouldin Score: {db_score}')

# Comparar os clusters com as classes reais
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data_cleaned, x='sepal_length', y='sepal_width', hue='species', style='Cluster', palette='Set1')
plt.title('Clusters versus Classes Reais')
plt.show()

# Relatório de Agrupamento Usando K-means

# 1. Pré-processamento dos Dados
#- **Identificação de Outliers**: Usamos o método IQR para identificar e remover outliers.
#- **Normalização**: Normalizamos os dados para garantir a mesma escala para todas as características.

# 2. Encontrar e Avaliar Agrupamentos
#- **Método Elbow**: Utilizamos o método Elbow para determinar o número ideal de clusters, observando o "cotovelo" no gráfico de WCSS.
#- **Silhouette Score**: Calculamos os Silhouette Scores para diferentes números de clusters e identificamos o valor ideal de k.

# 3. Métricas de Avaliação
#- **Silhouette Score**: Avalia a coesão e separação dos clusters.
#- **Davies-Bouldin Score**: Implementamos esta métrica adicional para avaliar a qualidade dos agrupamentos.

# 4. Resultados e Discussão
#- **Clusters versus Classes Reais**: Visualizamos as instâncias incorretamente agrupadas e discutimos a qualidade dos agrupamentos.

# Links para os Códigos
#- [Código do Pré-processamento e Agrupamento](#)

#Este relatório resume todas as etapas do pré-processamento e os resultados da análise de agrupamento usando K-means.
