import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style(style = 'ticks')
import warnings
import scipy.stats as stats
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score


csv_path = os.path.join('..', 'data', 'data_set_final_processed.csv')

df['DATA_VENDA'] = pd.to_datetime(df['DATA_VENDA'])
df["ABERTURA_LOJA"] = pd.to_datetime(df["ABERTURA_LOJA"],format="%H:%M")
df["FECHO_LOJA"] = pd.to_datetime(df["FECHO_LOJA"],format="%H:%M")
df["TEMPO_ABERTURA"] = df["FECHO_LOJA"]-df["ABERTURA_LOJA"]
df["HORAS_ABERTURA"] = df["TEMPO_ABERTURA"].dt.total_seconds() / 3600

##Atibuir produtividade a produtivida_hora = 0
df.loc[(df['PRODUTIVIDADE_HORA'] == 0) & (df['LOJA'] == 'S - Vila Franca de Xira'), 'PRODUTIVIDADE_HORA'] = 149
df.loc[(df['PRODUTIVIDADE_HORA'] == 0) & (df['LOJA'] == 'S - Cacilhas'), 'PRODUTIVIDADE_HORA'] = 147
df.loc[(df['PRODUTIVIDADE_HORA'] == 0) & (df['LOJA'] == 'S - Ajuda'), 'PRODUTIVIDADE_HORA'] = 158

# Filtrar o dataset para ter o mesmo nr de meses
start_date = '2019-10-30'
end_date = '2023-10-30'
filtered_data = filled_data[(filled_data['DATA_VENDA'] >= start_date) & (filled_data['DATA_VENDA'] <= end_date)]

# Acrescentar media diaria de venda por loja
filtered_data['VENDA_MEDIA_TOTAL'] = filtered_data.groupby(['LOJA_ID'])['VALOR_VENDA'].transform('mean')

#Acrescentar preco medio
filtered_data['PRECO_MEDIO'] = filtered_data['VALOR_VENDA']/filtered_data['ITEMS']

#filtered_data['PRECO_MEDIO'].isna().sum()
filtered_data['PRECO_MEDIO'].fillna(0, inplace=True)

#Acrescentar Mes
filtered_data['MES'] = filtered_data['DATA_VENDA'].dt.mont

# Agregar por loja, mes, regiao e calcular medias
monthly_data_by_store = filtered_data.groupby(['LOJA_ID', 'REGIAO', 'MES', 'PRODUTIVIDADE_HORA', 'VENDA_MEDIA_TOTAL']).agg(VENDA_MEDIA_MENSAL = ('VALOR_VENDA', 'mean'),
     PRECO_MEDIO_MES = ('PRECO_MEDIO', 'mean'), VENDA_TOTAL = ('VALOR_VENDA', 'sum')
).reset_index()

#Features para o clustering  
features = monthly_data_by_store[['VENDA_MEDIA_MENSAL','VENDA_MEDIA_TOTAL' , 'PRODUTIVIDADE_HORA', 'PRECO_MEDIO_MES']]

# Normalizacao das  features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Ver distribuicao das features 
features[[ 'VENDA_MEDIA_MENSAL'   ,'VENDA_MEDIA_TOTAL'   ,   'PRODUTIVIDADE_HORA', 'PRECO_MEDIO_MES']].hist(bins=30, figsize=(10, 7))
plt.show()

# Determinar o numero otimo de clusters com Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# K-Means clustering com 3 clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
monthly_data_by_store['cluster'] = kmeans.fit_predict(features_scaled)

#PCA para visualaizacao
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
monthly_data_by_store['PCA1'] = pca_result[:, 0]
monthly_data_by_store['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(12, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=monthly_data_by_store, palette='viridis')
plt.title('Clusters of Stores (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

#2 run
# K-Means clustering com 3 clusters e 3 execucoes
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters,n_init=3, random_state=42)
monthly_data_by_store['cluster_3'] = kmeans.fit_predict(features_scaled)


#PCA para visualaizacao
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
monthly_data_by_store['PCA1'] = pca_result[:, 0]
monthly_data_by_store['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(12, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster_3', data=monthly_data_by_store, palette='viridis')
plt.title('Clusters of Stores (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

import scipy.cluster.hierarchy as sch

#Dendograma
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(features_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Stores')
plt.ylabel('Euclidean distances')

threshold = 25
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()

# Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

hierarchical_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
monthly_data_by_store['hierarchical_cluster'] = hierarchical_cluster.fit_predict(features_scaled)


# PCA para visualização
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
monthly_data_by_store['PCA1'] = pca_result[:, 0]
monthly_data_by_store['PCA2'] = pca_result[:, 1]


plt.figure(figsize=(12, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='hierarchical_cluster', data=monthly_data_by_store, palette='viridis')
plt.title('Hierarchical Clusters (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

# Fit a RandomForest to determine feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(features_scaled, monthly_data_by_store['hierarchical_cluster'])

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance

# RandomForest para  determinar feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(features_scaled, monthly_data_by_store['cluster'])


feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance


#Confirmacao da robustez da estrutura atraves do silhouette score 

sil_score_hierarchical = silhouette_score(features_scaled, monthly_data_by_store['hierarchical_cluster'])
print(f'Silhouette Score for Hierarchical Clustering: {sil_score_hierarchical}')

sil_score_kmeans = silhouette_score(features_scaled, monthly_data_by_store['cluster'])
print(f'Silhouette Score for K-means Clustering: {sil_score_kmeans}')

monthly_data_by_store

plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='PRODUTIVIDADE_HORA', data=monthly_data_by_store)
plt.title('Distribuicao de PRODUTIVIDADE_HORA por Clusters')
plt.xlabel('Cluster')
plt.ylabel('Produtividade por Hora')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='VENDA_MEDIA_TOTAL', data=monthly_data_by_store)
plt.title('Distribuicao de VENDA_MEDIA_TOTAL por Clusters')
plt.xlabel('Cluster')
plt.ylabel('Venda Média Total')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='VENDA_MEDIA_MENSAL', data=monthly_data_by_store)
plt.title('Distribuicao de VENDA_MEDIA_MENSAL por Clusters')
plt.xlabel('Cluster')
plt.ylabel('Venda Média Mensal')
plt.show()

#Preco medio
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='PRECO_MEDIO_MES', data=monthly_data_by_store)

plt.title('Distribuicao de PRECO_MEDIO_MES por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Preco Medio Mensal')

plt.tight_layout() 
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='REGIAO', y='PRODUTIVIDADE_HORA', hue='cluster', data=monthly_data_by_store)
plt.title('Distribuicao de PRODUTIVIDADE_HORA por Regiao e Clusters')
plt.xlabel('Regiao')
plt.ylabel('Produtividade por Hora')
plt.show()

#Visualizar nr de lojas por cluster por regiao

plt.figure(figsize=(14, 8))
sns.countplot(data=monthly_data_by_store, x='REGIAO', hue='cluster')


plt.title('Distribuicao de REGIAO por Cluster')
plt.xlabel('Regiao')
plt.ylabel('Numero de Lojas')
plt.legend(title='Cluster')
plt.xticks(rotation=45)  

# Mostrar o grafico
plt.show()

#Vendas medias mensais por Cluster

clustered_monthly_sales = monthly_data_by_store.groupby(['MES', 'cluster']).agg(
    total_sales=('VENDA_MEDIA_MENSAL', 'mean')
).reset_index()


plt.figure(figsize=(12, 6))
sns.lineplot(data=clustered_monthly_sales, x='MES', y='total_sales', hue='cluster', marker='o')


plt.title('Vendas mensais por Cluster')
plt.xlabel('Mes')
plt.ylabel('Vendas')
plt.legend(title='Cluster')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()

# Contar o numero de lojas por cluster
contagem_clusters = monthly_data_by_store['cluster'].value_counts().reset_index()
contagem_clusters.columns = ['Cluster', 'Numero de Lojas']

# Exibir a tabela de frequencia
contagem_clusters

# Cluster 0
df_cluster_0 = monthly_data_by_store[monthly_data_by_store['cluster'] == 0]

tabela_frequencia = df_cluster_0['REGIAO'].value_counts().reset_index()
tabela_frequencia.columns = ['Região', 'Número de Lojas']

total_lojas = tabela_frequencia['Número de Lojas'].sum()

tabela_frequencia['Porcentagem (%)'] = (tabela_frequencia['Número de Lojas'] / total_lojas) * 100

tabela_frequencia = tabela_frequencia.sort_values(by='Número de Lojas', ascending=False)

tabela_frequencia

#monthly_data_by_store.to_excel('monthly_data_by_store.xlsx', index=False)

#Verificar Cluster 1
df_cluster_1 = monthly_data_by_store[monthly_data_by_store['cluster'] == 1]

df_mensal = df_cluster_1.groupby('MES')['VENDA_MEDIA_MENSAL'].mean().reset_index()


plt.figure(figsize=(12, 6))
sns.lineplot(x='MES', y='VENDA_MEDIA_MENSAL', data=df_mensal, marker='o', color='b')


plt.title('VENDA_MEDIA_MENSAL no Cluster 1 Distribuido por Mes')
plt.xlabel('Mes')
plt.ylabel('Venda Media Mensal')
plt.xticks(rotation=45)  # Rotacionar os rotulos do eixo x para melhor visualizacao

plt.tight_layout()  
plt.show()


