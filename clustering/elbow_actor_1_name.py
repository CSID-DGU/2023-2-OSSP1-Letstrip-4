import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

# 엑셀 파일 경로 지정
excel_file_path = 'C:/Users/jamie/Downloads/final.xlsx'

# 엑셀 파일 읽기
movies = pd.read_excel(excel_file_path)

# 출연진1 정보 전처리
mlb = MultiLabelBinarizer()
actor1_matrix = mlb.fit_transform(movies['actor_1_name'].apply(lambda x: str(x).split('|')))
actor1_df = pd.DataFrame(actor1_matrix, columns=mlb.classes_)

# KMeans 모델 훈련 및 Elbow Method 적용
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(actor1_df)
    sse.append(kmeans.inertia_)

# Elbow Method 그래프 그리기
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.show()

# Optimal k를 선택하여 KMeans 모델 재훈련
optimal_k = 4  # Elbow Method 그래프를 통해 최적의 k를 선택
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
actor1_df['cluster'] = kmeans_optimal.fit_predict(actor1_df)

# 클러스터 결과 확인
cluster_counts = actor1_df['cluster'].value_counts()
print(cluster_counts)

# 차원 축소 및 시각화 (예시로 2차원으로 축소)
pca = PCA(n_components=2)
actor1_pca = pca.fit_transform(actor1_df.drop('cluster', axis=1))
actor1_df[['pca1', 'pca2']] = actor1_pca

# 클러스터링 결과 시각화
plt.scatter(actor1_df['pca1'], actor1_df['pca2'], c=actor1_df['cluster'], cmap='viridis')
plt.title('Clustering Results')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
