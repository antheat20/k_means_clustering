import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

file_path = 'D:/clustering/heart3.csv'
data = pd.read_csv(file_path)

# Нормализация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Метод локтя
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
diff = np.diff(wcss)
diff2 = np.diff(diff)
# Точка наибольшего изменения
optimal_clusters = np.argmax(diff2) + 2 #+2 добавляется, чтобы компенсировать дважды уменьшенную размерность массива wcss при вычислении второй разности

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Метод локтя')
plt.xlabel('Число кластеров')
plt.ylabel('Сумма квадратов расстояний')
plt.axvline(x=optimal_clusters, linestyle='--', color='r')
plt.text(optimal_clusters, wcss[optimal_clusters-1], f'  Оптимальное число кластеров: {optimal_clusters}', color='r', verticalalignment='bottom')
plt.show()

# K-means с оптимальным количеством кластеров
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(scaled_data)

# Визуализация кластеризации
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], c=cluster_labels, cmap='viridis', marker='o', alpha=0.5)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=100, c='red', label='Центроид')
ax.set_title('Кластеризация алгоритмом k-means')
ax.set_xlabel('Возраст')
ax.set_ylabel('Артериальное давление')
ax.set_zlabel('Холестерин')
ax.legend()
plt.show()

# Оценка качества кластеризации
silhouette_avg = silhouette_score(scaled_data, cluster_labels)
davies_bouldin = davies_bouldin_score(scaled_data, cluster_labels)
calinski_harabasz = calinski_harabasz_score(scaled_data, cluster_labels)

print(f"Коэффициент силуэта: {silhouette_avg}")
print(f"Индекс Дэвиса-Болдина: {davies_bouldin}")
print(f"Индекс Калински-Харабаза: {calinski_harabasz}")