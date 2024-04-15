import pandas as pd
from sklearn.cluster import KMeans

# Cargar datos
data = pd.read_csv('Pasajeros_Transporte_Masivo.csv')

# Preparar datos (seleccionar características)
X = data[['Pasajeros_dia', 'Dia_Semana']]

# Aplicar análisis de clústeres
kmeans = KMeans(n_clusters=3, random_state=12)
kmeans.fit(X)

# Asignar etiquetas de clúster a los datos
data['cluster'] = kmeans.labels_

# Obtener los grupos y las etiquetas de los grupos
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Imprimir los resultados
print('Grupos encontrados:')
for i in range(3):
    print(f'Grupo {i+1}: {sum(labels == i)} muestras')
    print(f'Centroide del grupo {i+1}: {centroids[i]}')

# Analizar los clústeres
for i in range(3):
    cluster = data[data['cluster'] == i]
    print(f'Clúster {i}:')
    print(cluster.describe())
