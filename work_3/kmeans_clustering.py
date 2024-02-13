# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:13:11 2024

@author: aserrasimsek
"""

# Salinas_gt haritası(ground truth) ve Salinas_corrected haritası:

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# MATLAB dosyasından doğrulama verisini yükle
gt_veri = loadmat('Salinas_gt.mat')

# Değişken isimlerini çıkar
gt_degisken_isimleri = gt_veri.keys()

# Zemin doğrulama verisini içeren değişken adını bul
gt_degisken_adi = [var_adi for var_adi in gt_degisken_isimleri if not var_adi.startswith('__')][0]

# Değişken adını kullanarak zemin doğrulama verisine eriş
ground_truth = gt_veri[gt_degisken_adi]

# Zemin doğrulama verisini görselleştir
plt.figure(figsize=(10, 10))
plt.imshow(ground_truth, cmap='tab20', vmin=0, vmax=16)  # 17 sınıf olduğu varsayılarak (0 ile 16 arası)
plt.title("Salinas_gt Haritası")
plt.colorbar()
plt.show()

#*******************************************************************************************************

# MATLAB dosyasından veriyi yükle
mat_veri = loadmat('salinas_corrected.mat')

# Değişken isimlerini çıkar
degisken_isimleri = mat_veri.keys()

# Veriyi içeren değişken adını bul
veri_degisken_adi = [var_adi for var_adi in degisken_isimleri if not var_adi.startswith('__')][0]

# Değişken adını kullanarak veriye eriş
veri = mat_veri[veri_degisken_adi]

# Sahnenin bir alt kümesini görselleştir
alt_kume_veri = veri[:512, :217, :]  

# Görselleştirmek için bir bant seç
gorsellenecek_bant = 100

# Görüntüyü çiz
plt.figure(figsize=(10, 10))
plt.imshow(alt_kume_veri[:, :, gorsellenecek_bant], cmap='gray')
plt.title(f"Salinas_corrected haritası - Bant {gorsellenecek_bant + 5}")
plt.colorbar()
plt.show()



#sklearn kütüphanesi kullanmadan salinas_corrected öbeklenmesi:
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def k_means(data, k, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids

    return labels, centroids

def load_data(file_path):
    mat_data = loadmat(file_path)
    data_variable_name = [var_name for var_name in mat_data if not var_name.startswith('__')][0]
    return mat_data[data_variable_name]

def visualize_clusters(cluster_labels, title):
    plt.imshow(cluster_labels, cmap='tab20')
    plt.title(title)
    plt.colorbar()

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_matrix, cmap='Purples', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

def calculate_error_rate(corrected_labels, ground_truth_flat):
    error_rate = np.sum(corrected_labels != ground_truth_flat) / len(ground_truth_flat)
    print("Clustering Error Rate:", error_rate)

data = load_data('salinas_corrected.mat')
subset_data = data[:512, :217, :] 
reshaped_data = subset_data.reshape((-1, subset_data.shape[2]))

n_clusters = 17
labels, _ = k_means(reshaped_data, n_clusters)
cluster_labels = labels.reshape(subset_data.shape[:2])

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 2)
visualize_clusters(cluster_labels, "K-Means Clustering Result")

ground_truth = load_data('Salinas_gt.mat')

ground_truth_flat = ground_truth.flatten()
cluster_labels_flat = cluster_labels.flatten()
non_zero_indices = np.logical_and(ground_truth_flat != 0, cluster_labels_flat != 0)
ground_truth_non_zero = ground_truth_flat[non_zero_indices]
cluster_labels_non_zero = cluster_labels_flat[non_zero_indices]

confusion_matrix = np.zeros((17, 17))
for i in range(len(ground_truth_non_zero)):
    confusion_matrix[ground_truth_non_zero[i], cluster_labels_non_zero[i]] += 1

plot_confusion_matrix(confusion_matrix)

best_matches = np.argmax(confusion_matrix, axis=1)
corrected_labels = np.zeros_like(cluster_labels_flat)
corrected_labels[non_zero_indices] = best_matches[ground_truth_non_zero]

calculate_error_rate(corrected_labels, ground_truth_flat)


#sklearn kütüphanesi kullanarak salinas_corrected öbeklenmesi:
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans

# Load data from MATLAB file
mat_data = loadmat('salinas_corrected.mat')

# Extract variable names
variable_names = mat_data.keys()

# Find the variable name that contains the data
data_variable_name = [var_name for var_name in variable_names if not var_name.startswith('__')][0]

# Access the data using the variable name
data = mat_data[data_variable_name]

# Visualize a subset of the scene
subset_data = data[:512, :217, :]  

# Reshape the data for k-means clustering
reshaped_data = subset_data.reshape((-1, subset_data.shape[2]))

# Choose the number of clusters 
n_clusters = 17

# Apply k-means clustering
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=100)
clusters = kmeans.fit_predict(reshaped_data)

# Reshape the cluster labels to match the original shape
cluster_labels = clusters.reshape(subset_data.shape[:2])

# Plot the original image with cluster labels
plt.figure(figsize=(12, 6))

# Plot the clustered result
plt.subplot(1, 2, 2)
plt.imshow(cluster_labels, cmap='tab10')
plt.title("K-Means Clustering Result")
plt.colorbar()

plt.show()


def load_data(file_path):
    mat_data = loadmat(file_path)
    data_variable_name = [var_name for var_name in mat_data if not var_name.startswith('__')][0]
    return mat_data[data_variable_name]


def calculate_error_rate(corrected_labels, ground_truth_flat):
    error_rate = np.sum(corrected_labels != ground_truth_flat) / len(ground_truth_flat)
    print("Clustering Error Rate:", error_rate)
    
def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_matrix, cmap='Purples', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()


ground_truth = load_data('Salinas_gt.mat')

ground_truth_flat = ground_truth.flatten()
cluster_labels_flat = cluster_labels.flatten()
non_zero_indices = np.logical_and(ground_truth_flat != 0, cluster_labels_flat != 0)
ground_truth_non_zero = ground_truth_flat[non_zero_indices]
cluster_labels_non_zero = cluster_labels_flat[non_zero_indices]
    

confusion_matrix = np.zeros((17, 17))
for i in range(len(ground_truth_non_zero)):
    confusion_matrix[ground_truth_non_zero[i], cluster_labels_non_zero[i]] += 1

plot_confusion_matrix(confusion_matrix)

best_matches = np.argmax(confusion_matrix, axis=1)
corrected_labels = np.zeros_like(cluster_labels_flat)
corrected_labels[non_zero_indices] = best_matches[ground_truth_non_zero]

calculate_error_rate(corrected_labels, ground_truth_flat)
    