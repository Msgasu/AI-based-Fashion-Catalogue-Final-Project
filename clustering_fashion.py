# -*- coding: utf-8 -*-
"""Clustering_Fashion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Aj1kvVK4qgXv6Sphfdp577jx-egQLCdZ

## Data reading and processing

*Importing necessary libraries and resources that will be needed during the execution of the program.*
"""

! pip install opencv-python
! pip install numpy
! pip install scikit-learn
! pip install matplotlib

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from IPython.display import display, Image
from sklearn.metrics import silhouette_score
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import  calinski_harabasz_score, silhouette_score
from sklearn.cluster import DBSCAN, Birch, AgglomerativeClustering
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

drive.mount('/content/drive')

"""*Checking for the availability of GPU(s) using TensorFlow and configuring TensorFlow to use the GPU(s) if available*"""

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU')
    except RuntimeError as e:

        print(e)
else:
    print("No GPU available, using the CPU instead.")

"""*Storing the path to the folder with the images in the google drive in a variable*"""

image_folder_path = '/content/drive/My Drive/AI Final /Clustering Images'

"""**A function to process the images in the folder**

*It reads images from the folder, resizes it to a specified target size(width and height 120), converts it to RGB format, and adds an extra dimension to the array before returning the processed image*
"""

def process_image(image_path, target_width, target_height):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    return img_array

"""**Displays the total number of images in the folder**"""

image_files = os.listdir(image_folder_path)
print(f"Number of images in the folder: {len(image_files)}")

"""**Checks the number of unique image filenames**"""

unique_image_files = set(image_files)
print(f"Number of unique image filenames: {len(unique_image_files)}")

"""**storing information about each image in image_info list, while the actual processed images are stored in the list x. The target width and height (120x120 pixels) represents the width and height for resizing**"""

image_info = [] # List to store info about processed images

target_width = 120
target_height = 120

x = []  # List to store processed images

"""**This block of code checks for valid image extensions, processes the image using the process_image function , and then appends to the list x.
Also the code Stores information about the original image in imgae_info and finally prints the total number of images in each list so I can keep track and make sure there are no duplications.**
"""

for image_file in unique_image_files:

    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, image_file)
        img_array = process_image(image_path, target_width, target_height)
        x.append(img_array)

        image_info.append({'file_name': image_file, 'image_path': image_path})

print(f"Number of items in 'x': {len(x)}")
print(f"Number of items in 'image_info': {len(image_info)}")

"""**Displaying the contents of list X**"""

x

"""**Displaying the content of the image_info list**"""

image_info

"""**Loading the pretrained model and converting the image to an array**

*I used a pretrained model (Vgg16)*
"""

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(target_width, target_height, 3))

# Converting the list of images to a numpy array
x = np.vstack(x)
x = preprocess_input(x)

"""**Extracting the image features using the pretarained model I loaded earlier**

*After extracting the features I reshaped the images to have a flat representation. Next I applied Principal Component Analysis (PCA) for dimensionality reduction on the flattened features*
"""

features = vgg16_model.predict(x)

features_flattened = features.reshape((len(features), -1))

pca = PCA(n_components=50)
features_pca = pca.fit_transform(features_flattened)

"""**Displaying the image shape**

*I did this to check the  output of the VGG16 model after extracting features from the preprocessed images. The shape of features provides information about the dimensions of the feature representation for each image*
"""

features.shape

"""## Training  3 models and picking the best (KMeans model) out of the 3

**The four models I trained were:**

I trained the models using Gridsearch and cross validation based on the features extracted from the pre- trained model.
*   KMeans
*   DBSCAN
*  AgglomerativeClustering

**The metrics I used to evaluate and and pick the best model were:**

*   calinski_harabasz_score
*   silhouette_score
*   WSS (specific to only the KMeans model)

**Getting the silhouette score for the KMeans model**

I used this to also get the number of clusters that will get the best silhouette score. Upon printing the silhoutte score , I realized a cluster of 2 had a hgher silhoutte score. I however used 3 for my number of clusters (k=3), I did this because the cluster of 2 was not producing the best clustering results. This will be shown in the code blocks below.
"""

silhouette_scores = []
max_k = 5  # This is based on user specification

for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features_pca)
    score = silhouette_score(features_pca, labels)
    silhouette_scores.append(score)
    print(f"Silhouette Score for {k} clusters: {score}")

"""
**plotting the Silhouette Score**
"""

plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.show()

"""**Grid Search and cross validation to Get the metric scores for the different models.**

**Function to calculate silhouette score**
"""

def clustering_scorer(model, X):
    labels = model.fit_predict(X)
    if len(np.unique(labels)) == 1:
        return 0  # Silhouette score is not defined for a single cluster
    return silhouette_score(X, labels)

"""**Parameters for GridSearch with Cross Validation**"""

param_grid_dbscan = {'eps': [0.5, 1.0, 1.5], 'min_samples': [5, 10, 15]}
param_grid_agglo = {'n_clusters': [2, 3, 4, 5, 6], 'linkage': ['ward', 'complete', 'average', 'single']}
param_grid_kmeans = {'n_clusters': [2, 3, 4, 5],'init': ['k-means++', 'random'],'max_iter': [100, 200, 300]}

"""**Models**

"""

dbscan = DBSCAN()
agglo = AgglomerativeClustering()
kmeans = KMeans()

"""**GridSearchCV with silhouette score as the custom scorer**"""

grid_search_dbscan = GridSearchCV(dbscan, param_grid_dbscan, cv=5, scoring=clustering_scorer,)
grid_search_agglo = GridSearchCV(agglo, param_grid_agglo, cv=5, scoring=clustering_scorer)
grid_search_kmeans = GridSearchCV(kmeans, param_grid_kmeans, cv=5)

"""**Fitting the grid searches to the data**"""

grid_search_agglo.fit(features_pca)

grid_search_kmeans.fit(features_pca)

grid_search_dbscan.fit(features_pca)

"""**Extracting the best models and their hyperparameters**"""

best_dbscan_model = grid_search_dbscan.best_estimator_
best_agglo_model = grid_search_agglo.best_estimator_
best_kmeans_model = grid_search_kmeans.best_estimator_

"""**Printing the best hyperparameters**"""

print("Best Hyperparameters for DBSCAN:", grid_search_dbscan.best_params_)
print("Best Hyperparameters for Agglomerative:", grid_search_agglo.best_params_)
print("Best Hyperparameters for KMeans:", grid_search_kmeans.best_params_)

"""**Predict clusters**"""

clusters_dbscan = best_dbscan_model.fit_predict(features_pca)
clusters_agglo = best_agglo_model.fit_predict(features_pca)
clusters_kmeans = best_kmeans_model.fit_predict(features_pca)

"""**Evaluating the metrics of each model**"""

# Evaluation metrics for DBSCAN
print("\nEvaluation Metrics for DBSCAN:")
if len(np.unique(clusters_dbscan)) > 1:
    print("Calinski-Harabasz Index:", calinski_harabasz_score(features_pca, clusters_dbscan))
    print("Average Silhouette Score:", silhouette_score(features_pca, clusters_dbscan))
else:
    print("Insufficient number of clusters for Calinski-Harabasz Index and Silhouette Score.")

# Evaluation metrics for Agglomerative
print("\nEvaluation Metrics for Agglomerative:")
print("Calinski-Harabasz Index:", calinski_harabasz_score(features_pca, clusters_agglo))
print("Average Silhouette Score:", silhouette_score(features_pca, clusters_agglo))

#Evaluation metrics for KMeans
print("\nEvaluation Metrics for KMeans:")
print("Calinski-Harabasz Index:", calinski_harabasz_score(features_pca, clusters_kmeans))
print("Average Silhouette Score:", silhouette_score(features_pca, clusters_kmeans))
wss_kmeans = np.sum((features_pca - best_kmeans_model.cluster_centers_[clusters_kmeans]) ** 2)
print("Within-Cluster Sum of Squares (WSS) for KMeans:", wss_kmeans)



"""Agglomerative Clustering has a higher Silhouette Score, indicating better-defined clusters, but the Calinski-Harabasz Index is lower.
KMeans has a higher Calinski-Harabasz Index but a lower Silhouette Score and a relatively high WSS.
For that reason I decided to go with the KMeans model.I could not evaluate the metrics of the DBSCAN model so it was not an option.

## KMeans

**using the KMeans clustering algorithm to cluster data into three groups**

The reason I used 3 clusters instead of 4 or 2 even though they gave a higher silhoutte score is because , when I used 2 clusters, the clusters were not really well clustered. When I used 4 as the cluster number, 2 of the clusters were the same item so I just decided to go with 3 clusters. After doing this I had good clusters. I used the metrics to get the best model and now I wil train and visualize the clusters of images using the kMeans model.
"""

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features_pca)

"""**Visualizing Cluster Results with Reconstruction**

Here the images are being clustered and smaple images from each cluster are displayed to show if the images are being clustered well.
"""

for cluster_id in range(num_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    sample_indices = np.random.choice(cluster_indices, size=min(3, len(cluster_indices)), replace=False)
    for i in sample_indices:
        # Retrieving original image information
        original_info = image_info[i]
        original_image_path = original_info['image_path']
        # Displaying the reconstructed image
        plt.imshow(cv2.imread(original_image_path))
        plt.title(f'Cluster {cluster_id} - Original Image: {original_info["file_name"]}')
        plt.axis('off')
        plt.show()

"""**Visualizing and Analyzing Results with PCA Scatter Plot**

**Analysis:** From the scatter plot it can be seen that the clusters are well defined showing that the images are clustering well
"""

plt.figure(figsize=(10, 10))

for cluster_id in range(num_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    plt.scatter(features_pca[cluster_indices, 0], features_pca[cluster_indices, 1], label=f'Cluster {cluster_id}')

plt.title('PCA Scatter Plot of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

"""## Exploratory Data Analysis (EDA)

**Getting a dataframe with the images name , file path  and their specific clusters**
"""

image_df = pd.DataFrame(image_info)

# cluster labels obtained from KMeans
image_df['Cluster'] = clusters

image_df

"""**Cluster Size Distribution**"""

cluster_size_distribution = image_df['Cluster'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=cluster_size_distribution.index, y=cluster_size_distribution.values)
plt.title('Cluster Size Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Images')
plt.show()

"""**Image Size Distribution**"""

image_df['Image Width'] = image_df['image_path'].apply(lambda x: cv2.imread(x).shape[1])
image_df['Image Height'] = image_df['image_path'].apply(lambda x: cv2.imread(x).shape[0])
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(image_df['Image Width'], bins=20, kde=True)
plt.title('Image Width Distribution')

plt.subplot(1, 2, 2)
sns.histplot(image_df['Image Height'], bins=20, kde=True)
plt.title('Image Height Distribution')

plt.tight_layout()
plt.show()

"""**Color Distribution**"""

def plot_color_distribution(cluster_id):
    cluster_images = image_df[image_df['Cluster'] == cluster_id]['image_path'].tolist()
    cluster_colors = []

    for img_path in cluster_images:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cluster_colors.extend(img_rgb.reshape(-1, 3))

    cluster_colors = np.array(cluster_colors) / 255.0  # Normalize to [0, 1]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f'Color Distribution (Cluster {cluster_id})')
    plt.scatter(cluster_colors[:, 0], cluster_colors[:, 1], c=cluster_colors, marker='.')
    plt.xlabel('Red')
    plt.ylabel('Green')

    plt.subplot(1, 2, 2)
    plt.title(f'Color Distribution (Cluster {cluster_id})')
    plt.scatter(cluster_colors[:, 1], cluster_colors[:, 2], c=cluster_colors, marker='.')
    plt.xlabel('Green')
    plt.ylabel('Blue')

    plt.tight_layout()
    plt.show()

# Display color distribution for each cluster
for cluster_id in range(num_clusters):
    plot_color_distribution(cluster_id)

"""## Testing

**Saving the path to the images**
"""

test_image_folder_path = '/content/drive/My Drive/AI Final /test'

"""**Processing images and test images prediction**"""

test_clusters = []
# Processing and predict clusters for each test image
for image_file in os.listdir(test_image_folder_path):
    image_path = os.path.join(test_image_folder_path, image_file)

    # Checking if the file has a valid image extension (e.g., jpg, jpeg, png)
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Process the test image
        test_img_array = process_image(image_path, target_width, target_height)

        # Preprocessing the image for VGG16
        test_img_array_vgg16 = preprocess_input(test_img_array)

        # Using VGG16 model to extract features
        test_features = vgg16_model.predict(test_img_array_vgg16)

        # Flattening the features and reduce dimensionality using the loaded PCA model
        test_features_flattened = test_features.reshape(1, -1)

        # Applying PCA to the test features
        test_features_pca = pca.transform(test_features_flattened)

        # Predicting the cluster for the test image using the loaded KMeans model
        test_cluster_id = kmeans.predict(test_features_pca)[0]
        test_clusters.append(test_cluster_id)


        # Displaying the test image and predicted cluster
        display(Image(filename=image_path, width=150, height=150))
        print(f'Test Image: {image_file} - Predicted Cluster: {test_cluster_id}')

"""**Evaluating test metrics**"""

test_clusters = []
test_features_list = []

for image_file in os.listdir(test_image_folder_path):
    image_path = os.path.join(test_image_folder_path, image_file)

    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):

        test_img_array = process_image(image_path, target_width, target_height)
        test_img_array_vgg16 = preprocess_input(test_img_array)
        test_features = vgg16_model.predict(test_img_array_vgg16)
        test_features_flattened = test_features.reshape(1, -1)
        test_features_pca = pca.transform(test_features_flattened)
        test_cluster_id = kmeans.predict(test_features_pca)[0]
        test_clusters.append(test_cluster_id)
        test_features_list.append(test_features_pca.flatten())


test_clusters = np.array(test_clusters)
test_features_pca = np.vstack(test_features_list)


calinski_test_score = calinski_harabasz_score(test_features_pca, test_clusters)
silhouette_test_score = silhouette_score(test_features_pca, test_clusters)


wss_test_kmeans = np.sum((test_features_pca - kmeans.cluster_centers_[test_clusters]) ** 2)

print(f"Calinski-Harabasz Index for Test: {calinski_test_score}")
print(f"Average Silhouette Score for Test: {silhouette_test_score}")
print(f"Within-Cluster Sum of Squares (WSS) for KMeans on Test: {wss_test_kmeans}")

"""## Test EDA

**Distribution of Clusters**
"""

sns.countplot(x=test_clusters)
plt.title('Distribution of Images Across Clusters')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Images')
plt.show()

"""
**Cluster Size Statistics**"""

cluster_sizes = [len(np.where(test_clusters == cluster_label)[0]) for cluster_label in range(num_clusters)]
print(f"Mean Cluster Size: {np.mean(cluster_sizes)}")
print(f"Median Cluster Size: {np.median(cluster_sizes)}")
print(f"Standard Deviation of Cluster Size: {np.std(cluster_sizes)}")

"""**Feature Distributions**"""

plt.figure(figsize=(10, 6))
for cluster_id in range(num_clusters):
    cluster_indices = np.where(test_clusters == cluster_id)[0]
    plt.scatter(test_features_pca[cluster_indices, 0], test_features_pca[cluster_indices, 1], label=f'Cluster {cluster_id}')

plt.title('PCA Scatter Plot of Test Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

"""## Saving the models needed for deployment (PCA model and KMeans Model)"""

import joblib

joblib.dump(kmeans, 'kmeans_model.joblib')

import joblib
joblib.dump(pca,'pca_model.joblib')

