# AI based Fashion Catalogue
**Image Clustering using Machine Learning**

![image](https://github.com/Msgasu/AI-based-Fashion-Catalogue-Final-Project/assets/148695396/fa1c9a9e-931d-4bdd-9350-084e221db612)


## Overview:

This code is designed to perform image clustering using machine learning techniques. 
The primary goal is to group similar images into clusters, enabling better organization and analysis of large image datasets.
The model used for clustering was a KMeans model but a pre- trained Vgg16 model was used to extract the features from the images.

## Dependencies:
Required libraries installed before running the code.  
**Commands to install the necessary dependencies:**

   -!pip install opencv-python
   -!pip install numpy
   -!pip install scikit-learn
   -!pip install matplotlib

## Setup:

1. **Mount Google Drive:**
   - This code assumes your images are stored in a Google Drive folder. Make sure to mount your Google Drive using the provided code:
   drive.mount('/content/drive')


2. **GPU Configuration:**
   - The code checks for the availability of GPUs and configures TensorFlow to use them if available.
   - GPU was used instead of CPU to speed up the rate at which the features were extracted and the rate at which the model was trained.

3. **Image Folder Path:**
   - Update the `image_folder_path` variable with the path to your image folder in Google Drive( that is if you are using Google Drive).

## Data Reading and Processing:

- The code reads, processes, and analyzes images using a pre-trained VGG16 model.
- Images are resized to 120x120 pixels and converted to RGB format.

## Training 3 Models:

- The code trains three clustering models: KMeans, DBSCAN, and AgglomerativeClustering.
- Evaluation metrics such as Calinski-Harabasz Index and Silhouette Score are used to choose the best model (KMeans).

## KMeans Clustering:

- The selected KMeans model is used to cluster images into three groups based on extracted features.
- Visualizations, such as PCA scatter plots and image reconstructions, are provided to analyze the clustering results.

## Exploratory Data Analysis (EDA):

- The code performs exploratory data analysis on the clustered images, including cluster size distribution, image size distribution, and color distribution.

## Testing:

- Test images are processed and assigned to clusters using the trained KMeans model.
- Evaluation metrics for test images are calculated to assess the model's performance on new data.

## Saving Models for Deployment:

- The trained KMeans and PCA models are saved using the `joblib` library for future deployment.

## Instructions for Use:

1. Mount Google Drive and configure GPU settings.
2. Update the `image_folder_path` variable with the path to your image folder.
3. Install dependencies.
4. Run the code sections sequentially.

Feel free to adapt the code to your specific use case and image dataset. If you encounter any issues or have questions, please refer to the code comments or seek assistance.
