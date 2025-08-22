# Customer-Segmentation-with-K-Means-Clustering
This project uses K-Means clustering to segment mall customers based on Age, Annual Income, and Spending Score. It includes data preprocessing, optimal cluster selection using Elbow and Silhouette methods, and 3D visualizations with Matplotlib and Plotly. 

**Overview**

This repository contains a Python script that performs customer segmentation using the K-Means clustering algorithm on the Mall_Customers.csv dataset. The project analyzes customer data based on three features: Age, Annual Income, and Spending Score, to identify distinct customer segments for business insights, such as targeted marketing.

**Features**
-Data Preprocessing: Loads and standardizes data using pandas and scikit-learn's                StandardScaler.
-Cluster Analysis: Determines the optimal number of clusters using the Elbow and Silhouette     methods.
-Clustering: Applies K-Means clustering with 6 clusters to segment customers.

-Visualization:
--Static 3D scatter plot using Matplotlib to show scaled data and centroids.
--Interactive 3D scatter plot using Plotly to visualize original data and centroids.

-Evaluation: Computes inertia and silhouette scores to assess clustering quality.

**Requirements**
-Python 3.x
-Libraries:
--pandas
--matplotlib
--scikit-learn
--plotly

**Install dependencies using:**
-pip install pandas matplotlib scikit-learn plotly

**Dataset**

-The project uses Mall_Customers.csv, which should include columns: Age, Annual Income (k$),    and Spending Score (1-100).
-Place the dataset in the same directory as the script.


**Notes:**

-The script sets n_clusters=6 based on analysis; adjust as needed based on Elbow/Silhouette     results.

-On Windows with Anaconda, you may encounter a K-Means memory leak warning. To suppress it,     add the following at the start of the script:
import os
os.environ["OMP_NUM_THREADS"] = "1"
