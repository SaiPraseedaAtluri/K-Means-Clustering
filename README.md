# Customer Segmentation Using K-Means Clustering

This repository contains a project for customer segmentation using the K-Means clustering algorithm. The goal is to group customers into distinct clusters based on their demographic and purchasing behavior to better understand customer segments and target them effectively.

## Project Overview

The project involves the following steps:

1. **Data Preparation**
2. **Optimal Number of Clusters**
3. **K-Means Clustering**
4. **Evaluation and Visualization**
5. **Detailed Cluster Properties**

### Data Preparation

- **Load the data**: The dataset (`customers_data.csv`) is loaded using `pandas`.
- **Define features**: Numeric features include `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. The categorical feature is `Gender`.
- **Preprocessing**: Scaling for numeric features using `StandardScaler` and one-hot encoding for the categorical feature using `OneHotEncoder`.
- **Drop unnecessary columns**: The `CustomerID` column is dropped as it is not needed for clustering.

### Optimal Number of Clusters

- **Elbow Method**: Inertia (sum of squared distances to the nearest cluster center) is plotted against the number of clusters to find the "elbow point".
- **Silhouette Scores**: Average silhouette scores are calculated for different numbers of clusters to evaluate cluster cohesion and separation.

### K-Means Clustering

- **Applying K-Means**: The K-Means algorithm is applied with the optimal number of clusters.
- **Cluster Labels**: Cluster labels are added to the original dataset.

### Evaluation and Visualization

- **PCA for Dimensionality Reduction**: Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset for visualization.
- **Cluster Visualization**: Clusters are visualized in a 2D space.

### Detailed Cluster Properties

- **Cluster Summaries**: Statistical summaries for each cluster, including mean and standard deviation for numeric features and gender distribution.

## Repository Structure

- `customers_data.csv`: The dataset containing customer information.
- `K-Means.ipynb`: Jupyter notebook with the complete code for the project.
- `README.md`: Description and instructions for the project (this file).

## Installation and Usage

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/customer-segmentation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd customer-segmentation
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter notebook:
   ```bash
   jupyter notebook customer_segmentation.ipynb
   ```


## Output Details

### Elbow Method and Silhouette Scores

- **Elbow Method**: The plot indicates the point where the inertia starts decreasing more slowly. This point suggests the optimal number of clusters.
- **Silhouette Scores**: The plot shows the average silhouette scores for different numbers of clusters. Higher scores indicate better-defined clusters.

### Cluster Centers

The cluster centers represent the average values of the features for each cluster:


### Detailed Cluster Properties

For each cluster, statistical summaries and gender distributions are provided.

## Acknowledgments

- This project is based on a customer dataset that contains information on age, annual income, spending score, and gender.
