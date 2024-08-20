# Personalized emails - CLUSTERING

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters 
LabeledFileName =  r'./intermediate_data/Amazon_Reviews_Labeled.csv'
CusteredFileName = './intermediate_data/Amazon_Reviews_Clustered.csv'
CusteringResultsFileName = './intermediate_data/Amazon_Reviews_ClusteringResults.csv'

pd.options.display.float_format = '{:,.2f}'.format 

# *************** Data Loading & Preview ***************

# LOADING Labeled File
df = pd.read_csv(LabeledFileName)

# Initial Data Profiling 
print(f"Shape: {df.shape}\n")
print(f"DataFrame Info: {df.info()}\n")

# Missing Values Check
print(f"Original Missing Values:\n{'-'*40}\n{df.isnull().sum()}\n")
print(f"Confidence value counts:\n{'-'*40}")
print(df["Confidence"].value_counts(dropna=False).head(10),'\n')

# **************** Data Preprocessing ******************
# It seems that there in no need for handling missing values 

# Filter out low confidence rows (confidence should be > 2 to count for the clustering process), since there is a small number of records 
# with Confidence <=2
df = df[df['Confidence'] > 2]
df = df.drop('Confidence', axis =1)

# Mapping Sentiment to numerical values
SentimentNum = df["Sentiment"].map({'very negative': 1, 'negative': 2, 'neutral': 3, 'positive': 4, 'very positive': 5})
df.insert(5, "SentimentNum", SentimentNum)
df["SentimentNum"] = df.SentimentNum.astype(int)

# Filter rows where the absolute difference between SentimentNum | Satisfaction and Score is within a  discrepancy_threshold
discrepancy_threshold = 2
df = df[abs(df['SentimentNum'] - df['Score']) <= discrepancy_threshold] 
df = df[abs(df['Satisfaction'] - df['Score']) <= discrepancy_threshold] 

# Define groups of features to facilitate next processes
cat_features = df.select_dtypes(exclude="number").columns.drop(['ProductId', 'Comment', 'UserId'])
num_features = df.select_dtypes(include='number').columns

# Value Counts
print(f"Categorial Features Value Counts\n{'-'*40}")
for col in cat_features:
    print(df[col].value_counts(dropna=False).head(10),'\n')

df.describe()

# ******************* Data Visualization ************************
def show_histograms():
    fig, axes = plt.subplots(2, 2, figsize=(6, 4))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot histograms for each feature
    for i, feature in enumerate(num_features):
        sns.histplot(df[num_features[i]], bins=5, kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {num_features[i]}', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

show_histograms()

# Calculate correlations
correlation_matrix = df[num_features].corr()

# Plotting heatmap
plt.figure(figsize=(5, 3))
sns.heatmap(correlation_matrix, annot=True, annot_kws={"size": 10},  cmap='coolwarm', fmt=".2f")
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.title('Correlation between fields')
plt.show()

# *************** Feature Engineering ****************
# Feature scaling
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[num_features])

# *************** Custering *************************

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(6, 5))
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# It seems that 5 clusters is the optimal number 
k = 5

# K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)  # Example with 3 clusters
kmeans.fit(scaled_df)

# Adding cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Saving the results 
df.to_csv(CusteredFileName, index=False)

# Display the DataFrame with cluster labels
df.head()

# KMeans results overview
aggregated_df = df.groupby(['Cluster']).agg({
    'SentimentNum': ['mean', 'std'],
    'Aggressiveness': ['mean', 'std'],
    'Satisfaction': ['mean', 'std'],
    'Score': ['mean', 'std','count'],
}).reset_index()
aggregated_df.columns = ['Cluster', 
                         'Sentiment_mean', 'Sentiment_std', 
                         'Aggressiveness_mean', 'Aggressiveness_std', 
                         'Satisfaction_mean', 'Satisfaction_std', 
                         'Score_mean', 'Score_std','Count'
                        ]
# Moving 'Count' to be the second column of the results
aggregated_df.insert(1, 'Count', aggregated_df.pop('Count'))

aggregated_df.to_csv(CusteringResultsFileName, index=False)
aggregated_df

# ***************** Validation *********************
# Silhouette Score: This metric provides an indication of how similar each point is to points in its own cluster compared to points in other clusters. 
# A higher silhouette score indicates better-defined clusters.

# Validate the results using silhouette score
silhouette_avg = silhouette_score(scaled_df, kmeans.labels_)
print(f"\nSilhouette Score: {silhouette_avg:.3f}","\n\n")

# Optional: Plot clusters (for visualization if features are 2D or 3D)
plt.figure(figsize=(6, 4))
plt.scatter(df['SentimentNum'], df['Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel('SentimentNum', fontsize=9)
plt.ylabel('Score', fontsize=9)
plt.title('K-means Clustering')
plt.colorbar(label='Cluster')
plt.show()