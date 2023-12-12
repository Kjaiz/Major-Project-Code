# Major-Project-Code
import piplite
await piplite.install('seaborn')
import piplite
await piplite.install('yellowbrick')
# Importing the Libraries 
import numpy as np 
import pandas as pd 
import datetime 
import matplotlib 
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
from matplotlib import colors 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from yellowbrick.cluster import KElbowVisualizer 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt, numpy as np 
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import AgglomerativeClustering 
from matplotlib.colors import ListedColormap 
from sklearn import metrics 
import warnings 
import sys 
if not sys.warnoptions: 
 warnings.simplefilter("ignore")
np.random.seed(42) 
# Loading the dataset 
dataset = pd.read_csv("college.csv",on_bad_lines='skip') 
dataset.head()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'student_data' with the relevant columns
# 'Travel Time' and 'Region', and you've already loaded your data into it.

# Geographic Segmentation
region_counts = dataset['Which region do you belong to?\n\n'].value_counts()
region_counts.plot(kind='bar', title='Geographic Segmentation')
plt.xlabel('Region')
plt.ylabel('Number of Students')
plt.show()

# Travel Time Influence Segmentation
travel_time_counts = dataset['Does Travel Time plays key role in your choice making?'].value_counts()
travel_time_counts.plot(kind='bar', title='Travel Time Influence Segmentation')
plt.xlabel('Travel Time Influence')
plt.ylabel('Number of Students')
plt.show()

# We need to remove the NA values from our dataset, so we will use .dropna() 
datasetdataset = dataset.dropna() 
no=len(dataset) 
print("After eliminating the rows with missing values, there are ultimately {no} number of datapoints in the dataset ")

# Obtain a list of the category variables 
s = (dataset.dtypes == 'object') 
object_columns = list(s[s].index) 
 
print("the dataset's categorical variables are: \n", object_columns,"\
n")
# The object dtypes are label encoded. 
LE=LabelEncoder() 
for i in object_columns: 
 dataset[i]=dataset[[i]].apply(LE.fit_transform) 
 
print("Now, all attributes are numerical.")

dataset.head(289)

# making a duplicate of the data 
copy_dataset = dataset.copy() 
# Removing the features on deals accepted and promotions to create a subset of the dataframe 
columns_to_delete = ['Any suggested changes in your college?'] 
copy_datasetcopy_dataset = copy_dataset.drop(columns_to_delete, axis=1) 
# Scaling 
standard_scaler = StandardScaler() 
standard_scaler.fit_transform(copy_dataset) 
scaled_dataset = pd.DataFrame(standard_scaler.transform(copy_dataset),columns= copy_dataset.columns ) 
print(" Now, every feature is scaled ") 

# Using scaled data to reduce the dimensionality
print("Dataframe to be applied in further modelling:")
scaled_dataset.head(289)

x = dataset.iloc[:,].values 
#finding optimal number of clusters using the elbow method 
from sklearn.cluster import KMeans 
wcss_list= [] #Initializing the list for the values of WCSS 
 
#Using for loop for iterations from 1 to 10. 
for i in range(1, 11): 
 kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42) 
 kmeans.fit(x) 
 wcss_list.append(kmeans.inertia_) 
plt.plot(range(1, 11), wcss_list) 
plt.title('The Elobw Method Graph') 
plt.xlabel('Number of clusters(k)') 
plt.ylabel('wcss_list') 
plt.show()

algorithm = (KMeans(n_clusters = 8 ,init='k-means++', n_init =
10 ,max_iter=300, 
 tol=0.0001, random_state= 111 , 
algorithm='elkan') )
algorithm.fit(x)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_
y_kmeans = algorithm.fit_predict(x)
scaled_dataset['cluster'] = pd.DataFrame(y_kmeans)
scaled_dataset.head(20)

scaled_dataset.fillna(-999, inplace=True)
#Initiating PCA to reduce dimensions, aka features, to 3 
pca = PCA(n_components=3) 
pca.fit(scaled_dataset) 
PCA_dataset = pd.DataFrame(pca.transform(scaled_dataset), columns=(["col1","col2", "col3"])) 
PCA_dataset.describe().T 

# To find the accuracy of the data

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Assuming you have ground truth labels in a 'true_labels' column of your dataset
true_labels = scaled_dataset['Age']  # Replace 'true_labels' with the actual column name

# Assuming you have obtained cluster assignments using K-Means
kmeans_labels = scaled_dataset['cluster']  # Replace 'cluster' with the actual column name

# Calculate Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, kmeans_labels)

# Calculate Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(true_labels, kmeans_labels)

print(f'Adjusted Rand Index (ARI): {ari}')
print(f'Normalized Mutual Information (NMI): {nmi}')

# Standardize numerical data
scaler = StandardScaler()
cluster_data[numerical_columns] = scaler.
↪fit_transform(cluster_data[numerical_columns])
# Apply K-Prototypes algorithm
kproto = KPrototypes(n_clusters=8, init='Cao', n_init=1, verbose=2)
clusters = kproto.fit_predict(cluster_data.values, categorical=[2, 3])
# Add cluster labels to the original dataset
data['Cluster'] = clusters
# Plot the bar graph of cluster distribution
plt.figure(figsize=(6, 6))
data['Cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Data Points Across Clusters')
plt.xlabel('Cluster Number')
plt.ylabel('Number of Data Points')
plt.show()
# Map cluster numbers to meaningful names
cluster_names = {
0: 'Practical Planners',
1: 'Social Influencers',
2: 'Budget-Conscious Seekers',
3: 'Academic Enthusiasts',
4: 'Sports and Extracurricular Enthusiasts',
5: 'Regional Trend Followers',
6: 'Language-Priority Seekers',
7: 'Internship and Placement Focused'
}
# Add cluster labels to the original dataset
data['Cluster'] = clusters
data['Cluster Name'] = data['Cluster'].map(cluster_names)
# View the resulting clusters
print(data[['Your current status among the following?', 'Which Language you␣ prefer?', 'Cluster', 'Cluster Name']])


