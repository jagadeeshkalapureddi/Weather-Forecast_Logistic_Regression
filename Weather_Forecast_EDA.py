#!/usr/bin/env python
# coding: utf-8

# # `-------------------@ WEATHER_CLOUD CONDITION @----------------------------------------! EXPLORATORY DATA ANALYSIS !---------------------`

# ### `IMPORT ALL THE REQUIRED PACKAGES - EDA:`

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from matplotlib.cm import get_cmap
from sklearn import preprocessing
from sklearn.preprocessing import scale, StandardScaler
from sklearn.cluster import KMeans
from random import sample
from sklearn.preprocessing import OrdinalEncoder


# ### `READ THE DATASET:`

# In[ ]:


df = pd.read_csv('train_CloudCondition.csv')
test = pd.read_csv('test_CloudCondition.csv')


# ## `APPEARANCE OF THE DATASET:`

# ### `FIRST FIVE RECORDS OF THE DATASET:`

# In[ ]:


df.head()


# ### `SHAPE OR DIMENSIONS OF THE DATASET:`

# In[ ]:


print('The Shape or Dimensions of the dataset having the rows of "{}" and Columns of "{}".'. format(df.shape[0], df.shape[1]))


# ### `INFO OF THE DATASET:`

# In[ ]:


df.info()


# In[ ]:


df = df.rename(columns = {"Temperature (C)":"Temperature", "Apparent Temperature (C)" : "Apparent Temperature", "Wind Speed (km/h)" : "Wind Speed", "Wind Bearing (degrees)" : "Wind Bearing", "Visibility (km)" : "Visibility", "Pressure (millibars)" : "Pressure"})


# In[ ]:


test = test.rename(columns = {"Temperature (C)":"Temperature", "Apparent Temperature (C)" : "Apparent Temperature", "Wind Speed (km/h)" : "Wind Speed", "Wind Bearing (degrees)" : "Wind Bearing", "Visibility (km)" : "Visibility", "Pressure (millibars)" : "Pressure"})


# In[ ]:


df.columns


# ### `CHECKING FOR THE NA's IN DATASET:`

# In[ ]:


df.isna().sum()


# ### `CHANGE THE DATA TYPE OF THE TEMPERATURE FROM OBJECT TO INT :`

# In[ ]:


df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df = df.dropna(subset=['Temperature'])
df['Temperature'] = df['Temperature'].astype(int)
df.Temperature.dtype


# ### `MISSING VALUES TREATMENT:`

# In[ ]:


df.dropna(inplace=True, axis = 0)


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# ### `Statistical Information :`

# In[ ]:


df.describe()


# ### `CATEGORICAL VARIABLES EDA`

# In[ ]:


cat = df[df.columns[[1,2,10]]] 


# In[ ]:


cat.head()


# ### `Cloud_Condition - Attribute`

# In[ ]:


print('Column_name : ' ,cat.iloc[:,0].name)
print('Type : ',cat.iloc[:,0].dtype)

print('Null_value_count: ',cat.iloc[:,0].isna().sum())


# In[ ]:


cat.iloc[:,0].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
cat.iloc[:,0].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel(cat.iloc[:,0].name, fontsize = 20)
plt.title('Barplot_ '+ cat.iloc[:,0].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.9, top=1.2)


# ### `Rain_OR_SNOW - Attribute`

# In[ ]:


print('Column_name : ' ,cat.iloc[:,1].name)
print('Type : ',cat.iloc[:,1].dtype)

print('Null_value_count: ',cat.iloc[:,1].isna().sum())


# In[ ]:


cat.iloc[:,1].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
cat.iloc[:,1].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel(cat.iloc[:,1].name, fontsize = 20)
plt.title('Barplot_ '+ cat.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### `Condensation - Attribute`

# In[ ]:


print('Column_name : ' ,cat.iloc[:,2].name)
print('Type : ',cat.iloc[:,2].dtype)

print('Null_value_count: ',cat.iloc[:,2].isna().sum())


# In[ ]:


cat.iloc[:,2].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
cat.iloc[:,2].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel(cat.iloc[:,2].name, fontsize = 20)
plt.title('Barplot_ '+ cat.iloc[:,2].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### `NUMERICAL VARIABLES EDA`

# In[ ]:


num = df[df.columns[[0,3,4,5,6,7,8,9,11]]] 


# In[ ]:


num.head()


# ### `Day - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,0].name)
print('Type : ',num.iloc[:,0].dtype)
print('Null_value_count: ',num.iloc[:,0].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,0].skew())
num.iloc[:,0].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,0], color = 'green')
plt.xlabel(num.iloc[:,0].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,0].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,0], color = 'skyblue')
plt.xlabel(num.iloc[:,0].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,0].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Day" column has no Outliers in its dataset.`

# ### `Temperature - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,1].name)
print('Type : ',num.iloc[:,1].dtype)
print('Null_value_count: ',num.iloc[:,1].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,1].skew())
num.iloc[:,1].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,1], color = 'green')
plt.xlabel(num.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,1], color = 'skyblue')
plt.xlabel(num.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Temperature" column has no Outliers in its dataset.`

# ### `Apparent Temperature - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,2].name)
print('Type : ',num.iloc[:,2].dtype)
print('Null_value_count: ',num.iloc[:,2].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,2].skew())
num.iloc[:,2].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,2], color = 'green')
plt.xlabel(num.iloc[:,2].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,2].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,2], color = 'skyblue')
plt.xlabel(num.iloc[:,2].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,2].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Apparent Temperature" column has no Outliers in its dataset.`

# ### `Humidity - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,3].name)
print('Type : ',num.iloc[:,3].dtype)
print('Null_value_count: ',num.iloc[:,3].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,3].skew())
num.iloc[:,3].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,3], color = 'green')
plt.xlabel(num.iloc[:,3].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,3].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,3], color = 'skyblue')
plt.xlabel(num.iloc[:,3].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,3].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Humidity" column has no Outliers in its dataset.`

# ### `Wind Speed - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,4].name)
print('Type : ',num.iloc[:,4].dtype)
print('Null_value_count: ',num.iloc[:,4].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,4].skew())
num.iloc[:,4].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,4], color = 'green')
plt.xlabel(num.iloc[:,4].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,4].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,4], color = 'skyblue')
plt.xlabel(num.iloc[:,4].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,4].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Wind Speed" column has no Outliers in its dataset.`

# ### `Wind Bearing - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,5].name)
print('Type : ',num.iloc[:,5].dtype)
print('Null_value_count: ',num.iloc[:,5].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,5].skew())
num.iloc[:,5].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,5], color = 'green')
plt.xlabel(num.iloc[:,5].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,5].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,5], color = 'skyblue')
plt.xlabel(num.iloc[:,5].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,5].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Wind Bearing" column has no Outliers in its dataset.`

# ### `Visibility - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,6].name)
print('Type : ',num.iloc[:,6].dtype)
print('Null_value_count: ',num.iloc[:,6].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,6].skew())
num.iloc[:,6].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,6], color = 'green')
plt.xlabel(num.iloc[:,6].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,6].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,6], color = 'skyblue')
plt.xlabel(num.iloc[:,6].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,6].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Visibility" column has no Outliers in its dataset.`

# ### `Pressure - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,7].name)
print('Type : ',num.iloc[:,7].dtype)
print('Null_value_count: ',num.iloc[:,7].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,7].skew())
num.iloc[:,7].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,7], color = 'green')
plt.xlabel(num.iloc[:,7].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,7].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,7], color = 'skyblue')
plt.xlabel(num.iloc[:,7].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,7].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Pressure" column has no Outliers in its dataset.`

# ### `Solar irradiance intensity - Attribute`

# In[ ]:


print('Column_name : ' ,num.iloc[:,8].name)
print('Type : ',num.iloc[:,8].dtype)
print('Null_value_count: ',num.iloc[:,8].isna().sum())


# In[ ]:


print('Skewness: ', num.iloc[:,8].skew())
num.iloc[:,8].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(num.iloc[:,8], color = 'green')
plt.xlabel(num.iloc[:,8].name, fontsize = 20)
plt.title('Boxplot_ '+ num.iloc[:,8].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(num.iloc[:,8], color = 'skyblue')
plt.xlabel(num.iloc[:,8].name, fontsize = 20)
plt.title('Histogram_ '+ num.iloc[:,8].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### `Through Skewness, Box-Plot and Histogram plot found that the "Solar Irradiance Intensity" column has no Outliers in its dataset.`

# ### `Now the data is clean and neat without Outliers and Missing values.`

# ### `Merge the cleaned data into one dataframe`

# In[ ]:


fdf = pd.concat([num,cat], axis=1)


# In[ ]:


fdf.head()


# ### `VISUALIZATION :`

# #### `Heat Map-Plot`

# In[ ]:


f = plt.figure(figsize=(10, 10))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# #### `Correlation-Plot`

# In[ ]:


corr = fdf.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# #### `Pair-Plot`

# In[ ]:


ord_enc = OrdinalEncoder()
fdf[['Rain_OR_SNOW1', 'Condensation1','Cloud_Condition1']] = ord_enc.fit_transform(fdf[['Rain_OR_SNOW', 'Condensation','Cloud_Condition']])
fdf[['Rain_OR_SNOW1', 'Condensation1','Cloud_Condition1']].head(11)


# In[ ]:


fdf = fdf.drop(['Rain_OR_SNOW', 'Condensation','Cloud_Condition'], axis = 1)


# In[ ]:


fdf.head()


# In[ ]:


pvflights = fdf.pivot_table(values='Temperature',index='Rain_OR_SNOW1',columns='Cloud_Condition1')
sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)


# ### `K-MEANS`

# #### `Set the first column as an index`

# In[ ]:


# Set the state as an index.
fdf = fdf.set_index('Day')
fdf.head()


# #### `Select samples from the data randomly`

# In[ ]:


sample = fdf.sample(frac = 0.30, replace = False, random_state = 100)
print(len(sample))
sample


# #### `Scaling`

# In[ ]:


data_scaled = StandardScaler().fit_transform(fdf)
data_scaled


# #### `Optimization Plot with Elbow Method`

# In[ ]:


plt.figure(figsize = (10,3))
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1,11), wcss, 'bx-')
plt.title('The Elbow Method')
plt.axvline(4, color = 'red', linestyle = '--')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()


# #### `Cluster Membership`

# In[ ]:


kmeans = KMeans(n_clusters = 4) # just making the cluster in the backned (not fitted to dataset here)
clusters = kmeans.fit_predict(data_scaled)
clusters


# #### `Add Cluster Column to the actual data`

# In[ ]:


# Let's add column 'cluster' to the data

Final_cluster = clusters + 1
Cluster = list(Final_cluster)
fdf['cluster'] = Cluster
fdf.head()


# #### `Cluster - 1`

# In[ ]:


fdf[fdf['cluster'] == 1]


# #### `Cluster - 2`

# In[ ]:


fdf[fdf['cluster'] == 2]


# #### `Cluster - 3`

# In[ ]:


fdf[fdf['cluster'] == 3]


# #### `Cluster - 4`

# In[ ]:


fdf[fdf['cluster'] == 4]


# #### `Cluster Profiling`

# In[ ]:


fdf.groupby('cluster').mean()


# #### `Cluster Plot`

# In[ ]:


plt.figure(figsize = (12,6))
sns.scatterplot(df['Temperature'], df['Cloud_Condition'], hue = Final_cluster, palette = ['green', 'orange', 'blue', 'red'])


# ### `Cleaned Dataset for Main Dataset`

# In[ ]:


fdf.to_csv('Cleaned_set.csv')

