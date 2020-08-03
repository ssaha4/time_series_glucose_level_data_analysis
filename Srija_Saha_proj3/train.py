#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from scipy.fftpack import fft 
from sklearn import metrics
import pickle
import random


# In[2]:


#reading the input files
meal_data1 = pd.read_csv('mealData1.csv',names=list(range(0,30)),nrows=50,header=None)
meal_data2 = pd.read_csv("mealData2.csv",names=list(range(0,30)),nrows=50,header=None)
meal_data3 = pd.read_csv("mealData3.csv",names=list(range(0,30)),nrows=50,header=None)
meal_data4 = pd.read_csv("mealData4.csv",names=list(range(0,30)),nrows=50,header=None)
meal_data5 = pd.read_csv("mealData5.csv",names=list(range(0,30)),nrows=50,header=None)


# In[3]:


mealamount_data1 = pd.read_csv('mealAmountData1.csv',nrows=50,header=None)
mealamount_data2 = pd.read_csv("mealAmountData2.csv",nrows=50,header=None)
mealamount_data3 = pd.read_csv("mealAmountData3.csv",nrows=50,header=None)
mealamount_data4 = pd.read_csv("mealAmountData4.csv",nrows=50,header=None)
mealamount_data5 = pd.read_csv("mealAmountData5.csv",nrows=50,header=None)


# In[4]:


#replacing NA values with mean
meal_data1.fillna(meal_data1.mean(axis=1),inplace=True) 
meal_data2.fillna(meal_data2.mean(axis=1),inplace=True)
meal_data3.fillna(meal_data3.mean(axis=1),inplace=True) 
meal_data4.fillna(meal_data4.mean(axis=1),inplace=True) 
meal_data5.fillna(meal_data5.mean(axis=1),inplace=True) 


# In[5]:


#calculate mean
p1_meal_mean=meal_data1.mean(axis=1)
p2_meal_mean=meal_data2.mean(axis=1)
p3_meal_mean=meal_data3.mean(axis=1)
p4_meal_mean=meal_data4.mean(axis=1)
p5_meal_mean=meal_data5.mean(axis=1)


# In[6]:


#calculate median
p1_meal_median=meal_data1.median(axis=1)
p2_meal_median=meal_data2.median(axis=1)
p3_meal_median=meal_data3.median(axis=1)
p4_meal_median=meal_data4.median(axis=1)
p5_meal_median=meal_data5.median(axis=1)


# In[7]:


#calculate standard deviation
p1_meal_std=meal_data1.std(axis=1)
p2_meal_std=meal_data2.std(axis=1)
p3_meal_std=meal_data3.std(axis=1)
p4_meal_std=meal_data4.std(axis=1)
p5_meal_std=meal_data5.std(axis=1)


# In[8]:


#caculate interquartile range
p1_meal_quantile =(meal_data1.quantile(0.75,axis=1)) - (meal_data1.quantile(0.25,axis=1))
p2_meal_quantile =(meal_data2.quantile(0.75,axis=1)) - (meal_data2.quantile(0.25,axis=1))
p3_meal_quantile =(meal_data3.quantile(0.75,axis=1)) - (meal_data3.quantile(0.25,axis=1))
p4_meal_quantile =(meal_data4.quantile(0.75,axis=1)) - (meal_data4.quantile(0.25,axis=1))
p5_meal_quantile =(meal_data5.quantile(0.75,axis=1)) - (meal_data5.quantile(0.25,axis=1))


# In[9]:


#calculate rms
p1_meal_rms=np.sqrt(np.mean(meal_data1**2,axis=1))
p2_meal_rms=np.sqrt(np.mean(meal_data2**2,axis=1))
p3_meal_rms=np.sqrt(np.mean(meal_data3**2,axis=1))
p4_meal_rms=np.sqrt(np.mean(meal_data4**2,axis=1))
p5_meal_rms=np.sqrt(np.mean(meal_data5**2,axis=1))


# In[10]:


#calculate mean absolute difference
p1_meal_mad = (meal_data1.diff(axis=1)).abs().mean(axis=1)
p2_meal_mad = (meal_data2.diff(axis=1)).abs().mean(axis=1)
p3_meal_mad = (meal_data3.diff(axis=1)).abs().mean(axis=1)
p4_meal_mad = (meal_data4.diff(axis=1)).abs().mean(axis=1)
p5_meal_mad = (meal_data5.diff(axis=1)).abs().mean(axis=1)


# In[11]:


#reshaping the features
p1_meal_mean= np.reshape(np.array(p1_meal_mean),(-1,1))
p1_meal_median=np.reshape(np.array(p1_meal_median),(-1,1))
p1_meal_std=np.reshape(np.array(p1_meal_std),(-1,1))
p1_meal_quantile=np.reshape(np.array(p1_meal_quantile),(-1,1))
p1_meal_rms=np.reshape(np.array(p1_meal_rms),(-1,1))
p1_meal_mad=np.reshape(np.array(p1_meal_mad),(-1,1))

p2_meal_mean= np.reshape(np.array(p2_meal_mean),(-1,1))
p2_meal_median=np.reshape(np.array(p2_meal_median),(-1,1))
p2_meal_std=np.reshape(np.array(p2_meal_std),(-1,1))
p2_meal_quantile=np.reshape(np.array(p2_meal_quantile),(-1,1))
p2_meal_rms=np.reshape(np.array(p2_meal_rms),(-1,1))
p2_meal_mad=np.reshape(np.array(p2_meal_mad),(-1,1))
                       
p3_meal_mean= np.reshape(np.array(p3_meal_mean),(-1,1))
p3_meal_median=np.reshape(np.array(p3_meal_median),(-1,1))
p3_meal_std=np.reshape(np.array(p3_meal_std),(-1,1))
p3_meal_quantile=np.reshape(np.array(p3_meal_quantile),(-1,1))
p3_meal_rms=np.reshape(np.array(p3_meal_rms),(-1,1))
p3_meal_mad=np.reshape(np.array(p3_meal_mad),(-1,1))
                       
p4_meal_mean= np.reshape(np.array(p4_meal_mean),(-1,1))
p4_meal_median=np.reshape(np.array(p4_meal_median),(-1,1))
p4_meal_std=np.reshape(np.array(p4_meal_std),(-1,1))
p4_meal_quantile=np.reshape(np.array(p4_meal_quantile),(-1,1))
p4_meal_rms=np.reshape(np.array(p4_meal_rms),(-1,1))
p4_meal_mad=np.reshape(np.array(p4_meal_mad),(-1,1))  
                       
p5_meal_mean= np.reshape(np.array(p5_meal_mean),(-1,1))
p5_meal_median=np.reshape(np.array(p5_meal_median),(-1,1))
p5_meal_std=np.reshape(np.array(p5_meal_std),(-1,1))
p5_meal_quantile=np.reshape(np.array(p5_meal_quantile),(-1,1))
p5_meal_rms=np.reshape(np.array(p5_meal_rms),(-1,1))
p5_meal_mad=np.reshape(np.array(p5_meal_mad),(-1,1))
                       


# In[12]:


#Perform PCA 
p1_meal_finalarr= np.hstack([p1_meal_mean,p1_meal_median,p1_meal_std,p1_meal_quantile,p1_meal_rms,p1_meal_mad])
p1_meal_df=pd.DataFrame(p1_meal_finalarr)
pca=PCA(n_components=4)
pca.fit(p1_meal_df)
trans_pca=pca.transform(p1_meal_df)
p1_meal_pca_df=pd.DataFrame(trans_pca)
p1_meal_pca_df.shape


# In[13]:


p2_meal_finalarr= np.hstack([p2_meal_mean,p2_meal_median,p2_meal_std,p2_meal_quantile,p2_meal_rms,p2_meal_mad])
p2_meal_df=pd.DataFrame(p2_meal_finalarr)
pca=PCA(n_components=4)
pca.fit(p2_meal_df)
trans_pca=pca.transform(p2_meal_df)
p2_meal_pca_df=pd.DataFrame(trans_pca)
p2_meal_pca_df.shape


# In[14]:


p3_meal_finalarr= np.hstack([p3_meal_mean,p3_meal_median,p3_meal_std,p3_meal_quantile,p3_meal_rms,p3_meal_mad])
p3_meal_df=pd.DataFrame(p3_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p3_meal_df)
trans_pca=pca.transform(p3_meal_df)
p3_meal_pca_df=pd.DataFrame(trans_pca)
p3_meal_pca_df.shape


# In[15]:


p4_meal_finalarr= np.hstack([p4_meal_mean,p4_meal_median,p4_meal_std,p4_meal_quantile,p4_meal_rms,p4_meal_mad])
p4_meal_df=pd.DataFrame(p4_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p4_meal_df)
trans_pca=pca.transform(p4_meal_df)
p4_meal_pca_df=pd.DataFrame(trans_pca)
p4_meal_pca_df.shape


# In[16]:


p5_meal_finalarr= np.hstack([p5_meal_mean,p5_meal_median,p5_meal_std,p5_meal_quantile,p5_meal_rms,p5_meal_mad])
p5_meal_df=pd.DataFrame(p5_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p5_meal_df)
trans_pca=pca.transform(p5_meal_df)
p5_meal_pca_df=pd.DataFrame(trans_pca)
p5_meal_pca_df.shape


# In[17]:


final_feature_matrix=np.vstack([p1_meal_pca_df,p2_meal_pca_df,p3_meal_pca_df,p4_meal_pca_df,p5_meal_pca_df])







# In[18]:


#K means clustering
scaler = StandardScaler()
final_feature_matrix_scaled = scaler.fit_transform(final_feature_matrix)
kmeans=KMeans(n_clusters=6,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(final_feature_matrix_scaled)
y_kmeans     


# In[19]:


#plotting the clusters
plt.scatter(final_feature_matrix_scaled[:,0],final_feature_matrix_scaled[:,1],c= y_kmeans,cmap='rainbow')


# In[20]:


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red',label='Centroids')
plt.title('K means clustering')
plt.legend()
plt.show()


# In[21]:


#dbscan clustering

dbscan = DBSCAN(eps=0.7, min_samples = 5,metric='euclidean')
clusters = dbscan.fit(final_feature_matrix_scaled)
labels=clusters.labels_
plt.scatter(final_feature_matrix_scaled[:, 0], final_feature_matrix_scaled[:, 1], c=labels, cmap="plasma")
plt.title('DBScan clustering')
plt.show()



# In[22]:

#bin allocation
mealamount_data=np.vstack([mealamount_data1,mealamount_data2,mealamount_data3,mealamount_data4,mealamount_data5])
mealamount_data_df=pd.DataFrame(mealamount_data)
bins={1:[],2:[],3:[],4:[],5:[],6:[]}

for (index_label, row_series) in mealamount_data_df.iterrows():
    if row_series.values==0:
        bins.setdefault(1,[])
        bins[1].append(index_label)
    elif row_series.values>0 and row_series.values<=20:
             bins.setdefault(2,[])
             bins[2].append(index_label)
    elif row_series.values>20 and row_series.values<=40:
            bins.setdefault(3,[])
            bins[3].append(index_label)
    elif row_series.values>40 and row_series.values<=60:
            bins.setdefault(4,[])
            bins[4].append(index_label)
    elif row_series.values>60 and row_series.values<=80:
            bins.setdefault(5,[])
            bins[5].append(index_label)
    elif row_series.values>80 and row_series.values<=100:
            bins.setdefault(6,[])
            bins[6].append(index_label)



 


# In[23]:


y_kmeans_df=pd.DataFrame(y_kmeans)
kmeans_clusters={1:[],2:[],3:[],4:[],5:[],6:[]}
for (index_label, row_series) in y_kmeans_df.iterrows():
    if row_series.values==0:
        kmeans_clusters.setdefault(1,[])
        kmeans_clusters[1].append(index_label)
    elif row_series.values==1:
            kmeans_clusters.setdefault(2,[])
            kmeans_clusters[2].append(index_label)
    elif row_series.values==2:
            kmeans_clusters.setdefault(3,[])
            kmeans_clusters[3].append(index_label)
    elif row_series.values==3:
            kmeans_clusters.setdefault(4,[])
            kmeans_clusters[4].append(index_label)
    elif row_series.values==4:
            kmeans_clusters.setdefault(5,[])
            kmeans_clusters[5].append(index_label)
    elif row_series.values==5:
            kmeans_clusters.setdefault(6,[])
            kmeans_clusters[6].append(index_label)
 


 


# In[24]:

#cluster bin mapping
index=[]
max_ln=-1                     
for i in range(1,7):
        length=len(set(kmeans_clusters[1]).intersection(bins[i]))
        if length > max_ln and i not in index :
            max_ln= length
            best_kmclus1_bin=i
index.append(best_kmclus1_bin)
max_ln=-1        
for i in range(1,7):
        length=len(set(kmeans_clusters[2]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_kmclus2_bin=i
index.append(best_kmclus2_bin)
            
max_ln=-1            
for i in range(1,7):
        length=len(set(kmeans_clusters[3]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_kmclus3_bin=i
index.append(best_kmclus3_bin)
            
max_ln=-1            
for i in range(1,7):
        length=len(set(kmeans_clusters[4]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_kmclus4_bin=i
index.append(best_kmclus4_bin)
            
max_ln=-1            
for i in range(1,7):
        length=len(set(kmeans_clusters[5]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_kmclus5_bin=i
index.append(best_kmclus5_bin)
max_ln=-1          
for i in range(1,7):
        length=len(set(kmeans_clusters[6]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_kmclus6_bin=i
index.append(best_kmclus6_bin)


      
           


# In[25]:


#final feature matrix with bins

final_ftr_grth = np.empty((0,5))
                                                                           
for i in kmeans_clusters[1]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_kmclus1_bin]),(-1,5)))
    final_ftr_grth = np.concatenate((final_ftr_grth, v), axis=0)
for i in kmeans_clusters[2]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_kmclus2_bin]),(-1,5)))
    final_ftr_grth = np.concatenate((final_ftr_grth, v), axis=0)
for i in kmeans_clusters[3]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_kmclus3_bin]),(-1,5)))
    final_ftr_grth = np.concatenate((final_ftr_grth, v), axis=0)
for i in kmeans_clusters[4]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_kmclus4_bin]),(-1,5)))
    final_ftr_grth = np.concatenate((final_ftr_grth, v), axis=0)
for i in kmeans_clusters[5]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_kmclus5_bin]),(-1,5)))
    final_ftr_grth = np.concatenate((final_ftr_grth, v), axis=0)
for i in kmeans_clusters[6]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_kmclus6_bin]),(-1,5)))
    final_ftr_grth = np.concatenate((final_ftr_grth, v), axis=0)

final_ftr_grth


# In[26]:

#k fold
kf=KFold(n_splits=5)

X=final_ftr_grth[:,0:4]
Y=final_ftr_grth[:,4:5]

for train_index,test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
y = y_train.ravel()
y_train = np.array(y).astype(int)
 


# In[27]:

#knn for k means
classifier = KNeighborsClassifier(n_neighbors=4,metric='euclidean',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)
acc = acc * 100



# In[28]:


print("Accuracy for k means clustering: ",acc)
print("Kmeans SSE: ", kmeans.inertia_)


# In[29]:

#saving the k means trained model
with open('knn_kmeans_trained_model','wb') as f:
    pickle.dump(classifier,f)


# In[30]:


labels_df=pd.DataFrame(labels)
dbscn_clusters={1:[],2:[],3:[],4:[],5:[],6:[]}
for (index_label, row_series) in labels_df.iterrows():
    if row_series.values==-1:
        dbscn_clusters.setdefault(1,[])
        dbscn_clusters[1].append(index_label)
    elif row_series.values==0:
            dbscn_clusters.setdefault(2,[])
            dbscn_clusters[2].append(index_label)
    elif row_series.values==1:
            dbscn_clusters.setdefault(3,[])
            dbscn_clusters[3].append(index_label)
    elif row_series.values==2:
            dbscn_clusters.setdefault(4,[])
            dbscn_clusters[4].append(index_label)
    elif row_series.values==3:
            dbscn_clusters.setdefault(5,[])
            dbscn_clusters[5].append(index_label)
    elif row_series.values==4:
            dbscn_clusters.setdefault(6,[])
            dbscn_clusters[6].append(index_label)

            


# In[31]:

#db scan cluster bin mapping
index=[]
max_ln=-1                     
for i in range(1,7):
        length=len(set(dbscn_clusters[1]).intersection(bins[i]))
        if length > max_ln and i not in index :
            max_ln= length
            best_dbscnclus1_bin=i
index.append(best_dbscnclus1_bin)
max_ln=-1        
for i in range(1,7):
        length=len(set(dbscn_clusters[2]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_dbscnclus2_bin=i
index.append(best_dbscnclus2_bin)
            
max_ln=-1            
for i in range(1,7):
        length=len(set(dbscn_clusters[3]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_dbscnclus3_bin=i
index.append(best_dbscnclus3_bin)
            
max_ln=-1            
for i in range(1,7):
        length=len(set(dbscn_clusters[4]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_dbscnclus4_bin=i
index.append(best_dbscnclus4_bin)
            
max_ln=-1            
for i in range(1,7):
        length=len(set(dbscn_clusters[5]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_dbscnclus5_bin=i
index.append(best_dbscnclus5_bin)
max_ln=-1          
for i in range(1,7):
        length=len(set(dbscn_clusters[6]).intersection(bins[i]))
        if length > max_ln and i not in index:
            max_ln= length
            best_dbscnclus6_bin=i
index.append(best_dbscnclus6_bin)



      
           


# In[32]:

#final feature matrix with dbscan clusters bins
final_ftr_grth_ds = np.empty((0,5))
                                                                           
for i in dbscn_clusters[1]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_dbscnclus1_bin]),(-1,5)))
    final_ftr_grth_ds = np.concatenate((final_ftr_grth_ds, v), axis=0)
for i in dbscn_clusters[2]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_dbscnclus2_bin]),(-1,5)))
    final_ftr_grth_ds = np.concatenate((final_ftr_grth_ds, v), axis=0)
for i in dbscn_clusters[3]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_dbscnclus3_bin]),(-1,5)))
    final_ftr_grth_ds = np.concatenate((final_ftr_grth_ds, v), axis=0)
for i in dbscn_clusters[4]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_dbscnclus4_bin]),(-1,5)))
    final_ftr_grth_ds = np.concatenate((final_ftr_grth_ds, v), axis=0)
for i in dbscn_clusters[5]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_dbscnclus5_bin]),(-1,5)))
    final_ftr_grth_ds = np.concatenate((final_ftr_grth_ds, v), axis=0)
for i in dbscn_clusters[6]:                     
    v=(np.reshape(np.hstack([final_feature_matrix[i],best_dbscnclus6_bin]),(-1,5)))
    final_ftr_grth_ds = np.concatenate((final_ftr_grth_ds, v), axis=0)



# In[33]:

#k fold split in train and test data
kf=KFold(n_splits=5)

X=final_ftr_grth_ds[:,0:4]
Y=final_ftr_grth_ds[:,4:5]

for train_index,test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
y = y_train.ravel()
y_train = np.array(y).astype(int)
 
#accuaracy of db scan
classifier = KNeighborsClassifier(n_neighbors=4,metric='euclidean',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)
acc = acc * 100

mse = metrics.mean_squared_error(y_test,y_pred)
sse=mse*len(y_pred)
print("Accuracy for db scan clustering: ",acc)
print("DB Scan SSE: ",sse)


# In[34]:

#saving the db scan trained model
with open('knn_dbscan_trained_model','wb') as f:
    pickle.dump(classifier,f)


# In[ ]:




