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
import random
import pickle




#Taking the input file from user
inputfile = input("Enter the csv file:")
testdata=pd.read_csv(inputfile,header=None) 



#replacing NA values
testdata.fillna(testdata.mean(axis=1),inplace=True) 

#calculating the features
testdata_mean = testdata.mean(axis=1)
testdata_median = testdata.median(axis=1)
testdata_std = testdata.std(axis=1)
testdata_quantile =(testdata.quantile(0.75,axis=1)) - (testdata.quantile(0.25,axis=1))
testdata_rms=np.sqrt(np.mean(testdata**2,axis=1))
testdata_mad = (testdata.diff(axis=1)).abs().mean(axis=1)





#reshaping the features
testdata_mean=np.reshape(np.array(testdata_mean),(-1,1))
testdata_median=np.reshape(np.array(testdata_median),(-1,1))
testdata_std=np.reshape(np.array(testdata_std),(-1,1))
testdata_quantile=np.reshape(np.array(testdata_quantile),(-1,1))
testdata_rms=np.reshape(np.array(testdata_rms),(-1,1))
testdata_mad=np.reshape(np.array(testdata_mad),(-1,1))





#Combine all the features in an array
testdata_finalarr= np.hstack([testdata_mean,testdata_median,testdata_std,testdata_quantile,
                              testdata_rms,testdata_mad])
testdata_df=pd.DataFrame(testdata_finalarr)




#Perform PCA 
pca=PCA(n_components=4)
pca.fit(testdata_df)
testdata_finalfeature=pca.transform(testdata_df)




#loading the saved kmeans model and predicting the output
with open('knn_kmeans_trained_model','rb') as f:
    model = pickle.load(f)
kmean_predictions = model.predict(testdata_finalfeature)
print("Kmeans_labels:",kmean_predictions)

#loading the saved dbsacn model and predicting the output
with open('knn_dbscan_trained_model','rb') as f:
    model1 = pickle.load(f)
dbscan_predictions = model1.predict(testdata_finalfeature)
print("dbscan_labels:",dbscan_predictions)



# In[2]:

#output dataframe with db scan labels and k means labels
output_df=pd.DataFrame({'DBSCAN':dbscan_predictions,'KMEANS':kmean_predictions,})
print(output_df)
    


# In[4]:

#saving the csv file with db scan and k means labels
output_df.to_csv('output.csv',header=False,index=False) 


# In[ ]:




