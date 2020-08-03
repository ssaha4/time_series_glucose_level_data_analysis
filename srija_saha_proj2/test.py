#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA


# In[ ]:


#Taking the input file from user
inputfile = input("Enter the csv file:")
testdata=pd.read_csv(inputfile) 


# In[13]:



#dropping NA values
testdata.dropna(inplace=True)

#calculating the features
testdata_mean = testdata.mean(axis=1)
testdata_median = testdata.median(axis=1)
testdata_std = testdata.std(axis=1)
testdata_quantile =(testdata.quantile(0.75,axis=1)) - (testdata.quantile(0.25,axis=1))
testdata_rms=np.sqrt(np.mean(testdata**2,axis=1))
testdata_mad = (testdata.diff(axis=1)).abs().mean(axis=1)



# In[14]:


#reshaping the features
testdata_mean=np.reshape(np.array(testdata_mean),(-1,1))
testdata_median=np.reshape(np.array(testdata_median),(-1,1))
testdata_std=np.reshape(np.array(testdata_std),(-1,1))
testdata_quantile=np.reshape(np.array(testdata_quantile),(-1,1))
testdata_rms=np.reshape(np.array(testdata_rms),(-1,1))
testdata_mad=np.reshape(np.array(testdata_mad),(-1,1))


# In[15]:


#Combine all the features in an array
testdata_finalarr= np.hstack([testdata_mean,testdata_median,testdata_std,testdata_quantile,
                              testdata_rms,testdata_mad])
testdata_df=pd.DataFrame(testdata_finalarr)


# In[16]:


#Perform PCA 
pca=PCA(n_components=4)
pca.fit(testdata_df)
testdata_finalfeature=pca.transform(testdata_df)


# In[17]:


#loading the saved model and predicting the output
with open('trained_model','rb') as f:
  model = pickle.load(f)
predictions = model.predict(testdata_finalfeature)
print(predictions)

