#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle
import csv


# In[2]:


#reading the input files
meal_data1 = pd.read_csv('mealData1.csv',names=list(range(0,30)))
meal_data2 = pd.read_csv("mealData2.csv",names=list(range(0,30)))
meal_data3 = pd.read_csv("mealData3.csv",names=list(range(0,30)))
meal_data4 = pd.read_csv("mealData4.csv",names=list(range(0,30)))
meal_data5 = pd.read_csv("mealData5.csv",names=list(range(0,30)))


nomeal_data1= pd.read_csv("Nomeal1.csv")
nomeal_data2 = pd.read_csv("Nomeal1.csv")
nomeal_data3 = pd.read_csv("Nomeal1.csv")
nomeal_data4 = pd.read_csv("Nomeal1.csv")
nomeal_data5 = pd.read_csv("Nomeal1.csv")


# In[8]:



#removing NA values 
meal_data1.dropna(inplace=True) 
meal_data2.dropna(inplace = True) 
meal_data3.dropna(inplace = True) 
meal_data4.dropna(inplace = True) 
meal_data5.dropna(inplace = True) 

nomeal_data1.dropna(inplace = True) 
nomeal_data2.dropna(inplace = True) 
nomeal_data3.dropna(inplace = True) 
nomeal_data4.dropna(inplace = True) 
nomeal_data5.dropna(inplace = True) 


# In[19]:


#calculate mean
p1_meal_mean=meal_data1.mean(axis=1)
p2_meal_mean=meal_data2.mean(axis=1)
p3_meal_mean=meal_data3.mean(axis=1)
p4_meal_mean=meal_data4.mean(axis=1)
p5_meal_mean=meal_data5.mean(axis=1)
p1_nomeal_mean=nomeal_data1.mean(axis=1)
p2_nomeal_mean=nomeal_data2.mean(axis=1)
p3_nomeal_mean=nomeal_data3.mean(axis=1)
p4_nomeal_mean=nomeal_data4.mean(axis=1)
p5_nomeal_mean=nomeal_data5.mean(axis=1)


# In[25]:


#calculate median
p1_meal_median=meal_data1.median(axis=1)
p2_meal_median=meal_data2.median(axis=1)
p3_meal_median=meal_data3.median(axis=1)
p4_meal_median=meal_data4.median(axis=1)
p5_meal_median=meal_data5.median(axis=1)
p1_nomeal_median=nomeal_data1.median(axis=1)
p2_nomeal_median=nomeal_data2.median(axis=1)
p3_nomeal_median=nomeal_data3.median(axis=1)
p4_nomeal_median=nomeal_data4.median(axis=1)
p5_nomeal_median=nomeal_data5.median(axis=1)


# In[26]:


#calculate standard deviation
p1_meal_std=meal_data1.std(axis=1)
p2_meal_std=meal_data2.std(axis=1)
p3_meal_std=meal_data3.std(axis=1)
p4_meal_std=meal_data4.std(axis=1)
p5_meal_std=meal_data5.std(axis=1)
p1_nomeal_std=nomeal_data1.std(axis=1)
p2_nomeal_std=nomeal_data2.std(axis=1)
p3_nomeal_std=nomeal_data3.std(axis=1)
p4_nomeal_std=nomeal_data4.std(axis=1)
p5_nomeal_std=nomeal_data5.std(axis=1)


# In[27]:


#caculate interquartile range
p1_meal_quantile =(meal_data1.quantile(0.75,axis=1)) - (meal_data1.quantile(0.25,axis=1))
p2_meal_quantile =(meal_data2.quantile(0.75,axis=1)) - (meal_data2.quantile(0.25,axis=1))
p3_meal_quantile =(meal_data3.quantile(0.75,axis=1)) - (meal_data3.quantile(0.25,axis=1))
p4_meal_quantile =(meal_data4.quantile(0.75,axis=1)) - (meal_data4.quantile(0.25,axis=1))
p5_meal_quantile =(meal_data5.quantile(0.75,axis=1)) - (meal_data5.quantile(0.25,axis=1))

p1_nomeal_quantile =(nomeal_data1.quantile(0.75,axis=1)) - (nomeal_data1.quantile(0.25,axis=1))
p2_nomeal_quantile =(nomeal_data2.quantile(0.75,axis=1)) - (nomeal_data2.quantile(0.25,axis=1))
p3_nomeal_quantile =(nomeal_data3.quantile(0.75,axis=1)) - (nomeal_data3.quantile(0.25,axis=1))
p4_nomeal_quantile =(nomeal_data4.quantile(0.75,axis=1)) - (nomeal_data4.quantile(0.25,axis=1))
p5_nomeal_quantile =(nomeal_data5.quantile(0.75,axis=1)) - (nomeal_data5.quantile(0.25,axis=1))


# In[29]:



#calculate rms
p1_meal_rms=np.sqrt(np.mean(meal_data1**2,axis=1))
p2_meal_rms=np.sqrt(np.mean(meal_data2**2,axis=1))
p3_meal_rms=np.sqrt(np.mean(meal_data3**2,axis=1))
p4_meal_rms=np.sqrt(np.mean(meal_data4**2,axis=1))
p5_meal_rms=np.sqrt(np.mean(meal_data5**2,axis=1))

p1_nomeal_rms=np.sqrt(np.mean(nomeal_data1**2,axis=1))
p2_nomeal_rms=np.sqrt(np.mean(nomeal_data2**2,axis=1))
p3_nomeal_rms=np.sqrt(np.mean(nomeal_data3**2,axis=1))
p4_nomeal_rms=np.sqrt(np.mean(nomeal_data4**2,axis=1))
p5_nomeal_rms=np.sqrt(np.mean(nomeal_data5**2,axis=1))


# In[30]:


#calculate mean absolute difference
p1_meal_mad = (meal_data1.diff(axis=1)).abs().mean(axis=1)
p2_meal_mad = (meal_data2.diff(axis=1)).abs().mean(axis=1)
p3_meal_mad = (meal_data3.diff(axis=1)).abs().mean(axis=1)
p4_meal_mad = (meal_data4.diff(axis=1)).abs().mean(axis=1)
p5_meal_mad = (meal_data5.diff(axis=1)).abs().mean(axis=1)

p1_nomeal_mad = (nomeal_data1.diff(axis=1)).abs().mean(axis=1)
p2_nomeal_mad = (nomeal_data2.diff(axis=1)).abs().mean(axis=1)
p3_nomeal_mad = (nomeal_data3.diff(axis=1)).abs().mean(axis=1)
p4_nomeal_mad = (nomeal_data4.diff(axis=1)).abs().mean(axis=1)
p5_nomeal_mad = (nomeal_data5.diff(axis=1)).abs().mean(axis=1)


# In[32]:


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
                       
p1_nomeal_mean= np.reshape(np.array(p1_nomeal_mean),(-1,1))
p1_nomeal_median=np.reshape(np.array(p1_nomeal_median),(-1,1))
p1_nomeal_std=np.reshape(np.array(p1_nomeal_std),(-1,1))
p1_nomeal_quantile=np.reshape(np.array(p1_nomeal_quantile),(-1,1))
p1_nomeal_rms=np.reshape(np.array(p1_nomeal_rms),(-1,1))
p1_nomeal_mad=np.reshape(np.array(p1_nomeal_mad),(-1,1))
                         
p2_nomeal_mean= np.reshape(np.array(p2_nomeal_mean),(-1,1))
p2_nomeal_median=np.reshape(np.array(p2_nomeal_median),(-1,1))
p2_nomeal_std=np.reshape(np.array(p2_nomeal_std),(-1,1))
p2_nomeal_quantile=np.reshape(np.array(p2_nomeal_quantile),(-1,1))
p2_nomeal_rms=np.reshape(np.array(p2_nomeal_rms),(-1,1))
p2_nomeal_mad=np.reshape(np.array(p2_nomeal_mad),(-1,1))
                         
p3_nomeal_mean= np.reshape(np.array(p3_nomeal_mean),(-1,1))
p3_nomeal_median=np.reshape(np.array(p3_nomeal_median),(-1,1))
p3_nomeal_std=np.reshape(np.array(p3_nomeal_std),(-1,1))
p3_nomeal_quantile=np.reshape(np.array(p3_nomeal_quantile),(-1,1))
p3_nomeal_rms=np.reshape(np.array(p3_nomeal_rms),(-1,1))
p3_nomeal_mad=np.reshape(np.array(p3_nomeal_mad),(-1,1))
                    
p4_nomeal_mean= np.reshape(np.array(p4_nomeal_mean),(-1,1))
p4_nomeal_median=np.reshape(np.array(p4_nomeal_median),(-1,1))
p4_nomeal_std=np.reshape(np.array(p4_nomeal_std),(-1,1))
p4_nomeal_quantile=np.reshape(np.array(p4_nomeal_quantile),(-1,1))
p4_nomeal_rms=np.reshape(np.array(p4_nomeal_rms),(-1,1))
p4_nomeal_mad=np.reshape(np.array(p4_nomeal_mad),(-1,1))
            
p5_nomeal_mean= np.reshape(np.array(p5_nomeal_mean),(-1,1))
p5_nomeal_median=np.reshape(np.array(p5_nomeal_median),(-1,1))
p5_nomeal_std=np.reshape(np.array(p5_nomeal_std),(-1,1))
p5_nomeal_quantile=np.reshape(np.array(p5_nomeal_quantile),(-1,1))
p5_nomeal_rms=np.reshape(np.array(p5_nomeal_rms),(-1,1))
p5_nomeal_mad=np.reshape(np.array(p5_nomeal_mad),(-1,1))
                         


# In[34]:



#Perform PCA 
p1_meal_finalarr= np.hstack([p1_meal_mean,p1_meal_median,p1_meal_std,p1_meal_quantile,p1_meal_rms,p1_meal_mad])
p1_meal_df=pd.DataFrame(p1_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p1_meal_df)
trans_pca=pca.transform(p1_meal_df)
p1_meal_pca_df=pd.DataFrame(trans_pca)


# In[35]:


p2_meal_finalarr= np.hstack([p2_meal_mean,p2_meal_median,p2_meal_std,p2_meal_quantile,p2_meal_rms,p2_meal_mad])
p2_meal_df=pd.DataFrame(p2_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p2_meal_df)
trans_pca=pca.transform(p2_meal_df)
p2_meal_pca_df=pd.DataFrame(trans_pca)


# In[36]:


p3_meal_finalarr= np.hstack([p3_meal_mean,p3_meal_median,p3_meal_std,p3_meal_quantile,p3_meal_rms,p3_meal_mad])
p3_meal_df=pd.DataFrame(p3_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p3_meal_df)
trans_pca=pca.transform(p3_meal_df)
p3_meal_pca_df=pd.DataFrame(trans_pca)


# In[37]:


p4_meal_finalarr= np.hstack([p4_meal_mean,p4_meal_median,p4_meal_std,p4_meal_quantile,p4_meal_rms,p4_meal_mad])
p4_meal_df=pd.DataFrame(p4_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p4_meal_df)
trans_pca=pca.transform(p4_meal_df)
p4_meal_pca_df=pd.DataFrame(trans_pca)


# In[38]:


p5_meal_finalarr= np.hstack([p5_meal_mean,p5_meal_median,p5_meal_std,p5_meal_quantile,p5_meal_rms,p5_meal_mad])
p5_meal_df=pd.DataFrame(p5_meal_finalarr)

pca=PCA(n_components=4)
pca.fit(p5_meal_df)
trans_pca=pca.transform(p5_meal_df)
p5_meal_pca_df=pd.DataFrame(trans_pca)


# In[39]:


p1_nomeal_finalarr= np.hstack([p1_nomeal_mean,p1_nomeal_median,p1_nomeal_std,p1_nomeal_quantile,p1_nomeal_rms,p1_nomeal_mad])
p1_nomeal_df=pd.DataFrame(p1_nomeal_finalarr)

pca=PCA(n_components=4)
pca.fit(p1_nomeal_df)
trans_pca=pca.transform(p1_nomeal_df)
p1_nomeal_pca_df=pd.DataFrame(trans_pca)


# In[40]:


p2_nomeal_finalarr= np.hstack([p2_nomeal_mean,p2_nomeal_median,p2_nomeal_std,p2_nomeal_quantile,p2_nomeal_rms,p2_nomeal_mad])
p2_nomeal_df=pd.DataFrame(p2_nomeal_finalarr)

pca=PCA(n_components=4)
pca.fit(p2_nomeal_df)
trans_pca=pca.transform(p2_nomeal_df)
p2_nomeal_pca_df=pd.DataFrame(trans_pca)


# In[41]:


p3_nomeal_finalarr= np.hstack([p3_nomeal_mean,p3_nomeal_median,p3_nomeal_std,p3_nomeal_quantile,p3_nomeal_rms,p3_nomeal_mad])
p3_nomeal_df=pd.DataFrame(p3_nomeal_finalarr)

pca=PCA(n_components=4)
pca.fit(p3_nomeal_df)
trans_pca=pca.transform(p3_nomeal_df)
p3_nomeal_pca_df=pd.DataFrame(trans_pca)


# In[42]:


p4_nomeal_finalarr= np.hstack([p4_nomeal_mean,p4_nomeal_median,p4_nomeal_std,p4_nomeal_quantile,p4_nomeal_rms,p4_nomeal_mad])
p4_nomeal_df=pd.DataFrame(p4_nomeal_finalarr)

pca=PCA(n_components=4)
pca.fit(p4_nomeal_df)
trans_pca=pca.transform(p4_nomeal_df)
p4_nomeal_pca_df=pd.DataFrame(trans_pca)


# In[43]:


p5_nomeal_finalarr= np.hstack([p5_nomeal_mean,p5_nomeal_median,p5_nomeal_std,p5_nomeal_quantile,p5_nomeal_rms,p5_nomeal_mad])
p5_nomeal_df=pd.DataFrame(p5_nomeal_finalarr)

pca=PCA(n_components=4)
pca.fit(p5_nomeal_df)
trans_pca=pca.transform(p5_nomeal_df)
p5_nomeal_pca_df=pd.DataFrame(trans_pca)


# In[62]:


#Create Class label data
p1_meal_lbl = np.ones([43,1])
p2_meal_lbl = np.ones([37,1])
p3_meal_lbl = np.ones([43,1])
p4_meal_lbl = np.ones([48,1])
p5_meal_lbl = np.ones([45,1])

p1_nomeal_lbl = np.zeros([47,1])
p2_nomeal_lbl = np.zeros([47,1])
p3_nomeal_lbl = np.zeros([47,1])
p4_nomeal_lbl = np.zeros([47,1])
p5_nomeal_lbl = np.zeros([47,1])


# In[64]:


#Concatenate features and class label data
p1_meal_ftr_lbl=np.hstack([p1_meal_pca_df,p1_meal_lbl])
p2_meal_ftr_lbl=np.hstack([p2_meal_pca_df,p2_meal_lbl])
p3_meal_ftr_lbl=np.hstack([p3_meal_pca_df,p3_meal_lbl])
p4_meal_ftr_lbl=np.hstack([p4_meal_pca_df,p4_meal_lbl])
p5_meal_ftr_lbl=np.hstack([p5_meal_pca_df,p5_meal_lbl])

p1_nomeal_ftr_lbl=np.hstack([p1_nomeal_pca_df,p1_nomeal_lbl])
p2_nomeal_ftr_lbl=np.hstack([p2_nomeal_pca_df,p2_nomeal_lbl])
p3_nomeal_ftr_lbl=np.hstack([p3_nomeal_pca_df,p3_nomeal_lbl])
p4_nomeal_ftr_lbl=np.hstack([p4_nomeal_pca_df,p4_nomeal_lbl])
p5_nomeal_ftr_lbl=np.hstack([p5_nomeal_pca_df,p5_nomeal_lbl])


# In[65]:


#final feature matrix
final_feature_matrix=np.vstack([p1_meal_ftr_lbl,p1_nomeal_ftr_lbl,p2_meal_ftr_lbl,p2_nomeal_ftr_lbl,
                                p3_meal_ftr_lbl,p3_nomeal_ftr_lbl,p4_meal_ftr_lbl,p4_nomeal_ftr_lbl,
                                p5_meal_ftr_lbl,p5_nomeal_ftr_lbl])
final_feature_matrix.shape


# In[50]:


# Perform k fold to split training and test data
kf=KFold(n_splits=5)
kf


# In[51]:


X=final_feature_matrix[:,0:4]
Y=final_feature_matrix[:,4:5]


# In[63]:


for train_index,test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    


# In[66]:


y = y_train.ravel()
y_train = np.array(y).astype(int)


# In[57]:


#Logical Regression
model=LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
acc = metrics.accuracy_score(y_test,predictions)
acc = acc * 100
acc


# In[58]:


#Naive Bayes
model=GaussianNB()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
acc = metrics.accuracy_score(y_test,predictions)
acc = acc * 100
acc


# In[67]:


#Random Forest
model=RandomForestClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
acc = metrics.accuracy_score(y_test,predictions)
acc = acc * 100
acc


# In[68]:


#Decision Tree
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
acc = metrics.accuracy_score(y_test,predictions)
acc = acc * 100

recall = metrics.recall_score(y_test,predictions)
prec = metrics.precision_score(y_test,predictions)
f1_score = metrics.f1_score(y_test,predictions)
print("Accuracy: ",acc)
print("Recall: ",recall)
print("Precision: ",prec)
print("F1 score: ",f1_score)


# In[69]:


#Saved the decision tree model as it is giving me highest accuracy

with open('trained_model','wb') as f:
    pickle.dump(model,f)

