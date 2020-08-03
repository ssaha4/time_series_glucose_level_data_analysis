#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA


# In[2]:



cgmdata1=pd.read_csv('CGMDatenumLunchPat1.csv')
cgmdata2=pd.read_csv('CGMDatenumLunchPat2.csv')
cgmdata3=pd.read_csv('CGMDatenumLunchPat3.csv')
cgmdata4=pd.read_csv('CGMDatenumLunchPat4.csv')
cgmdata5=pd.read_csv('CGMDatenumLunchPat5.csv')

cgmseriesdata1=pd.read_csv('CGMSeriesLunchPat1.csv')
cgmseriesdata2=pd.read_csv('CGMSeriesLunchPat2.csv')
cgmseriesdata3=pd.read_csv('CGMSeriesLunchPat3.csv')
cgmseriesdata4=pd.read_csv('CGMSeriesLunchPat4.csv')
cgmseriesdata5=pd.read_csv('CGMSeriesLunchPat5.csv')


# In[3]:


cgmdata1.head()


# In[4]:


cgmseriesdata1.info()


# In[5]:


plt.scatter(cgmdata1.iloc[0],cgmseriesdata1.iloc[0])


# In[6]:


mean_p1=cgmseriesdata1.mean(axis=1)
mean_p1


# In[7]:


mean_p2=cgmseriesdata2.mean(axis=1)
mean_p2


# In[8]:


mean_p3=cgmseriesdata3.mean(axis=1)
mean_p3


# In[9]:


mean_p4=cgmseriesdata4.mean(axis=1)
mean_p4


# In[10]:


mean_p5=cgmseriesdata5.mean(axis=1)
mean_p5


# In[11]:


y = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
y


# In[12]:


plt.figure(figsize=(4,4))
plt.xlabel('Timestamp') 
plt.ylabel('Mean')  
plt.title('Mean vs Timestamp for patient 1') 

plt.scatter(y,mean_p1)


# In[13]:


median_p1 = cgmseriesdata1.median(axis=1)
median_p1


# In[14]:


median_p2 = cgmseriesdata2.median(axis=1)
median_p2


# In[15]:


median_p3 = cgmseriesdata3.median(axis=1)
median_p3


# In[16]:


median_p4 = cgmseriesdata4.median(axis=1)
median_p4


# In[17]:


median_p5 = cgmseriesdata5.median(axis=1)
median_p5


# In[18]:


plt.figure(figsize=(8,8))
plt.xlabel('Timestamp') 
plt.ylabel('Median')  
plt.title('Median vs Timestamp for patient 1') 

plt.scatter(y,median_p1)


# In[19]:


std_p1 = cgmseriesdata1.std(axis=1)
std_p1.head()


# In[20]:


plt.figure(figsize=(4,4))
plt.xlabel('Timestamp') 
plt.ylabel('Standard Deviation')  
plt.title('Standard Deviation vs Timestamp for patient 1') 
plt.scatter(y,std_p1)


# In[21]:


std_p2  = cgmseriesdata2.std(axis=1)
std_p2.head()


# In[22]:


std_p3  = cgmseriesdata3.std(axis=1)
std_p3.head()


# In[23]:


std_p4  = cgmseriesdata4.std(axis=1)
std_p4.head()


# In[24]:


std_p5  = cgmseriesdata5.std(axis=1)
std_p5.head()


# In[25]:


q1_p1 = cgmseriesdata1.quantile(0.25,axis=1)
q1_p1.head()


# In[26]:


q3_p1 = cgmseriesdata1.quantile(0.75,axis=1)
q3_p1.head()


# In[27]:


iqr_p1 = q3_p1 - q1_p1
iqr_p1.head()


# In[28]:


plt.figure(figsize=(4,4))
plt.xlabel('Timestamp') 
plt.ylabel('First Quartile')  
plt.title('First Quartile vs Timestamp for patient 1') 
plt.scatter(y,iqr_p1)


# In[29]:


plt.figure(figsize=(4,4))
plt.xlabel('Timestamp') 
plt.ylabel('3rd Quartile')  
plt.title('Third Quartile vs Timestamp for patient 1') 
plt.scatter(y,iqr_p1)


# In[30]:


plt.figure(figsize=(4,4))
plt.xlabel('Timestamp') 
plt.ylabel('Interquartile Range')  
plt.title('Interquartile range vs Timestamp for patient 1') 
plt.scatter(y,iqr_p1)


# In[31]:


q1_p2 = cgmseriesdata2.quantile(0.25,axis=1)
q3_p2 = cgmseriesdata2.quantile(0.75,axis=1)
iqr_p2 = q3_p2 - q1_p2
iqr_p2.head()


# In[32]:


q1_p3 = cgmseriesdata3.quantile(0.25,axis=1)
q3_p3 = cgmseriesdata3.quantile(0.75,axis=1)
iqr_p3 = q3_p3 - q1_p3
iqr_p3.head()


# In[33]:


q1_p4 = cgmseriesdata4.quantile(0.25,axis=1)
q3_p4 = cgmseriesdata4.quantile(0.75,axis=1)
iqr_p4 = q3_p4 - q1_p4
iqr_p4.head()


# In[34]:


q1_p5 = cgmseriesdata5.quantile(0.25,axis=1)
q3_p5 = cgmseriesdata5.quantile(0.75,axis=1)
iqr_p5 = q3_p5 - q1_p5
iqr_p5.head()


# In[35]:


m=np.mean(cgmseriesdata1**2,axis=1)
rms_p1=np.sqrt(m)
rms_p1


# In[36]:


plt.figure(figsize=(4,4))
plt.xlabel('Timestamp') 
plt.ylabel('Root Mean Square')  
plt.title('Root Mean Square vs Timestamp for patient 1') 
plt.scatter(y,rms_p1)


# In[37]:


m=np.mean(cgmseriesdata2**2,axis=1)
rms_p2=np.sqrt(m)
rms_p2


# In[38]:


m=np.mean(cgmseriesdata3**2,axis=1)
rms_p3=np.sqrt(m)
rms_p3


# In[39]:


m=np.mean(cgmseriesdata4**2,axis=1)
rms_p4=np.sqrt(m)
rms_p4


# In[40]:


m=np.mean(cgmseriesdata5**2,axis=1)
rms_p5=np.sqrt(m)
rms_p5



# In[41]:


cgmseriesdata1.shape


# In[42]:


cor_p1 = cgmseriesdata1.corr()
cor_p1.shape


# In[43]:


plt.figure(figsize=(6,6))
sns.heatmap(cor_p1)


# In[44]:


cor_p2 = cgmseriesdata2.corr()
cor_p2.shape


# In[45]:


cor_p3 = cgmseriesdata3.corr()
cor_p3.shape


# In[46]:


cor_p4 = cgmseriesdata4.corr()
cor_p4.shape


# In[47]:


cor_p5 = cgmseriesdata5.corr()
cor_p5.shape


# In[48]:


cgmseriesdata1.shape


# In[49]:


mag_p1 = (cgmseriesdata1.diff(axis=1)).abs().mean(axis=1)
mag_p1


# In[50]:


plt.figure(figsize=(4,4))
plt.xlabel('Timestamp') 
plt.ylabel('Mean Absolute Deviation')  
plt.title('Mean Absolute Deviation vs Timestamp for patient 1') 
plt.scatter(y,mag_p1)


# In[51]:


mag_p2 = (cgmseriesdata2.diff(axis=1)).abs().mean(axis=1)
mag_p3 = (cgmseriesdata3.diff(axis=1)).abs().mean(axis=1)
mag_p4 = (cgmseriesdata4.diff(axis=1)).abs().mean(axis=1)
mag_p5 = (cgmseriesdata5.diff(axis=1)).abs().mean(axis=1)



# In[52]:



mean_p1= np.reshape(np.array(mean_p1),(-1,1))
median_p1=np.reshape(np.array(median_p1),(-1,1))
std_p1=np.reshape(np.array(std_p1),(-1,1))
q1_p1=np.reshape(np.array(q1_p1),(-1,1))
q3_p1=np.reshape(np.array(q3_p1),(-1,1))
iqr_p1=np.reshape(np.array(iqr_p1),(-1,1))
rms_p1=np.reshape(np.array(rms_p1),(-1,1))
mag_p1=np.reshape(np.array(mag_p1),(-1,1))
mag_p1.shape


# In[60]:


mean_p2= np.reshape(np.array(mean_p2),(-1,1))
median_p2=np.reshape(np.array(median_p2),(-1,1))
std_p2=np.reshape(np.array(std_p2),(-1,1))
q1_p2=np.reshape(np.array(q1_p2),(-1,1))
q3_p2=np.reshape(np.array(q3_p2),(-1,1))
iqr_p2=np.reshape(np.array(iqr_p2),(-1,1))
rms_p2=np.reshape(np.array(rms_p2),(-1,1))
mag_p2=np.reshape(np.array(mag_p2),(-1,1))


# In[61]:


mean_p3= np.reshape(np.array(mean_p3),(-1,1))
median_p3=np.reshape(np.array(median_p3),(-1,1))
std_p3=np.reshape(np.array(std_p3),(-1,1))
q1_p3=np.reshape(np.array(q1_p3),(-1,1))
q3_p3=np.reshape(np.array(q3_p3),(-1,1))
iqr_p3=np.reshape(np.array(iqr_p3),(-1,1))
rms_p3=np.reshape(np.array(rms_p3),(-1,1))
mag_p3=np.reshape(np.array(mag_p3),(-1,1))


# In[ ]:


mean_p4= np.reshape(np.array(mean_p4),(-1,1))
median_p4=np.reshape(np.array(median_p4),(-1,1))
std_p4=np.reshape(np.array(std_p4),(-1,1))
q1_p4=np.reshape(np.array(q1_p4),(-1,1))
q3_p4=np.reshape(np.array(q3_p4),(-1,1))
iqr_p4=np.reshape(np.array(iqr_p4),(-1,1))
rms_p4=np.reshape(np.array(rms_p4),(-1,1))
mag_p4=np.reshape(np.array(mag_p4),(-1,1))


# In[62]:


mean_p5= np.reshape(np.array(mean_p5),(-1,1))
median_p5=np.reshape(np.array(median_p5),(-1,1))
std_p5=np.reshape(np.array(std_p5),(-1,1))
q1_p5=np.reshape(np.array(q1_p5),(-1,1))
q3_p5=np.reshape(np.array(q3_p5),(-1,1))
iqr_p5=np.reshape(np.array(iqr_p5),(-1,1))
rms_p5=np.reshape(np.array(rms_p5),(-1,1))
mag_p5=np.reshape(np.array(mag_p5),(-1,1))


# In[53]:


finalarr_p1= np.hstack([mean_p1,median_p1,std_p1,q1_p1,q3_p1,iqr_p1,rms_p1,mag_p1])
df_p1=pd.DataFrame(finalarr_p1)
df_p1.to_csv("allfeatures.csv")
df_p1
                                                                               
                                                                               
 

                   


# In[67]:


finalarr_p2= np.hstack([mean_p2,median_p2,std_p2,q1_p2,q3_p2,iqr_p2,rms_p2,mag_p2])
df_p2=pd.DataFrame(finalarr_p2)
df_p2.to_csv("allfeatures_p2.csv")


# In[63]:


finalarr_p3= np.hstack([mean_p3,median_p3,std_p3,q1_p3,q3_p3,iqr_p3,rms_p3,mag_p3])
df_p3=pd.DataFrame(finalarr_p3)
df_p3.to_csv("allfeatures_p3.csv")


# In[ ]:


finalarr_p4= np.hstack([mean_p4,median_p4,std_p4,q1_p4,q3_p4,iqr_p4,rms_p4,mag_p4])
df_p4=pd.DataFrame(finalarr_p4)
df_p4.to_csv("allfeatures_p4.csv")


# In[64]:


finalarr_p5= np.hstack([mean_p5,median_p5,std_p5,q1_p5,q3_p5,iqr_p5,rms_p5,mag_p5])
df_p5=pd.DataFrame(finalarr_p5)
df_p5.to_csv("allfeatures_p5.csv")


# In[54]:


cgmseriesdata1.head()


# In[55]:


df_p1.shape


# In[56]:


pca=PCA(n_components=5)


# In[57]:


pca.fit(df_p1)


# In[70]:


trans_pca=pca.transform(df_p1)
pca_df_p1=pd.DataFrame(trans_pca)
pca_df_p1.to_csv("pcafeatures.csv")


# In[59]:


plt.figure(figsize=(6,6))
plt.xlabel('Timestamp') 
plt.ylabel('5 PCA components') 
  
plt.title('PCA Component vs Timestamp for patient 1') 
  

plt.scatter(y,trans_pca[ :,0],color = 'red')
plt.scatter(y,trans_pca[ :,1],color = 'green')
plt.scatter(y,trans_pca[ :,2],color = 'blue')
plt.scatter(y,trans_pca[ :,3],color = 'yellow')
plt.scatter(y,trans_pca[ :,4],color = 'aquamarine')
plt.show()

