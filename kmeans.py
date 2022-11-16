#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans


# In[2]:


df=pd.read_csv("sales_data_sample.csv",encoding="latin1")


# In[3]:


df


# In[4]:


x=df.iloc[:,[1,4]]
lis=[]


# In[5]:


for i in range(1,11):
    model=KMeans(n_clusters=i,init="k-means++",random_state=44)
    model.fit(x)
    lis.append(model.inertia_)
    
    


# In[6]:


import matplotlib.pyplot as plt
plt.plot(range(1,11),lis)
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.show()


# In[7]:


model=KMeans(n_clusters=3,init="k-means++",random_state=44)
y_pred=model.fit_predict(x)


# In[8]:


x=x.assign(pred=y_pred)


# In[9]:


x0=x[x.pred==0]
x1=x[x.pred==1]
x2=x[x.pred==2]


# In[10]:


plt.scatter(x0["QUANTITYORDERED"],x0["SALES"])
plt.scatter(x1["QUANTITYORDERED"],x1["SALES"])
plt.scatter(x2["QUANTITYORDERED"],x2["SALES"])
plt.xlabel("Order Quantity")
plt.ylabel("Sales")
plt.title("KMeans")
plt.show()


# In[ ]:




