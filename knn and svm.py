#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv("emails.csv")


# In[4]:


df


# In[7]:


df.drop('Email No.',axis=1,inplace=True)


# In[9]:


df.isnull().sum()


# In[11]:


df.dropna(axis=0,inplace=True)


# In[12]:


df.dtypes


# In[15]:


x=df.drop("Prediction",axis=1)


# In[16]:


y=df["Prediction"]


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=44)


# In[20]:


model=KNeighborsClassifier(n_neighbors=3)


# In[22]:


model.fit(x_train,y_train)


# In[23]:


y_pred=model.predict(x_test)


# In[25]:


from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error


# In[26]:


accuracy_score(y_test,y_pred)


# In[27]:


mean_squared_error(y_test,y_pred)


# In[28]:


mean_absolute_error(y_test,y_pred)


# In[29]:


from sklearn.svm import SVC


# In[30]:


model=SVC()


# In[31]:


model.fit(x_train,y_train)


# In[32]:


y_pred=model.predict(x_test)


# In[33]:


accuracy_score(y_test,y_pred)


# In[34]:


mean_squared_error(y_test,y_pred)


# In[35]:


mean_absolute_error(y_test,y_pred)


# In[ ]:




