#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# 

# In[2]:


df=pd.read_csv("diabetes.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


x=df.drop("Outcome",axis=1)


# In[8]:


y=df["Outcome"]


# In[9]:


x.shape


# In[10]:


y.shape


# In[11]:


from sklearn.model_selection import train_test_split 


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[13]:


x_train.shape


# In[14]:


y_train.shape


# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# In[16]:


model=KNeighborsClassifier(n_neighbors=3)


# In[17]:


model.fit(x_train,y_train)


# In[18]:


y_pred=model.predict(x_test)


# In[19]:


from sklearn.metrics import confusion_matrix


# In[23]:


tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()


# In[26]:


accuracy=(tp+tn)/(tp+tn+fp+fn)


# In[27]:


accuracy


# In[30]:


prec=tp/(tp+fp)


# In[31]:


prec


# In[32]:


recall=tp/(tp+fn)


# In[33]:


recall


# In[37]:


from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error


# In[36]:


accuracy_score(y_test,y_pred)


# In[38]:


mean_squared_error(y_test,y_pred)


# In[39]:


mean_absolute_error(y_test,y_pred)


# In[ ]:




