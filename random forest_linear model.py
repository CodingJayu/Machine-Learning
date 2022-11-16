#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd


# In[37]:


df=pd.read_csv("uber.csv")


# In[38]:


df.columns


# In[39]:


df.head()


# In[40]:


df.drop(['Unnamed: 0','key'],axis=1,inplace=True)


# In[41]:


df.describe()


# In[42]:


df.isnull().sum()


# In[43]:


df.info()


# In[44]:


df.dropna(axis=0,inplace=True)


# In[45]:


df.isnull().sum()


# In[46]:


df.dtypes


# In[47]:


df["pickup_datetime"]=pd.to_datetime(df["pickup_datetime"])


# In[48]:


df=df.assign(day=df["pickup_datetime"].dt.day,month=df["pickup_datetime"].dt.month,hour=df["pickup_datetime"].dt.hour,year=df["pickup_datetime"].dt.year)


# In[49]:


df.drop("pickup_datetime",axis=1,inplace=True)


# In[50]:


df


# In[51]:


x=df.drop("fare_amount",axis=1)


# In[52]:


y=df["fare_amount"]


# In[53]:


from sklearn.model_selection import train_test_split 


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=45)


# In[55]:


from sklearn.ensemble import RandomForestRegressor


# In[56]:


model=RandomForestRegressor()


# In[57]:


model.fit(x_train,y_train)


# In[62]:


y_pred=model.predict(x_test).ravel()


# In[64]:


y_pred,y_test


# In[60]:


from sklearn.metrics import mean_absolute_error,mean_squared_error 


# In[66]:


mean_squared_error(y_test,y_pred)


# In[68]:


mean_absolute_error(y_test,y_pred)


# In[70]:


import numpy as np
np.sqrt(mean_squared_error(y_test,y_pred))


# In[71]:


from sklearn.linear_model import LinearRegression


# In[72]:


model=LinearRegression()


# In[73]:


model.fit(x_train,y_train)


# In[74]:


y_pred=model.predict(x_test)


# In[75]:


mean_squared_error(y_test,y_pred)


# In[76]:


mean_absolute_error(y_test,y_pred)


# In[81]:


np.sqrt(mean_squared_error(y_pred,y_test))


# In[ ]:




