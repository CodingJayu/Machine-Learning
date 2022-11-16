#!/usr/bin/env python
# coding: utf-8

# In[10]:


def gradfun(x):
    return 2*x+6

def decentgrad(start,gradfunction,learning_rate,maxiter,tot=0.01):
    steps=[start]
    x=start
    
    for _ in range(maxiter):
        diff=learning_rate*gradfunction(x)
        if abs(diff)<tot:
            break

        x=x-diff
        steps.append(x)
    
    return x,len(steps),learning_rate,tot,steps

decentgrad(10,gradfun,0.03,23)


# In[ ]:




