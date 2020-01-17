#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[2]:


print(boston.keys())


# In[3]:


print(boston.DESCR)


# In[4]:


import pandas as pd
df = pd.DataFrame(boston.data) # load numerical data'
df.columns = boston.feature_names # set column names
df['MEDV'] = boston.target # define target as median house price


# In[5]:


df.head(6)


# In[6]:


print("num samples=",len(df.index),", num attributes=",len(df.columns))


# In[7]:


import numpy as np
y=np.array(df['MEDV'])


# In[8]:


print("The mean house price is ",round(np.mean(y), 2)," thousands of dollars.")
percent=100*(np.sum(y>40)/len(y))
print("Only ",round(percent,1)," percent are above $40k.")


# In[9]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df2 = df['RM']
x=np.array(df2)


# In[11]:


plt.plot(x,y,'o')
plt.xlabel('Average Number Of Rooms')
plt.ylabel('Average Price')
plt.grid(True)


# In[12]:


def fit_linear(x,y):
    """
    Given vectors of data points (x,y), performs a fit for the linear model:
       yhat = beta0 + beta1*x, 
    The function returns beta0, beta1 and rsq, where rsq is the coefficient of determination.
    """
xm = np.mean(x)
ym = np.mean(y)
sxx = np.mean((x-xm)**2)
syy = np.mean((y-ym)**2)
sxy = np.mean((y-ym)*(x-xm))

beta1 = sxy/sxx
beta0 = ym-xm*beta1
rsq = sxy/(np.sqrt(syy)*np.sqrt(sxx))
print(beta0, beta1, rsq)


# In[17]:


xp = np.array([4,9])          
yp = beta1*xp + beta0

plt.plot(x,y,'o')
plt.plot(xp,yp,'-',linewidth=3)
plt.xlabel('Average Number Of Rooms')
plt.ylabel('Average Price')
plt.grid(True)


# In[ ]:




