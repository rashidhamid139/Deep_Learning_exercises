#!/usr/bin/env python
# coding: utf-8

# In[53]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import time


# In[56]:


def timer(f):
    start = time.time()
    res =f()
    end = time.time()
    print("Fitting:{}".format(end-start))
    return res


# In[57]:


def build_model_for_data(data, target):
    X_train, X_test, y_train, y_test  = train_test_split(data, target, random_state=2)
    pipeline = make_pipeline(LinearRegression())
    model = timer(lambda:pipeline.fit(X_train, y_train))
    return X_test, y_test, model


# In[59]:


boston = load_boston()


# In[60]:


for x in boston['data'][0]:
    print(x)


# In[61]:


min_max = MinMaxScaler()
boston_min_max = min_max.fit_transform(boston['data'])


# In[62]:


for x in boston_min_max[0]:
    print(x)


# In[64]:


std = StandardScaler()
boston_std = std.fit_transform(boston['data'])


# In[65]:


for x in boston_std[0]:
    print(x)


# In[ ]:





# In[66]:


X_test, y_test, model = build_model_for_data(boston['data'], boston['target'])


# In[70]:


predictions = model.predict(X_test)
print("Mean_squared_error: {}".format(mean_squared_error(y_test, predictions)))


# In[71]:


X_test, y_test, model = build_model_for_data(boston_std, boston['target'])
predictions = model.predict(X_test)
print("MSE: {}".format(mean_squared_error(y_test, predictions)))


# In[72]:


X_test, y_test, model = build_model_for_data(boston_min_max, boston['target'])
predictions =model.predict(X_test)
print("MSE: {}".format(mean_squared_error(y_test, predictions)))


# In[ ]:




