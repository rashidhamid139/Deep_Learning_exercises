#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=12,10


# In[2]:


x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)
y = np.sin(x)+ np.random.normal(0,0.15, len(x))
data = pd.DataFrame(np.column_stack([x,y]), columns=['x', 'y'])


# In[3]:


plt.plot(data['x'], data['y'], '.')


# In[4]:


for i in range(2,16):
    colname = 'x_%d'%i
    data[colname] = data['x']**i


# In[5]:


data.head(3)


# In[6]:


from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
    
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors], data['y'])
    y_pred= linreg.predict(data[predictors])
    
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'], y_pred)
        plt.plot(data['x'], data['y'], '.')
        plt.title('Plot for Power: %d'%power)
    
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret


# In[ ]:





# In[7]:


col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}
for i in range(1, 16):
    coef_matrix_simple.iloc[i-1, 0:i+2]= linear_regression(data, power=i, models_to_plot=models_to_plot)


# In[8]:


coef_matrix_simple


# In[9]:


'''RidgeRegression: Performs L2 regularization: adds penalty equivalent to
square of the magnitude of coefficients
Minimization Objective=  Least Square Objective + alpha*(sum of square of coefficients)'''
'''Lasso Regression: Performs L1 Regularization: add penalty equivalent to absolute value
of the magnitude of coefficients
Minimization objective: Least Square Objective + alpha* (sum of absolute value of coefficients)
'''

'''Least square objective is basically objective function or cost function 
without reglarization term
'''


# In[ ]:





# In[10]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=12,10


# In[11]:


x = np.array([i*np.pi/180 for i in range(60, 300,4)])
np.random.seed(10)
y= np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]), columns=['x', 'y'])
plt.plot(data['x'], data['y'], '^')


# In[12]:


for i in range(2,16):
    colname = 'x_%d'%i
    data[colname]=data['x']**2


# In[13]:


data.head(3)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


#Import Linear Regression model from scikit-learn.
from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
    #initialize predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
    
    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for power: %d'%power)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret


# In[16]:


#Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)


# In[17]:


from sklearn.linear_model import Ridge


# In[18]:


def ridge_regression(data, predictors, alpha, models_to_plot={}):
    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(data[predictors], data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
        
        
    rss = sum((y_pred,-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret


# In[19]:


predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

alpha_ridge =[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5,10, 20]

col = ['rss', 'intercept_']+['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)
models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}

for i in range(10):
    coef_matrix_ridge.iloc[i,]=ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)


# In[ ]:


coef_matrix_ridge


# In[ ]:





# In[ ]:




