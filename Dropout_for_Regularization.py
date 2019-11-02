#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

import os
os.chdir("c:////Users/rashid/Desktop/Machine_Learning_Models")


# In[43]:


dataframe = pd.read_csv("sonar.csv", names=['V'+str(i)for i in range(61)])


# In[44]:


dataframe.head()


# In[18]:


dataframe['V60'].value_counts()


# In[20]:


dataframe.head()


# In[21]:


dataframe.info()


# In[45]:


dataset = dataframe.values


# In[46]:


dataset.shape


# In[47]:


X= dataset[:, 0:60]


# In[48]:


X.shape


# In[49]:


Y= dataset[:, 60]


# In[50]:


Y.shape


# In[52]:


encoder=LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# In[55]:


encoded_Y


# In[76]:


def create_baseline():
    model = Sequential()
    model.add(Dense(60, input_dim=60, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    
    sgd = SGD(lr=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[77]:


create_baseline().summary()


# In[78]:


estimators=[]
estimators.append(('Standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn = create_baseline, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print(results.mean()*100)
print(results.std()*100)


# In[74]:


#Example Of Dropout on the dataset 
#Before Input Layer


# In[81]:


from keras.constraints import maxnorm
from keras.layers import Dropout


# In[90]:


def create_model():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(1,activation='sigmoid'))
    
    sgd = SGD(lr=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[91]:


create_model().summary()


# In[96]:


estimators=[]
estimators.append(('Standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Mean: ", results.mean()*100)
print("SD: ", results.std()*100)


# In[ ]:


#Model with dropout layer in between the Hidden Layers


# In[ ]:


def create_model2():
    model = Sequential()
    model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))

    sgd = SGD(lr=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[108]:


create_model2().summary()


# In[107]:


estimators=[]
estimators.append(('Standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model2, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Mean: ", results.mean()*100)
print("SD: ", results.std()*100)


# In[109]:


os.getcwd()


# In[ ]:




