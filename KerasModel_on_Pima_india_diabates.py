#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# In[5]:


dataset = loadtxt(r"C:\Users\rashid\Desktop\Datasets\pima-indians-diabetes.csv", delimiter=',')


# In[8]:


dataset.shape


# In[9]:


X = dataset[:, 0:8]


# In[13]:


Y = dataset[:,8]


# In[15]:


model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[16]:


model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[21]:


model.fit(X,Y, epochs=150, batch_size=10, verbose=1)


# In[1]:


predictions = model.predict(X)
# round predictions 
rounded = [round(x[0]) for x in predictions]
rounded


# In[ ]:




