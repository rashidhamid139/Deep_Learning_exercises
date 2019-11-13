#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.chdir("c://Users/rashid/Desktop/Datasets/")


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('housepricedata.csv')


# In[5]:


df.head(4)


# In[6]:


dataset = df.values


# In[8]:


dataset.shape


# In[9]:


X = dataset[:, 0:10]


# In[10]:


X.shape


# In[11]:


Y = dataset[:, 10]


# In[13]:


Y.shape


# In[14]:


from sklearn import preprocessing


# In[15]:


min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(X)


# In[26]:


X_scale.shape


# In[27]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)


# In[34]:


X_train.shape, X_val_and_test.shape


# In[36]:


Y_train.shape, Y_val_and_test.shape


# In[37]:


X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test,
                                                Y_val_and_test, test_size=0.5)


# In[38]:


X_val.shape, X_test.shape


# In[39]:


Y_val.shape, Y_test.shape


# In[42]:


from keras.models import Sequential
from keras.layers import Dense


# In[43]:


model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[44]:


model.summary()


# In[45]:


model.compile(optimizer='sgd', loss= 'binary_crossentropy', metrics=['accuracy'])


# In[49]:


hist = model.fit(X_train, Y_train, batch_size=64, epochs=200, validation_data=(X_val, Y_val))


# In[54]:


model.evaluate(X_test, Y_test)


# In[ ]:





# In[55]:


import matplotlib.pyplot as plt


# In[56]:


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[58]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model acc')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[61]:


model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[62]:


hist_2 = model_2.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))


# In[64]:


plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[66]:


plt.plot(hist_2.history['acc'])
plt.plot(hist_2.history['val_acc'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[67]:


from keras.layers import Dropout
from keras import regularizers


# In[68]:


model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), 
         input_shape=(10,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])


# In[73]:


model_3.compile(optimizer='adam', loss='binary_crossentropy', 
             metrics=['accuracy'])


# In[74]:


hist_3= model_3.fit(X_train, Y_train,
                   batch_size=32, epochs=100,
                   validation_data=(X_val, Y_val))


# In[75]:


plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()


# In[76]:


plt.plot(hist_3.history['acc'])
plt.plot(hist_3.history['val_acc'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()


# In[ ]:




