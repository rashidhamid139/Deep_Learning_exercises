#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


# In[3]:


breast = load_breast_cancer()
breast.keys()


# In[9]:


breast_data = breast.data
breast_data.shape


# In[10]:


breast_labels = breast.target 


# In[26]:


features = breast.feature_names
features.shape


# In[13]:


breast_labels.shape


# In[15]:


labels = np.reshape(breast_labels,(569,1))


# In[18]:


final_breast_data = np.concatenate([breast_data, labels], axis=1)


# In[20]:


final_breast_data.shape


# In[22]:


breast_dataset = pd.DataFrame(final_breast_data)


# In[23]:


breast_dataset.head()


# In[27]:


feature_labels = np.append(features, 'label')


# In[28]:


feature_labels.shape


# In[29]:


breast_dataset.columns = feature_labels


# In[32]:


breast_dataset.info()


# In[33]:


breast_dataset['label'].replace(0, 'Benign', inplace=True)


# In[37]:


breast_dataset['label'].value_counts()


# In[36]:


breast_dataset['label'].replace(1, 'Malignent', inplace=True)


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


x = breast_dataset.loc[:, features].values


# In[43]:


x.shape


# In[44]:


x = StandardScaler().fit_transform(x)


# In[49]:


x.shape[1]


# In[47]:


np.mean(x)


# In[48]:


np.std(x)


# In[50]:


feat_cols = ['feature'+str(i) for i in range(x.shape[1])]


# In[51]:


feat_cols


# In[52]:


normalised_breast = pd.DataFrame(x, columns=feat_cols)


# In[54]:


normalised_breast.head()


# In[57]:


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalcomponent_breast = pca_breast.fit_transform(x)


# In[58]:


principal_component_df = pd.DataFrame(principalcomponent_breast, columns=['PrincipalComp1', 'PrincipalComp2'])


# In[60]:


principal_component_df.head(2)


# In[61]:


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


# In[74]:


import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignent']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_component_df.loc[indicesToKeep, 'PrincipalComp1']
               , principal_component_df.loc[indicesToKeep, 'PrincipalComp2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# In[75]:


import numpy as np
from sklearn.manifold import TSNE
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])


# In[77]:


X.shape


# In[89]:


X_embedded = TSNE(n_components=2).fit_transform(X)


# In[82]:


X_embedded.fit_transform(X)


# In[90]:


X_embedded.shape


# In[ ]:




