#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# In[11]:


df = pd.read_csv('/Users/priya/Downloads/train.csv')


# In[12]:


df
print(df['TARGET_5Yrs'].value_counts())


# In[13]:


target = df.pop('TARGET_5Yrs')


# In[14]:


df


# In[15]:


df.drop('Id_old', axis=1, inplace=True)
df.drop('Id', axis=1, inplace=True)


# In[16]:


df.columns = df.columns.str.strip()
df


# In[17]:


target


# In[18]:


scaler = StandardScaler()


# In[19]:


df_cleaned = scaler.fit_transform(df)


# In[20]:


df_cleaned


# In[21]:


#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[22]:


# Make an instance of the Model
#pca = PCA(.95)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(df_cleaned, target, test_size=0.2, random_state=8)


# In[24]:


#pca.fit(X_train)


# In[25]:


#pca.n_components_


# In[26]:


#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)


# In[27]:


#logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[28]:


#logisticRegr.fit(X_train, y_train)


# In[29]:


#logisticRegr.predict(y_test)


# In[30]:


model = LogisticRegression(max_iter = 10000)
model.fit(X_train,y_train)


# In[31]:


y_pred = model.predict(X_test)


# In[32]:


model.score(X_test, y_test)


# In[33]:


model.predict_proba(X_test)


# In[34]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[35]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[36]:


roc=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
roc


# In[37]:


df1 = pd.read_csv('/Users/priya/Downloads/test.csv')
df1.drop('Id', axis=1, inplace=True)


# In[39]:


df1.drop('Id_old', axis=1, inplace=True)
df1


# In[40]:


df1_cleaned = scaler.fit_transform(df1)


# In[41]:


#pca = PCA(.95)


# In[42]:


#v = model.predict_proba(df1)[:,1]

#pca.fit(df1_cleaned)
#pca.n_components_


# In[43]:


#v = pca.transform(df1_cleaned)


# In[44]:


final = model.predict_proba(df1_cleaned)[:,1]


# In[45]:


final


# In[46]:


pd.DataFrame(final).to_csv("/Users/priya/Downloads/final1.csv")


# In[ ]:




