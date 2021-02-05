#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('/Users/priya/Downloads/train.csv')


# In[3]:


df
print(df['TARGET_5Yrs'].value_counts())


# In[4]:


target = df.pop('TARGET_5Yrs')


# In[5]:


df.drop('Id_old', axis=1, inplace=True)
df.drop('Id', axis=1, inplace=True)


# In[6]:


df.columns = df.columns.str.strip()
df


# In[7]:


scaler = StandardScaler()


# In[8]:


df_cleaned = scaler.fit_transform(df)


# In[9]:


from imblearn.over_sampling import SMOTE 


# In[11]:


sm = SMOTE(random_state=42)

X_sm, y_sm = sm.fit_resample(df_cleaned, target)

print(f'''Shape of X before SMOTE: {df_cleaned.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=8)


# In[15]:


model = LogisticRegression(max_iter = 10000)
model.fit(X_train,y_train)


# In[16]:


y_pred = model.predict(X_test)


# In[17]:


model.score(X_test, y_test)


# In[18]:


model.predict_proba(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[21]:


roc=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
roc


# In[22]:


df1 = pd.read_csv('/Users/priya/Downloads/test.csv')
df1.drop('Id', axis=1, inplace=True)


# In[23]:


df1.drop('Id_old', axis=1, inplace=True)
df1


# In[24]:


df1_cleaned = scaler.fit_transform(df1)


# In[25]:


final = model.predict_proba(df1_cleaned)[:,1]


# In[26]:


pd.DataFrame(final).to_csv("/Users/priya/Downloads/final2.csv")


# In[ ]:




