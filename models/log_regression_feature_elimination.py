#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('/Users/priya/Downloads/train.csv')


# In[3]:



print(df['TARGET_5Yrs'].value_counts())


# In[4]:


print(df.head())
df.tail()


# In[5]:


target = df.pop('TARGET_5Yrs')


# In[6]:


df.drop('Id_old', axis=1, inplace=True)
df.drop('Id', axis=1, inplace=True)


# In[7]:


df.columns = df.columns.str.strip()
df


# In[8]:


df.info()


# In[9]:


print(df.describe())


# In[10]:


df.isnull().sum()


# In[11]:


duplicate = df.duplicated()
print(duplicate.sum())


# In[12]:


corr = df.corr()
plt.figure(figsize=(5,5))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[13]:


df_cleaned = df.drop(['FG%', '3P%', 'FT%'], axis = 1) 


# In[14]:


df_cleaned


# In[15]:


corr = df_cleaned.corr()
plt.figure(figsize=(5,5))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[16]:


scaler = StandardScaler()


# In[17]:


df_cleaned = scaler.fit_transform(df_cleaned)


# In[18]:


from imblearn.over_sampling import SMOTE 


# In[19]:


sm = SMOTE(random_state=42)

X_sm, y_sm = sm.fit_resample(df_cleaned, target)

print(f'''Shape of X before SMOTE: {df_cleaned.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=8)


# In[21]:


model = LogisticRegression(max_iter = 10000)
model.fit(X_train,y_train)


# In[22]:


y_pred = model.predict(X_test)


# In[23]:


model.score(X_test, y_test)


# In[24]:


model.predict_proba(X_test)


# In[25]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[27]:


roc=roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
roc


# In[39]:


df1 = pd.read_csv('/Users/priya/Downloads/test.csv')
df1.drop('Id', axis=1, inplace=True)
df1.drop('Id_old', axis=1, inplace=True)


# In[40]:


df2 = df1.drop(['FG%', '3P%', 'FT%'], axis = 1) 


# In[41]:



df2


# In[42]:


df1_cleaned = scaler.fit_transform(df2)


# In[43]:


df1_cleaned


# In[ ]:





# In[44]:


final = model.predict_proba(df1_cleaned)[:,1]


# In[45]:


df1_cleaned


# In[47]:


pd.DataFrame(final).to_csv("/Users/priya/Downloads/eliminated_var.csv")


# In[ ]:




