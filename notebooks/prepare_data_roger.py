#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from joblib import dump
from src.data import make_dataset
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns


# # Set paths

# In[41]:


project_dir = Path.cwd().parent
data_dir = project_dir / 'data'
raw_data_dir = data_dir / 'raw'
interim_data_dir = data_dir / 'interim'
report_dir = project_dir / 'reports'


# # Download data

# In[5]:


competition = 'uts-advdsi-nba-career-prediction'
make_datasetdownload_data(competition=competition,
                          path=raw_data_dir,
                          unzip=True)


# # Load data

# In[6]:


df_train = pd.read_csv(raw_data_dir / 'train.csv')
df_train


# In[7]:


X_test = pd.read_csv(raw_data_dir / 'test.csv')
X_test


# In[8]:


df_train.describe()


# In[9]:


X_test.describe()


# # Profile Report

# In[40]:


profile_report = ProfileReport(df_train,
                               title='Raw data report',
                               explorative=True)
profile_report.to_file(report_dir / 'profile_report.html')


# # Check percentages

# In[45]:


# test
df_train[made_col] / df_train[attempt_col]


# In[52]:


# test
col_prefix = '3P'
made_col = f'{col_prefix} Made' if col_prefix == '3P' else f'{col_prefix}M'
attempt_col = f'{col_prefix}A'
percent_col = f'{col_prefix}%'
((df_train[made_col] / df_train[attempt_col]) - df_train[percent_col]/100).sum()


# In[63]:


# test
# df_train['3P Made'].loc[lambda x: x.isnull()]
# df_train['3PA'].loc[lambda x: x.isnull()]
df_train['3P%'].loc[lambda x: x.isnull()]


# In[69]:


((df_train[made_col] / df_train[attempt_col]) - df_train[percent_col]/100).loc[lambda x: x.isnull()]


# In[72]:


# test
col_prefix = '3P'
made_col = f'{col_prefix} Made' if col_prefix == '3P' else f'{col_prefix}M'
attempt_col = f'{col_prefix}A'
percent_col = f'{col_prefix}%'
df_train.loc[25, [made_col, attempt_col, percent_col]]


# In[77]:


for col_prefix in ['FG', '3P', 'FT']:
    print(col_prefix)
    made_col = f'{col_prefix} Made' if col_prefix == '3P' else f'{col_prefix}M'
    attempt_col = f'{col_prefix}A'
    percent_col = f'{col_prefix}%'
    
    diff = ((df_train[made_col] / df_train[attempt_col]) - df_train[percent_col]/100) 
    
    print(f'Number not equal {len(diff.loc[diff != 0])}')
    print(f'Number of zero attempts {len(df_train[attempt_col].loc[df_train[attempt_col] == 0])}')


# # Train test split

# In[10]:


target = 'TARGET_5Yrs'
X, y = make_dataset.separate_target(df_train, target=target)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)


# In[ ]:





# ## Save to interim

# In[11]:


np.save(interim_data_dir / 'X_train', X_train)
np.save(interim_data_dir / 'X_val', X_val)
np.save(interim_data_dir / 'y_train', y_train)
np.save(interim_data_dir / 'y_val', y_val)
np.save(interim_data_dir / 'X_test', X_test)


# # Standard Scaling

# In[12]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ## Save scaled data to interim

# In[13]:


np.save(interim_data_dir / 'X_train_scaled', X_train_scaled)
np.save(interim_data_dir / 'X_val_scaled', X_val_scaled)
np.save(interim_data_dir / 'X_test_scaled', X_test_scaled)


# # PCA

# In[16]:


pca = PCA()
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)


# In[38]:


## Save PCA data
np.save(interim_data_dir / 'X_train_pca', X_train_pca)
np.save(interim_data_dir / 'X_val_pca', X_val_pca)
np.save(interim_data_dir / 'X_test_pca', X_test_pca)


# In[37]:


plt.figure(figsize=(10,5))
ax = sns.scatterplot(x=range(1, len(pca.explained_variance_ratio_) + 1),
                     y=pca.explained_variance_ratio_,
                     color='Grey')


ax.set(title='PCA Scree Plot',
       xlabel='Component',
       ylabel='Explained Variance Ratio',
       xticks=range(1, len(pca.explained_variance_ratio_) + 1))

