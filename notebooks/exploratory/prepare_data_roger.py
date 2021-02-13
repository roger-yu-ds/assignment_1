#!/usr/bin/env python
# coding: utf-8

# In[39]:


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

# In[40]:


project_dir = Path.cwd().parent
data_dir = project_dir / 'data'
raw_data_dir = data_dir / 'raw'
interim_data_dir = data_dir / 'interim'
report_dir = project_dir / 'reports'


# # Download data

# In[41]:


competition = 'uts-advdsi-nba-career-prediction'
make_dataset.download_data(competition=competition,
                           path=raw_data_dir,
                           unzip=True)


# # Load data

# In[42]:


df_train = pd.read_csv(raw_data_dir / 'train.csv')
df_train


# In[43]:


X_test = pd.read_csv(raw_data_dir / 'test.csv')
X_test


# In[44]:


df_train.describe()


# In[45]:


X_test.describe()


# # Drop ID columns

# In[46]:


df_train.drop(columns=['Id_old', 'Id'], inplace=True)
X_test.drop(columns=['Id_old'], inplace=True)
test_id = X_test.pop('Id')


# # Profile Report
profile_report = ProfileReport(df_train,
                               title='Raw data report',
                               explorative=True)
profile_report.to_file(report_dir / 'profile_report.html')
# # Check percentages

# In[47]:


for col_prefix in ['FG', '3P', 'FT']:
    print(col_prefix)
    made_col = f'{col_prefix} Made' if col_prefix == '3P' else f'{col_prefix}M'
    attempt_col = f'{col_prefix}A'
    percent_col = f'{col_prefix}%'
    
    diff = ((df_train[made_col] / df_train[attempt_col]) - df_train[percent_col]/100) 
    
    print(f'Number not equal {len(diff.loc[diff != 0])}')
    print(f'Number of zero attempts {len(df_train[attempt_col].loc[df_train[attempt_col] == 0])}')


# # Train test split

# In[48]:


target = 'TARGET_5Yrs'
X, y = make_dataset.separate_target(df_train, target=target)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)


# ## Save to interim

# In[49]:


np.save(interim_data_dir / 'X_train', X_train)
np.save(interim_data_dir / 'X_val', X_val)
np.save(interim_data_dir / 'y_train', y_train)
np.save(interim_data_dir / 'y_val', y_val)
np.save(interim_data_dir / 'X_test', X_test)
test_id.to_csv(interim_data_dir / 'test_id.csv', index=False)


# # Standard Scaling

# In[50]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ## Save scaled data to interim

# In[51]:


np.save(interim_data_dir / 'X_train_scaled', X_train_scaled)
np.save(interim_data_dir / 'X_val_scaled', X_val_scaled)
np.save(interim_data_dir / 'X_test_scaled', X_test_scaled)


# # PCA

# In[52]:


pca = PCA()
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)


# In[53]:


## Save PCA data
np.save(interim_data_dir / 'X_train_pca', X_train_pca)
np.save(interim_data_dir / 'X_val_pca', X_val_pca)
np.save(interim_data_dir / 'X_test_pca', X_test_pca)


# In[54]:


plt.figure(figsize=(10,5))
ax = sns.scatterplot(x=range(1, len(pca.explained_variance_ratio_) + 1),
                     y=pca.explained_variance_ratio_,
                     color='Grey')


ax.set(title='PCA Scree Plot',
       xlabel='Component',
       ylabel='Explained Variance Ratio',
       xticks=range(1, len(pca.explained_variance_ratio_) + 1))

