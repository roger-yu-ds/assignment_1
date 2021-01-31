#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from scipy import stats
from joblib import dump
from joblib import load
import xgboost as xgb
import matplotlib.pyplot as plt


# In[3]:


project_dir = Path.cwd().parent
data_dir = project_dir / 'data'
raw_data_dir = data_dir / 'raw'
interim_data_dir = data_dir / 'interim'


# # Load data

# In[34]:


df_train = pd.read_csv(raw_data_dir / 'train.csv')
df_test = pd.read_csv(raw_data_dir / 'test.csv')
X_train = np.load(interim_data_dir / 'X_train.npy')
X_val = np.load(interim_data_dir / 'X_val.npy')
y_train = np.load(interim_data_dir / 'y_train.npy')
y_val = np.load(interim_data_dir / 'y_val.npy')
X_test = np.load(interim_data_dir / 'X_test.npy')


# # Baseline
# 
# The base line prediction is simply to make them all negative.

# In[19]:


labels = 'Positive', 'Negative'
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
sizes = [pos_count, neg_count]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[22]:


labels = 'Positive', 'Negative'
pos_count = (y_val == 1).sum()
neg_count = (y_val == 0).sum()
sizes = [pos_count, neg_count]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[21]:


preds = [1] * len(y_val)
roc_auc_score(y_val, preds)


# # XGB

# In[23]:


clf_xgb = xgb.XGBClassifier()


# In[24]:


clf_xgb.fit(X_train, y_train)


# In[26]:


preds = clf_xgb.predict(X_val)
probs = clf_xgb.predict_proba(X_val)


# In[32]:


X_val.shape


# In[35]:


len(y_val)


# In[39]:


auc = roc_auc_score(y_val, probs[:, 1])
tpr, fpr, threshold = roc_curve(y_val, probs[:, 1])
auc


# In[ ]:





# # RandomizedSearchCV

# In[35]:


# test
df_train.info()


# In[38]:


pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        ))
])

param_dist = {
    'pca__n_components': stats.randint(1, X_train.shape[1]),
    'classifier__n_estimators': stats.randint(150, 1000),
    'classifier__learning_rate': stats.uniform(0.01, 0.6),
    'classifier__subsample': stats.uniform(0.3, 0.9),
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9],
    'classifier__colsample_bytree': stats.uniform(0.5, 0.9),
    'classifier__min_child_weight': [1, 2, 3, 4]
}

cv = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    random_state=42,
    n_iter=5,
    cv=5,
    n_jobs=-1
)

cv.fit(X_train, y_train)


# In[41]:


pd.DataFrame(cv.cv_results_)


# ## Predictions

# In[48]:


preds = cv.predict(X_val)
probs = cv.predict_proba(X_val)


# In[47]:


len(probs[:, 1])


# In[49]:


fpr, tpr, thresholds = roc_curve(y_val, probs[:, 1])
roc_auc_score(y_val, probs[:, 1])


# In[ ]:




